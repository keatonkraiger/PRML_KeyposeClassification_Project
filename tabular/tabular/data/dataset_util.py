import glob
import os
import re

import numpy as np
import tabular.data.data_operations as norms


_MODAL_STEM_ALIASES = {
    'mocap': 'mocap_3d',
    'mocap_3d': 'mocap_3d',
    'pressure': 'pressure',
    'body25': 'body25_v1',
    'body25_v1': 'body25_v1',
    'body25_v2': 'body25_v2',
}


def _extract_subject_id(subject_dir):
    name = os.path.basename(subject_dir.rstrip(os.sep))
    match = re.search(r'(\d+)$', name)
    if match is None:
        return None
    return int(match.group(1))


def _extract_take_id(npy_file, stem):
    basename = os.path.basename(npy_file)
    prefix = f"{stem}_"
    if not basename.startswith(prefix) or not basename.endswith('.npy'):
        return None
    take_str = basename[len(prefix):-4]
    if not take_str.isdigit():
        return None
    return int(take_str)


def _resolve_modal_stem(modal_name, cfg):
    modal_name = str(modal_name)
    data_cfg = cfg['data']
    modal_cfg = data_cfg[modal_name]

    file_stem = modal_cfg.get('file_stem')
    if not file_stem:
        legacy_modal = modal_cfg.get('modal')
        if legacy_modal:
            file_stem = str(legacy_modal).lower()
        else:
            file_stem = _MODAL_STEM_ALIASES.get(modal_name.lower(), modal_name.lower())

    return file_stem


def _get_modal_norm(modal_name, cfg):
    data_cfg = cfg['data']
    modal_cfg = data_cfg.get(modal_name)
    if isinstance(modal_cfg, dict):
        return modal_cfg.get('norm')
    return None


def create_datasets(cfg):
    """Build per-frame tabular datasets from the Taiji folder layout.

    Returns frame indices, (subject, take) info, and one flat feature
    matrix per requested modality.
    """
    data_modals = cfg['data']['modals']
    data_dir = cfg['default']['data_path']
    allowed_modals = {'pressure', 'mocap_3d'}
    unknown_modals = [m for m in data_modals if m not in allowed_modals]
    if unknown_modals:
        raise ValueError(
            f"Unsupported modality(ies): {unknown_modals}. "
            "This student pipeline supports only ['pressure', 'mocap_3d']."
        )

    subject_dirs = sorted(
        d for d in glob.glob(os.path.join(data_dir, 'Subject*')) if os.path.isdir(d)
    )
    if not subject_dirs:
        raise ValueError(f"No subject folders found in {data_dir}")

    per_modal_rows = [[] for _ in data_modals]
    all_frames = []
    all_subinfo = []

    for subject_dir in subject_dirs:
        subject = _extract_subject_id(subject_dir)
        if subject is None:
            continue

        per_modal_take_files = {}
        for modal in data_modals:
            stem = _resolve_modal_stem(modal, cfg)
            files = sorted(glob.glob(os.path.join(subject_dir, f"{stem}_*.npy")))
            take_files = {}
            for file_path in files:
                take = _extract_take_id(file_path, stem)
                if take is not None:
                    take_files[take] = file_path
            per_modal_take_files[modal] = take_files

        take_sets = [set(v.keys()) for v in per_modal_take_files.values() if len(v) > 0]
        if not take_sets:
            continue
        common_takes = sorted(set.intersection(*take_sets))
        if not common_takes:
            continue

        for take in common_takes:
            loaded = {}
            min_frames = None
            for modal in data_modals:
                arr = np.load(per_modal_take_files[modal][take], allow_pickle=False)
                arr = np.asarray(arr)
                if arr.ndim < 2:
                    arr = arr.reshape(-1, 1)
                loaded[modal] = arr
                if min_frames is None:
                    min_frames = arr.shape[0]
                else:
                    min_frames = min(min_frames, arr.shape[0])

            for i, modal in enumerate(data_modals):
                arr = loaded[modal][:min_frames]
                # Apply pressure LOD before flattening. Pressure has shape
                # (T, H, W, 2), where the last dim is [left, right] foot.
                if modal == 'pressure':
                    lod_level = int(cfg['data']['pressure'].get('lod_level', 0))
                    if lod_level > 0:
                        arr = norms.apply_lod(arr, lod_level)

                flat = arr.reshape(min_frames, -1).astype(np.float32)
                per_modal_rows[i].append(flat)

            all_frames.append(np.arange(min_frames, dtype=np.int32))
            all_subinfo.append(
                np.repeat(np.array([[subject, int(take)]], dtype=np.int32), min_frames, axis=0)
            )

    if not all_subinfo:
        raise ValueError(
            f"No aligned modality files found in {data_dir}. "
            "Expected files like Subject1/mocap_3d_1.npy and Subject1/pressure_1.npy"
        )

    frames = np.concatenate(all_frames, axis=0)
    sub_info = np.concatenate(all_subinfo, axis=0)
    valid_frames_combined = np.arange(frames.shape[0], dtype=np.int64)

    datasets = []
    modal_order = []
    for i, modal in enumerate(data_modals):
        data_matrix = np.concatenate(per_modal_rows[i], axis=0) if per_modal_rows[i] else np.empty((0, 0))
        datasets.append({
            'modal': modal,
            'file': None,
            'norm': _get_modal_norm(modal, cfg),
            'data_matrix': data_matrix,
            'dim': data_matrix.shape[1] if data_matrix.ndim == 2 else 0,
            'channels': 1,
        })
        modal_order.append(modal)

    featnames = []
    return valid_frames_combined, sub_info, frames, featnames, datasets, modal_order


def merge_classes(vals, class_map):
    """Merge class labels using the given class_map.

    class_map is loaded from assets/pose_info.json.
    """
    fixed_class_map = {int(k): int(v) for k, v in class_map.items()}
    mapped_arr = np.vectorize(fixed_class_map.get)(vals)
    return mapped_arr


def gen_labels(frame_list, subinfo, keypose_table, takemap):
    """Assign class labels using a legacy keypose table and takemap."""
    labels = np.zeros(frame_list.shape, dtype=np.int64)
    for i in range(len(frame_list)):
        col_idx = np.where((takemap[0, :] == subinfo[i, 0]) & (takemap[1, :] == subinfo[i, 1]))[0][0]
        for j in range(keypose_table.shape[0]):
            if np.isnan(keypose_table[j, col_idx]):
                continue
            if frame_list[i] <= keypose_table[j, col_idx] + 19 and frame_list[i] >= keypose_table[j, col_idx] - 101:
                labels[i] = j + 1
                break
    return labels


def gen_labels_from_keypose_df(frame_list, subinfo, keypose_df, m=100, n=20, label_offset=1):
    """Assign class labels from a keypose CSV over a [m,n] frame window."""
    df = keypose_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'class_index' in df.columns:
        class_col = 'class_index'
    elif 'label' in df.columns:
        class_col = 'label'
    else:
        raise ValueError("Keypose CSV must have either 'class_index' or 'label' column.")
    for col in ['subject', 'take', 'frame_idx', class_col]:
        if col not in df.columns:
            raise ValueError(f"Keypose CSV missing required column '{col}'.")

    keypose_ranges = {}
    for row in df.itertuples(index=False):
        subject = int(getattr(row, 'subject'))
        take = int(getattr(row, 'take'))
        center = int(getattr(row, 'frame_idx'))
        label = int(getattr(row, class_col)) + int(label_offset)
        start = center - int(m)
        end = center + int(n)
        keypose_ranges.setdefault((subject, take), []).append((label, start, end))

    for key in keypose_ranges:
        keypose_ranges[key].sort(key=lambda x: x[0])

    labels = np.zeros(frame_list.shape, dtype=np.int64)
    for i in range(len(frame_list)):
        key = (int(subinfo[i, 0]), int(subinfo[i, 1]))
        frame = int(frame_list[i])
        for label, start, end in keypose_ranges.get(key, []):
            if frame >= start and frame <= end:
                labels[i] = label
                break
    return labels


def class_balance(labels, subinfo, takemap=None, indices=None):
    """Downsample background (class 0) so it does not dominate training."""
    np.random.seed(0)
    if indices is None:
        indices = np.array([], dtype=np.int64)
        if takemap is None:
            takes = np.unique(subinfo, axis=0)
        else:
            takes = np.stack((takemap[0, :], takemap[1, :]), axis=1)

        for subject, take in takes:
            take_indices = np.where((subinfo[:, 0] == subject) & (subinfo[:, 1] == take))[0]
            if len(take_indices) == 0:
                continue
            take_labels = labels[take_indices]
            zero_indices = take_indices[np.nonzero(take_labels == 0)[0]]
            nonzero_indices = take_indices[np.nonzero(take_labels)[0]]
            if len(zero_indices) > 13:
                chosen = np.random.choice(zero_indices, (13,), replace=False)
            else:
                chosen = zero_indices
            indices = np.concatenate((indices, chosen, nonzero_indices))

    indices = np.sort(indices)
    return indices


def early_fusion(datasets: dict, indices: list, valid_frames: list, cfg):
    """Concatenate modality feature matrices into a single tabular array."""
    data_modalities = []
    for i in range(len(datasets)):
        modal_name = datasets[i].get('modal')
        if 'data_matrix' in datasets[i]:
            dataset = datasets[i]['data_matrix']
            if valid_frames is not None:
                dataset = dataset[valid_frames, :]
        else:
            dataset = np.load(os.path.join(cfg['default']['data_path'], f"{datasets[i]['file']}.npz"))
            dataset = dataset['data'][valid_frames, :]

        if indices is not None:
            dataset = dataset[indices, :]

        norm = datasets[i].get('norm')

        # Pressure is already converted to a distribution during dataset
        # creation, so we avoid any additional feature-wise normalization
        # here in the tabular pipeline.
        if modal_name == 'pressure':
            norm = None

        # Keep the built-in options intentionally simple for students:
        #   - None   : no normalization
        #   - MINMAX : feature-wise min/max scaling to [0, 1]
        #   - ZSCORE : feature-wise z-score normalization (default for mocap_3d)
        if norm == "MINMAX":
            dataset, _, _ = norms.minmax(dataset)
        elif norm == "ZSCORE":
            dataset, _, _ = norms.zscore(dataset)
        elif norm is None:
            pass
        else:
            # Students are encouraged to implement their own normalization
            # schemes (e.g., a log transform) if they want to experiment.
            raise NotImplementedError(f"The normalization '{norm}' is not supported.")
        data_modalities.append(dataset)

    if len(data_modalities) == 1:
        data = data_modalities[0]
    else:
        data = np.concatenate(data_modalities, axis=1)
    return data
