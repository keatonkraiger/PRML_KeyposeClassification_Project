import pandas as pd
import h5py
import argparse
import numpy as np
import glob
import os
import scipy.io as sio
from tabular.tabular.data.data_operations import *

"""
Dataset creation script for Taiji keypose classification.

Supports temporal downsampling via --downsample_rate argument:
    --downsample_rate 1   : No downsampling (all frames)
    --downsample_rate 5   : Keep every 5th frame (5x compression)

Downsampling is applied during raw data loading, BEFORE any processing,
so all downstream indices (keyframe locations, frame numbers) align correctly.

Example: python create_dataset.py --data_path /path/to/data --downsample_rate 5
"""


def extract_take_number(filename, modality):
    """Extract take number from file like MOCAP_3D_10.npy"""
    basename = os.path.basename(filename)
    prefix = modality + '_'
    if not basename.startswith(prefix):
        raise ValueError(f"Filename '{filename}' does not start with expected modality prefix '{modality}_'")
    number_str = basename[len(prefix):].split('.')[0]
    return int(number_str)


def _transpose_descending_dims(arr):
    """Transpose array so axis sizes are in descending order."""
    if arr.ndim < 3:
        raise ValueError(f"Expected at least 3 dims, got shape {arr.shape}")

    # Stable sort by dimension size (largest -> smallest).
    axes = sorted(range(arr.ndim), key=lambda i: arr.shape[i], reverse=True)
    arr = np.transpose(arr, axes=axes)

    # Equal dimensions cannot satisfy strict '>', so we keep a stable order and continue.
    if any(arr.shape[i] == arr.shape[i + 1] for i in range(arr.ndim - 1)):
        print(f"[WARN] Tied dimensions after transpose: shape={arr.shape}")

    return arr


def _load_mat_array(file_path, candidate_keys):
    """Load one array from a MAT file using candidate keys.

    Supports both v7.3 (HDF5) and older MAT formats.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            for key in candidate_keys:
                if key in f.keys():
                    return np.asarray(f[key]), key
    except OSError:
        # Not an HDF5-based MAT file, fall back to scipy loader below.
        pass

    f = sio.loadmat(file_path)
    for key in candidate_keys:
        if key in f:
            return np.asarray(f[key]), key

    raise ValueError(f"None of keys {candidate_keys} found in {file_path}")


def _safe_modality_name(modality):
    """Convert modality string to a filesystem-safe file stem."""
    safe = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(modality))
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_')

def load_subject_data(subject_dir, modality):
    pattern = os.path.join(subject_dir, f"{modality}_*.mat")
    data = []

    if 'PRESSURE' in modality.upper():
        candidate_keys = ('PRESSURE',)
    else:
        candidate_keys = ('POSE',)

    for file_path in sorted(glob.glob(pattern)):
        take = extract_take_number(os.path.basename(file_path), modality)
        arr, loaded_key = _load_mat_array(file_path, candidate_keys)
        arr = _transpose_descending_dims(arr)
        print(f"[INFO] Loaded {os.path.basename(file_path)} key={loaded_key} shape={arr.shape}")
        data.append((take, arr))

    return data
    
def process_subject(sub, args, summary_records):
    print(f"[INFO] Processing Subject {sub}")
    subj_dir = os.path.join(args.data_path, 'Subject_wise', f"Subject{sub}")
    save_dir = os.path.join(args.output_dir,  f"Subject{sub}")
    os.makedirs(save_dir, exist_ok=True)

    # 1) load each modality
    data = {mod: dict(load_subject_data(subj_dir, mod))
            for mod in args.modalities}

    take_sets = [set(mod_data.keys()) for mod_data in data.values() if len(mod_data) > 0]
    if not take_sets:
        print(f"[WARN] No take files found for Subject {sub}")
        return

    all_takes = sorted(set.union(*take_sets))
    common_takes = sorted(set.intersection(*take_sets))
    for take in all_takes:
        missing_mods = [mod for mod, mod_data in data.items() if take not in mod_data]
        if missing_mods:
            print(f"[WARN] Subject {sub} take {take} missing modalities: {missing_mods}")

    for take in common_takes:
        print(f"[INFO] Processing take {take} for Subject {sub}")
        
        for modality in data.keys():
            mod_data = data[modality]
            take_data = mod_data[take]

            # Apply temporal downsampling if requested
            if args.downsample_rate > 1:
                take_data = take_data[::args.downsample_rate]
                print(f"[INFO] Downsampled {modality} by factor {args.downsample_rate}, new shape: {take_data.shape}")
            
            if 'PRESSURE' in modality.upper():
                max_pressure = np.nanmax(take_data)
                if args.pressure_norm == 'dist':
                    take_data = normalize_pressure_dist(take_data)
                elif args.pressure_norm == 'max':
                    take_data = normalize_pressure_max(take_data, max_pressure)
                    
            else: # Is either MOCAP_3D or BODY25_V1/2
                pose_data = take_data[..., :-1] 
                conf_data = take_data[...,-1]
                
                # Center the data
                center_idx = 8 if 'BODY' in modality else 12
                pose_centered = center_pose(pose_data, center_idx)
                pose_with_conf = np.concatenate([pose_centered, conf_data[..., None]], axis=-1)
                
                if pose_centered.shape[-1] == 3:  # Then its 3D data
                    if args.pose_representation == 'quaternion':
                        pose_feats = convert_to_quaternion(pose_with_conf, center_joint=center_idx)
                    elif args.pose_representation == 'euler':
                        pose_feats = convert_to_euler(pose_with_conf, center_joint=center_idx)
                    else:
                        pose_feats = pose_centered
                else:
                    pose_feats = pose_centered
                
                # Normalize the pose data
                if args.pose_norm == 'mean':
                    mean, std = compute_norm_stats(pose_feats)
                    pose_feats = normalize(pose_feats, mean, std)
                else:
                    raise NotImplementedError(f"Pose normalization '{args.pose_norm}' not implemented")
                
                take_data = np.concatenate([pose_feats, conf_data[..., None]], axis=-1)   

            safe_modal = _safe_modality_name(modality)
            out_file = os.path.join(save_dir, f"{safe_modal}_{take}.npy")
            take_data = np.asarray(take_data, dtype=np.float32)
            np.save(out_file, take_data)

            summary_records.append({
                'subject': sub,
                'take': int(take),
                'modality': modality,
                'num_frames': int(take_data.shape[0]),
                'sample_shape': str(tuple(take_data.shape[1:])),
                'file': os.path.relpath(out_file, args.output_dir),
                'downsample_rate': int(args.downsample_rate),
            })
            print(f"[INFO] Saved take data -> {out_file}")
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the root directory of the PSUTMM-100 dataset')
    parser.add_argument('--modalities', nargs='+', default=['MOCAP_3D', 'Pressure'])
    parser.add_argument('--output_dir', type=str, default='Taiji_dataset')
    parser.add_argument('--pose_representation', type=str, default='quaternion', choices=['quaternion'])
    parser.add_argument('--pose_norm', type=str, default='mean', choices=['mean'])  # Extend later
    parser.add_argument('--pressure_norm', type=str, default='dist', choices=['dist', 'max'])
    parser.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    parser.add_argument('--downsample_rate', type=int, default=1, help='Temporal downsampling: keep every Nth frame (1=no downsampling, 5=keep every 5th frame)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_records = []

    for sub in args.subjects:
        process_subject(sub, args, summary_records)

    summary_cols = [
        'subject', 'take', 'modality', 'num_frames',
        'sample_shape', 'file', 'downsample_rate'
    ]
    summary_df = pd.DataFrame(summary_records, columns=summary_cols)
    summary_df.to_csv(
        os.path.join(args.output_dir, 'summary.csv'),
        index=False
    )
    total_takes = int(len(summary_df))
    print(f"[INFO] Dataset creation complete. {total_takes} modality-take files saved.")

if __name__ == '__main__':
    main()
