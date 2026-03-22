import joblib
import json
import numpy as np
import os
import time
import pandas as pd
import sys
import yaml

from tabular.util.utils import *
from tabular.data.dataset_util import *
from tabular.util.viz import *

from sklearn.neighbors import KNeighborsClassifier

# Define a dictionary to map the classifier names to their classes
classifier_mapping = {
    'knn': KNeighborsClassifier,
}

# Define a dictionary to map the classifier names to their file extensions
file_ext_mapping = {
    'knn': '.joblib',
}

data_dims = {
    'BODY25': (24, 3), # 2D openpose joints (minus hip) + 1 conf. 
    'MOCAP_3D': (16, 4), # 3D MOCAP joints (minus pelvis) + 1 conf.
}

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model

def test_model(model, test_data):
    predictions = model.predict(test_data)
    return predictions


def _resolve_subjects(subject_cfg):
    if isinstance(subject_cfg, str):
        if subject_cfg.strip().lower() == 'all':
            return np.arange(1, 11)
        return np.array([int(x.strip()) for x in subject_cfg.split(',') if x.strip()])
    if isinstance(subject_cfg, (list, tuple, np.ndarray)):
        return np.array([int(x) for x in subject_cfg])
    return np.array([int(subject_cfg)])


def _load_class_map(cfg):
    pose_info_json = cfg['default']['pose_info_json']
    with open(pose_info_json, 'r') as f:
        pose_info = json.load(f)
    class_map = pose_info['class_map']
    return {int(k): int(v) for k, v in class_map.items()}


def _prepare_merge_map(class_map, labels):
    if class_map is None:
        return None
    merge_map = dict(class_map)
    merge_map.setdefault(0, 0)
    for lbl in np.unique(labels):
        li = int(lbl)
        if li not in merge_map:
            merge_map[li] = li
    return merge_map

def main(cfg):
    print(f"Training and testing a {cfg['classify']['classifier']} on {cfg['data']['modals']} data.\n")

    if not os.path.exists(cfg['default']['results_path']):
        os.makedirs(cfg['default']['results_path'])

    subjects = _resolve_subjects(cfg['default']['subject'])
    do_merged_eval = bool(cfg['default']['enable_merged_eval'])

    class_map = _load_class_map(cfg)

    # Prep the data and labels
    valid_frames, sub_info, frames, featnames, datasets, modal_order = create_datasets(cfg)
    keypose_csv = cfg['default']['keypose_csv']

    keypose_df = pd.read_csv(keypose_csv)
    m = int(cfg['default']['m'])
    n = int(cfg['default']['n'])
    labels = gen_labels_from_keypose_df(frames, sub_info, keypose_df, m=m, n=n, label_offset=1)

    # ---------------- Dataset summary ----------------
    num_samples = labels.shape[0]
    unique_labels = np.unique(labels)
    print("Data summary:")
    print(f"  Total labeled frames: {num_samples}")
    print(f"  Subjects in config: {list(_resolve_subjects(cfg['default']['subject']))}")
    print(f"  Num classes (unmerged): {int(np.max(labels)) + 1}")
    print(f"  Label values present: {unique_labels.tolist()}")
    print("  Modalities and feature dims:")
    for ds in datasets:
        modal = ds.get('modal')
        dim = ds.get('dim')
        channels = ds.get('channels', 1)
        norm = ds.get('norm')
        print(f"    - {modal}: dim={dim}, channels={channels}, norm={norm}")
    print("")

    indices = None
    if cfg['default']['balance_classes']:
        indices = class_balance(labels, sub_info)
        sub_info = sub_info[indices]
        labels = labels[indices]

    num_forms = int(np.max(labels)) + 1
    merge_map = _prepare_merge_map(class_map, labels) if do_merged_eval else None
    if merge_map is not None:
        merged_labels_all = merge_classes(labels, merge_map)
        num_forms_merged = int(np.max(merged_labels_all)) + 1
    else:
        num_forms_merged = None

    num_subs = len(subjects)

    run_details = experiment_name(cfg)
    experiment_path = os.path.join(cfg['default']['results_path'], run_details)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    stats_unmerged_dir = os.path.join(experiment_path, 'stats', 'unmerged')
    os.makedirs(stats_unmerged_dir, exist_ok=True)
    if do_merged_eval and merge_map is not None:
        stats_merged_dir = os.path.join(experiment_path, 'stats', 'merged')
        os.makedirs(stats_merged_dir, exist_ok=True)

    with open(f"{os.path.join(experiment_path, 'cfg')}.json", 'w') as f:
        json.dump(cfg, f, indent=4) 

    subject_stats_unmerged = {}
    subject_stats_merged = {}
    stat_track = StatTracker(num_classes=num_forms, num_subs=num_subs, class_map=merge_map if do_merged_eval else None)
    data = None

    start = time.time()
    for i, subject in enumerate(subjects):
        if data is None:
            if (cfg['default']['fusion'] == 'early') or (len(cfg['data']['modals']) == 1):
                data = early_fusion(datasets, indices, valid_frames, cfg)
            else:
                raise NotImplementedError(f"You will need to implement support for {cfg['default']['fusion']}' fusion.")

        train_data, train_labels, test_data, test_labels = loso_split(data, labels, sub_info, subject)

        # Optional shuffle of the training split, controlled via
        # cfg['training']['shuffle_train']. This is a no-op for KNN
        # but mirrors the behavior students may want for other models.
        if cfg.get('training', {}).get('shuffle_train', False):
            perm = np.random.permutation(train_data.shape[0])
            train_data = train_data[perm]
            train_labels = train_labels[perm]

        print(f"\n[{i+1}/{num_subs}] Training for subject {subject}...")
        print(f"  Train samples: {train_data.shape[0]}, Test samples: {test_data.shape[0]}, Features: {train_data.shape[1]}")
        classifier = cfg['classify']['classifier']

        if classifier not in classifier_mapping:
            raise NotImplementedError(f"You will need to add support for {classifier} classifier.")

        # Allows us to pass classifier config kwargs directly
        classifier_cfg = cfg['classify'].copy()
        del classifier_cfg['classifier']
        model = classifier_mapping[classifier](**classifier_cfg)
        file_ext = file_ext_mapping[classifier]

        model = train_model(model, train_data, train_labels)
        train_predictions = test_model(model, train_data)
        test_predictions = test_model(model, test_data)

        per_class_acc_train, conf_train, _= get_stats(train_predictions, train_labels, num_classes=num_forms)
        stat_track.record(per_class_acc_train, conf_train, i, train=True, merged=False)
        per_class_acc_test, conf_test, test_metrics_unmerged = get_stats(test_predictions, test_labels, num_classes=num_forms)
        stat_track.record(per_class_acc_test, conf_test, i, train=False, merged=False)

        # By default, we only evaluate on the unmerged classes. If you
        # enable cfg['default']['enable_merged_eval'], we also evaluate on
        # the merged classes defined by the class_map in pose_info_json.
        if do_merged_eval and merge_map is not None and num_forms_merged is not None:
            per_class_acc_train_merged, conf_train_merged, _ = get_stats(
                train_predictions,
                train_labels,
                num_classes=num_forms_merged,
                class_map=merge_map,
            )
            stat_track.record(per_class_acc_train_merged, conf_train_merged, i, train=True, merged=True)
            per_class_acc_test_merged, conf_test_merged, test_metrics_merged = get_stats(
                test_predictions,
                test_labels,
                num_classes=num_forms_merged,
                class_map=merge_map,
            )
            stat_track.record(per_class_acc_test_merged, conf_test_merged, i, train=False, merged=True)
            test_metrics_merged['accuracy'] = float(np.mean(per_class_acc_test_merged))
            subject_stats_merged[f'{subject}'] = test_metrics_merged

        test_metrics_unmerged['accuracy'] = float(np.mean(per_class_acc_test))
        subject_stats_unmerged[f'{subject}'] = test_metrics_unmerged

        sub_file_unmerged = os.path.join(stats_unmerged_dir, f'subject_{subject}.npz')
        stat_track.save_sub(sub_file_unmerged, merged=False)
        if do_merged_eval and merge_map is not None:
            sub_file_merged = os.path.join(stats_merged_dir, f'subject_{subject}.npz')
            stat_track.save_sub(sub_file_merged, merged=True)

        stat_track.print_stats_by_subject(i, merged=False)
        if do_merged_eval and merge_map is not None:
            stat_track.print_stats_by_subject(i, merged=True)

        if cfg['default']['save_model']:
            model_dir = os.path.join(experiment_path, cfg['default']['model_path'])
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_file = os.path.join(model_dir, f'subject_{subject}' + file_ext)
            if cfg['classify']['classifier'] == 'knn':
                joblib.dump(model, model_file)
            else:
                raise NotImplementedError(f"You will need to implement (or look up how to save) {cfg['classify']['classifier']} models.")

        # Simple progress timing
        elapsed = time.time() - start
        avg_per_sub = elapsed / (i + 1)
        remaining = avg_per_sub * (num_subs - (i + 1))
        print(f"  Elapsed: {elapsed/60:.2f} min, ETA: {remaining/60:.2f} min")
                

    end = time.time()
    test_metrics_unmerged = calculate_overall_subject_stats(subject_stats_unmerged)
    if do_merged_eval and merge_map is not None and subject_stats_merged:
        test_metrics_merged = calculate_overall_subject_stats(subject_stats_merged)
    else:
        test_metrics_merged = None

    print(f'Training and testing took {(end-start)//60} mins. {(end-start)%60:.2f} sec.')

    overall_file_name_unmerged = os.path.join(stats_unmerged_dir, 'overall.npz')
    stat_track.save_overall(overall_file_name_unmerged, merged=False)
    stat_track.print_overall(merged=False)

    # Create plots of the unmerged results
    plot_stats(experiment_path, classes='unmerged')

    if do_merged_eval and merge_map is not None:
        overall_file_name_merged = os.path.join(stats_merged_dir, 'overall.npz')
        stat_track.save_overall(overall_file_name_merged, merged=True)
        stat_track.print_overall(merged=True)
        plot_stats(experiment_path, classes='merged')
        plot_merged_and_unmerged(experiment_path)

    return {
        'unmerged': test_metrics_unmerged,
        'merged': test_metrics_merged,
    }

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test classifiers on the Taiji dataset.')
    parser.add_argument('--cfg', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    with open(args.cfg, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(cfg)