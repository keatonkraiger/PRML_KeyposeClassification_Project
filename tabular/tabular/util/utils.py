import time
import json
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import label_binarize

def print_data_info(cfg):
    print('--------------------')
    print(f"Training {cfg['classify']['classifier']} on {cfg['data']['modals']} data")


    print(f"Training with the following options:")
    for modal in cfg['data']['modals']:
        print(f"\t{modal}: ", end = '\t')
        for key, val in cfg['data'][modal].items():
            print(f"\t{key}: {val}", end='\t')
        print()

def calculate_metrics(true_labels, pred_labels, num_classes):
    # Binarize the output
    true_labels_bin = label_binarize(true_labels, classes=np.arange(num_classes))
    pred_labels_bin = label_binarize(pred_labels, classes=np.arange(num_classes))

    fpr = dict()
    tpr = dict()

    # Calculate ROC curve and ROC AUC for each class
    for i in range(num_classes):
        if np.any(true_labels_bin[:, i]) or np.any(pred_labels_bin[:, i]):
            fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], pred_labels_bin[:, i])
        else:
            fpr[i] = tpr[i] = np.array([0, 1])

    # Compute micro-average ROC curve and ROC AUC
    fpr['micro'], tpr['micro'], _ = roc_curve(true_labels_bin.ravel(), pred_labels_bin.ravel())

    # Calculate macro ROC AUC score
    try:
        roc_auc = roc_auc_score(true_labels_bin, pred_labels_bin, average='macro')
    except ValueError:
        roc_auc = np.nan

    # Compute precision, recall, F-measure for each class
    precision_macro, recall_macro, f1_score_macro, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0)

    # Compute micro-average precision, recall, F-measure
    precision_micro, recall_micro, f1_score_micro, _ = precision_recall_fscore_support(true_labels, pred_labels, average='micro', zero_division=0)
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_score_macro': f1_score_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_score_micro': f1_score_micro,
    }

def calculate_overall_subject_stats(subject_stats, include_keys=['roc_auc', 'precision_macro', 'recall_macro', 'f1_score_macro']):
    overall_stats = {}
    for key in include_keys:
        overall_stats[key] = np.mean([subject_stats[subject][key] for subject in subject_stats.keys()])
    return overall_stats

def aggregate_metrics(subject_stats, keys_to_aggregate=['gt', 'pred', 'fpr', 'tpr', 'precision_micro', 'recall_micro', 'f1_score_micro']):
    aggregated_metrics = {}
    for key in keys_to_aggregate:
        aggregated_metrics[key] = {subject: subject_stats[subject][key] for subject in subject_stats.keys()}
    return aggregated_metrics

# Split data into train and test where test is provided subject number
def loso_split(data, labels, subject_info, subject):
    # Sanity check
    if subject < 1 or subject > 10:
        print("WARNING: Invalid subject provided")
    # Split the data
    train_list = np.where(subject_info[:,0] != subject)
    test_list = np.where(subject_info[:,0] == subject)
    train_data = data[train_list,:].squeeze()
    train_labels = labels[train_list]
    train_sub_info = subject_info[train_list, :].squeeze()
    test_data = data[test_list,:].squeeze()
    test_labels = labels[test_list]
    test_sub_info = subject_info[test_list, :].squeeze()
    # Fix NaNs
    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)
    # Return
    return train_data, train_labels.ravel(), test_data, test_labels.ravel()

def merge_classes(vals, class_map):
    # Convert the class map to ints if it's not already
    fixed_class_map = {int(k):int(v) for k,v in class_map.items()}
    mapped_arr = np.vectorize(fixed_class_map.get)(vals)
    return mapped_arr

def get_stats(preds, targets, num_classes, class_map=None):
    """
    Calculates the prediction stats.
    Args:
        preds (numpy array): Class predictions.
        targets (numpy array): Target values.
        num_classes (int): Number of classes.
        class_map (dict): Dictionary mapping classes to new classes.
    Returns:
        conf_diag(numpy array): Array of the number of correct predictions for each class.
        conf_mat (numpy array): Confusion matrix.
    """
    if class_map is not None:
        preds = merge_classes(preds, class_map)
        targets = merge_classes(targets, class_map)

    labels = np.arange(num_classes)
    conf_mat = confusion_matrix(targets, preds, labels=labels, normalize='true')
    conf_diag = np.diag(conf_mat)

    # Get ROC_AUC, pre., recall, f1
    metrics = calculate_metrics(targets, preds, num_classes)

    return conf_diag, conf_mat, metrics

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

class StatTracker():
    def __init__(self, num_classes, num_subs, class_map=None):
        self.num_classes = num_classes
        self.num_subs = num_subs

        self.subject_train_accs = np.zeros(num_subs)
        self.subject_test_accs = np.zeros(num_subs)

        self.subject_conf_mat_train = np.zeros((num_subs, num_classes))
        self.subject_conf_mat_test = np.zeros((num_subs, num_classes))

        self.overall_conf_mat_train = np.zeros((num_classes, num_classes))
        self.overall_conf_mat_test = np.zeros((num_classes, num_classes))

        if class_map is not None:
            self.class_map = class_map
            self.num_classes_merged = len(np.unique(list(class_map.values())))
            self.subject_train_accs_merged = np.zeros(num_subs)
            self.subject_conf_mat_train_merged = np.zeros((num_subs, self.num_classes_merged))
            self.subject_test_accs_merged = np.zeros(num_subs)
            self.subject_conf_mat_test_merged = np.zeros((num_subs, self.num_classes_merged))
            self.overall_conf_mat_train_merged = np.zeros((self.num_classes_merged, self.num_classes_merged))
            self.overall_conf_mat_test_merged = np.zeros((self.num_classes_merged, self.num_classes_merged))

    def record(self, per_class_acc, conf, sub, train=True, merged=False):
        if merged:
            if train:
                self.subject_train_accs_merged[sub] = np.mean(per_class_acc)
                self.subject_conf_mat_train_merged[sub, :] = per_class_acc 
                self.overall_conf_mat_train_merged += (conf * (1 / self.num_subs))
            else:
                self.subject_test_accs_merged[sub] = np.mean(per_class_acc)
                self.subject_conf_mat_test_merged[sub, :] = per_class_acc 
                self.overall_conf_mat_test_merged += (conf * (1 / self.num_subs))
        else:
            if train:
                self.subject_train_accs[sub] = np.mean(per_class_acc)    
                self.subject_conf_mat_train[sub, :] = per_class_acc 
                self.overall_conf_mat_train += (conf * (1 / self.num_subs))
            else:
                self.subject_test_accs[sub] = np.mean(per_class_acc)
                self.subject_conf_mat_test[sub, :] = per_class_acc
                self.overall_conf_mat_test += (conf * (1 / self.num_subs))

    def save_sub(self, sub_file, merged=False):
        if merged:
            np.savez(sub_file, train_acc=self.subject_train_accs_merged, test_acc=self.subject_test_accs_merged,
                    conf_mat_train=self.overall_conf_mat_train_merged, conf_mat_test=self.overall_conf_mat_test_merged)
        else:
            np.savez(sub_file, train_acc=self.subject_train_accs, test_acc=self.subject_test_accs,
                        conf_mat_train=self.overall_conf_mat_train, conf_mat_test=self.overall_conf_mat_test)
            
    def save_overall(self, overall_file, merged=False):
        if merged:
            np.savez(overall_file, train_acc=self.subject_train_accs_merged, test_acc=self.subject_test_accs_merged,
                    conf_mat_train=self.overall_conf_mat_train_merged, conf_mat_test=self.overall_conf_mat_test_merged)
        else:
            np.savez(overall_file, train_acc=self.subject_train_accs, test_acc=self.subject_test_accs,
                        conf_mat_train=self.overall_conf_mat_train, conf_mat_test=self.overall_conf_mat_test)

    def reset(self):
        # Set all of the tracked stats to zero arrays of the correct size
        self.subject_train_accs = np.zeros(self.num_subs)
        self.subject_conf_mat_train = np.zeros((self.num_subs, self.num_classes))
        self.subject_test_accs = np.zeros(self.num_subs)
        self.subject_conf_mat_test = np.zeros((self.num_subs, self.num_classes))
        self.overall_conf_mat_train = np.zeros((self.num_classes, self.num_classes))
        self.overall_conf_mat_test = np.zeros((self.num_classes, self.num_classes))

        if hasattr(self, 'num_classes_merged'):
            self.subject_train_accs_merged = np.zeros(self.num_subs)
            self.subject_conf_mat_train_merged = np.zeros((self.num_subs, self.num_classes_merged))
            self.subject_test_accs_merged = np.zeros(self.num_subs)
            self.subject_conf_mat_test_merged = np.zeros((self.num_subs, self.num_classes_merged))
            self.overall_conf_mat_train_merged = np.zeros((self.num_classes_merged, self.num_classes_merged))
            self.overall_conf_mat_test_merged = np.zeros((self.num_classes_merged, self.num_classes_merged))

    def print_stats_by_subject(self, i, merged=False):
        if merged:
            print(f"Train Accuracy (merged): {self.subject_train_accs_merged[i]*100:.2f}")
            print(f"Test Accuracy (merged): {self.subject_test_accs_merged[i]*100:.2f}")
        else:
            print("--------------------")
            print(f"Train Accuracy: {self.subject_train_accs[i]*100:.2f}")
            print(f"Test Accuracy: {self.subject_test_accs[i]*100:.2f}")

    def print_overall(self, merged=False):
        if merged:
            print('--------------------')
            print(f"Overall Train Accuracy (merged): {np.mean(self.subject_train_accs_merged)*100:.2f}")
            print(f"Overall Test Accuracy (merged): {np.mean(self.subject_test_accs_merged)*100:.2f}")
        else:
            print('\n--------------------')
            print(f"Overall Train Accuracy: {np.mean(self.subject_train_accs)*100:.2f}")
            print(f"Overall Test Accuracy: {np.mean(self.subject_test_accs)*100:.2f}")

def numpy_to_python_native_types(obj):
    if isinstance(obj, dict):
        return {key: numpy_to_python_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python_native_types(element) for element in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

def experiment_name(cfg):
    if 'yaml_path' in cfg['default']:
        experiment_name = cfg['default']['yaml_path'].split('/')[-1].split('.')[0]
    else:
        model = cfg['classify']['classifier']
        modals = cfg['data']['modals']
        opts = []
        for modal in modals:
            # Build a stable, string-only description of each modal's
            # configuration. Skip None values so optional settings like
            # norm: null do not cause type errors when joining.
            for key, value in cfg['data'][modal].items():
                if value is None:
                    continue
                opts.append(f"{modal}-{key}-{value}")

        modals = '_'.join(modals)
        opt_str = '_'.join(opts) if opts else 'default'
        timestamp = time.strftime("%m%d%H")
        experiment_name = f"{model}/{modals}/{opt_str}-{timestamp}"
    return experiment_name