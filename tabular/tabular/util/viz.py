import argparse
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import os
from sklearn.metrics import auc
from sklearn.metrics import PrecisionRecallDisplay

from tabular.util.utils import *

def plot_feet(data, file_name):
    assert data.shape == (60, 21, 2), 'Invalid data shape, expected (2, 21, 60)'
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    
    for i, (ax, title) in enumerate(zip(axs, ['Left Foot', 'Right Foot'])):
        #foot_data = np.transpose(data[i, :, :])
        foot_data = data[:,:, i]
        c = ax.imshow(foot_data, cmap='jet', interpolation='nearest')
        ax.set_title(title)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(c, cax=cbar_ax)
    # Save with no pads or bordedr
    plt.savefig(file_name, bbox_inches=None, pad_inches=0)
    plt.close()

def plot_mocap_3d(data, fname, **kwargs):
    # Extract data for the given frame number
    x_vals = data[0, :]
    y_vals = data[1, :]
    z_vals = data[2, :]
    
    joint_names = ["RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", 
                   "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "PelvisCenter", 
                   "Waist", "TopOfNeck", "Clavicle", "Thorax"]

    # Define connections based on anatomical positions
    connections = [
        (1, 2), (2, 3),       # Right arm
        (4, 5), (5, 6),       # Left arm
        (7, 8), (8, 9),       # Right leg
        (10, 11), (11, 12),   # Left leg
        (13, 17), (17, 15),
        #(13, 14), (14, 15),   # Spine up to neck
        (13, 10), (13, 7),    # Pelvis Center to hips (to represent lower body)
        (17, 1), (17, 4),      # Shoulders to clavicle (to represent upper body)
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each connection
    for start, end in connections:
        ax.plot([x_vals[start - 1], x_vals[end - 1]], 
                [-y_vals[start - 1], -y_vals[end - 1]], 
                [z_vals[start - 1], z_vals[end - 1]], color=kwargs['line_color'], linewidth=kwargs['line_width'])

    ax.grid(False)
    ax.axis('off')

    # Set the limits to focus on the skeleton
    margin = kwargs['margin']
    ax.set_xlim([x_vals.min() - margin, x_vals.max() + margin])
    ax.set_ylim([-(y_vals.max() + margin), -(y_vals.min() - margin)])
    ax.set_zlim([z_vals.min(), z_vals.max()])

    # Adjust view parameters
    ax.view_init(elev=kwargs['elev'], azim=kwargs['azim'])
    ax.set_proj_type('ortho')  # Set to orthographic
    ax.dist = 6.5  # This controls the distance of the camera to the plot, consider adjusting for zoom

    plt.subplots_adjust(left=0, right=.5, bottom=0, top=.5)  # Remove padding around the plot
    plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_stats(results_path, classes):
    save_dir = os.path.join(results_path, 'plots', classes)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    overall = np.load(os.path.join(results_path, 'stats', classes, 'overall.npz'))
    sub_train_acc = overall['train_acc']
    sub_test_acc = overall['test_acc']
    overall_train_mat = overall['conf_mat_train']
    overall_test_mat = overall['conf_mat_test']

    num_subs = sub_train_acc.shape[0]
    num_classes = overall_train_mat.shape[0]
    x = np.arange(num_subs)

    # Bar chart for subject test accuracy with mean and std lines
    fig, ax = plt.subplots()
    ax.bar(x, sub_test_acc)
    ax.set_xticks(x)  # Set the x-tick locations
    ax.set_xticklabels(x + 1)  # Set the x-tick labels to be subject numbers from 1 to 10
    mean = np.mean(sub_test_acc)
    std = np.std(sub_test_acc)
    ax.axhline(mean, color='r', linestyle='-', label=f'Mean: {mean:.2f}')
    ax.axhline(mean + std, color='orange', linestyle='--', label=f'+ Std: {mean+std:.2f}')
    ax.axhline(mean - std, color='orange', linestyle='--', label=f'- Std: {mean-std:.2f}')
    ax.set_xlabel('Subject', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title(f'Test Accuracy by Subject ({classes.capitalize()})', fontweight='bold')
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'test_acc.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Bar chart for subject train accuracy with mean and std lines
    fig, ax = plt.subplots()
    ax.bar(x, sub_train_acc)
    ax.set_xticks(x)  # Set the x-tick locations
    ax.set_xticklabels(x + 1)  # Set the x-tick labels to be subject numbers from 1 to 10
    mean = np.mean(sub_train_acc)
    std = np.std(sub_train_acc)
    ax.axhline(mean, color='r', linestyle='-', label=f'Mean: {mean:.2f}')
    ax.axhline(mean + std, color='orange', linestyle='--', label=f'+ Std: {mean+std:.2f}')
    ax.axhline(mean - std, color='orange', linestyle='--', label=f'- Std: {mean-std:.2f}')
    ax.set_xlabel('Subject', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title(f'Train Accuracy by Subject ({classes.capitalize()})', fontweight='bold')
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'train_acc.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Plot training confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(overall_train_mat, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(f'Training Confusion Matrix ({classes.capitalize()})', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'train_conf.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    fig, ax = plt.subplots()
    cax = ax.matshow(overall_test_mat, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(f'Testing Confusion Matrix ({classes.capitalize()})', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'test_conf.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_merged_and_unmerged(results_path):
    save_dir = os.path.join(results_path, 'plots')

    overall_merged = np.load(os.path.join(results_path, 'stats', 'merged', 'overall.npz'))
    train_acc_merged = overall_merged['train_acc']
    test_acc_merged = overall_merged['test_acc']

    overall_unmerged = np.load(os.path.join(results_path, 'stats', 'unmerged', 'overall.npz'))
    train_acc_unmerged = overall_unmerged['train_acc']
    test_acc_unmerged = overall_unmerged['test_acc']

    num_subs = train_acc_merged.shape[0]
    x = np.arange(num_subs)

    # Create a grouped bar chart with merged and unmerged train acc
    fig, ax = plt.subplots()
    width = 0.45
    bars1 = ax.bar(x - width/2, train_acc_merged, width, color='blue', hatch='//') 
    bars2 = ax.bar(x + width/2, train_acc_unmerged, width, color='tab:orange')
    mean_line1 = ax.axhline(np.mean(train_acc_merged), linestyle='--', color='black')
    mean_line2 = ax.axhline(np.mean(train_acc_unmerged), linestyle='--', color='orange')

    ax.set_xticks(x)  # Set the x-tick locations
    ax.set_xticklabels(x + 1)  # Set the x-tick labels to be subject numbers from 1 to 10
    ax.set_xlabel('Subject', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Train Accuracy by Subject (Merged vs. Unmerged)', fontweight='bold', fontsize=10)
    
    # create custom legend 
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='Merged', markerfacecolor='blue', markersize=10, linestyle='None'),
                       Line2D([0], [0], marker='s', color='w', label='Unmerged', markerfacecolor='orange', markersize=10, linestyle='None'),
                       Line2D([0], [0], color='black', lw=2, linestyle='--', label='Merged Mean'),
                       Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Unmerged Mean')]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.4, 0.55), fontsize=8)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'train_acc_merged_vs_unmerged.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


    # Create a grouped bar chart with merged and unmerged test acc
    fig, ax = plt.subplots()
    width = 0.45
    ax.bar(x - width/2, test_acc_merged, width, color='blue', hatch='//') 
    ax.bar(x + width/2, test_acc_unmerged, width, color='tab:orange')
    ax.axhline(np.mean(test_acc_merged), linestyle='--', color='black')
    ax.axhline(np.mean(test_acc_unmerged), linestyle='--', color='orange')
    ax.set_xticks(x)  # Set the x-tick locations
    ax.set_xticklabels(x + 1)  # Set the x-tick labels to be subject numbers from 1 to 10
    ax.set_xlabel('Subject', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Test Accuracy by Subject (Merged vs. Unmerged)', fontweight='bold', fontsize=10)

        # create custom legend 
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='Merged', markerfacecolor='blue', markersize=10, linestyle='None'),
                       Line2D([0], [0], marker='s', color='w', label='Unmerged', markerfacecolor='orange', markersize=10, linestyle='None'),
                       Line2D([0], [0], color='black', lw=2, linestyle='--', label='Merged Mean'),
                       Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Unmerged Mean')]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.4, 0.55), fontsize=8)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'test_acc_merged_vs_unmerged.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_metrics(metrics_dict, experiment_path, merged):
    save_dir = os.path.join(experiment_path, 'plots', merged)
    os.makedirs(save_dir, exist_ok=True)

    # ROC Curve
    plt.figure(figsize=(15, 5))
    for sub in metrics_dict['fpr'].keys():
        plt.plot(metrics_dict['fpr'][sub]['micro'], metrics_dict['tpr'][sub]['micro'], label=f'Subject {sub} ROC curve (area = {auc(metrics_dict["fpr"][sub]["micro"], metrics_dict["tpr"][sub]["micro"]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average ROC curve over subjects')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # F1-Score, Recall, Precision, roc-auc bar plot
    plt.figure(figsize=(15, 5))
    bar_values = [metrics_dict['f1_score_macro'], metrics_dict['recall_macro'], metrics_dict['precision_macro'], metrics_dict['roc_auc']]
    bar_labels = ['F1-Score Macro', 'Recall Macro', 'Precision Macro', 'ROC_AUC Macro']
    plt.bar(bar_labels, bar_values)
    plt.title('Metrics Bar Plot')
    plt.savefig(os.path.join(save_dir, 'metrics_bar_plot.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
