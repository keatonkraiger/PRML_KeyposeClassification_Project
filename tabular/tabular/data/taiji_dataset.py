import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import tabular.data.data_operations as norms

def preprocess_indices(valid_frames, initial_indices=None):
    if initial_indices is not None:
        return np.intersect1d(valid_frames, initial_indices)
    return valid_frames

def get_LOSO_splits(subject_info, subject, train_val_ratio, valid_frames, initial_indices=None):
    preprocessed_indices = preprocess_indices(valid_frames, initial_indices)

    test_indices = np.intersect1d(np.where(subject_info[:, 0] == subject)[0], preprocessed_indices)
    non_test_indices = np.intersect1d(np.where(subject_info[:, 0] != subject)[0], preprocessed_indices)
    
    # Shuffle non_test_indices to ensure randomness when splitting to train and val
    np.random.shuffle(non_test_indices)
    
    split_point = int(len(non_test_indices) * train_val_ratio)
    train_indices = non_test_indices[:split_point]
    val_indices = non_test_indices[split_point:]
    
    if train_val_ratio == 1.0:
        train_indices = np.concatenate([train_indices, val_indices])
        val_indices = None
    
    return train_indices, val_indices, test_indices

class TaijiData(Dataset):
    def __init__(self, transform, cfg, subject, split='train'):
        """
        Args:
            transform (tv.transforms): Transform to apply to the data
            cfg (dict): Configuration dictionary
        """
        self.cfg = cfg
        self.transform = transform
        self.window_size = cfg['classify']['window_size'] 
        self.n_modalities = len(datasets)
        per_modal_data= self.load_np_data()
        if cfg['data']['flattened_data']:
            self.data = np.concatenate([per_modal_data[i] for i in range(self.n_modalities)], axis=1)
        else:
            self.data = list(zip(*per_modal_data))

    def load_np_data(self):
        data_modalities = []
        for i in range(self.n_modalities):
            np_dataset = np.load(os.path.join(self.cfg['default']['data_path'], f"{self.datasets[i]['file']}.npz"))
            data = np_dataset['data'][self.indices, :]
            if self.datasets[i]['modal'] == 'pressure' and not self.cfg['data']['flattened_data']:
                # Flatten spatial pressure data by multiplying the last two dims together
                data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))
            norm = self.datasets[i]['norm']

            # Normalize the data. For the student project we keep the
            # built-in options intentionally small:
            #   - None   : no normalization
            #   - MINMAX : feature-wise min/max scaling to [0, 1]
            #   - ZSCORE : feature-wise z-score normalization
            if norm == "MINMAX":
                data, _, _ = norms.minmax(data)
            elif norm == "ZSCORE":
                data, _, _ = norms.zscore(data)
            elif norm is None:
                pass
            else:
                # If students want to try other schemes (e.g., log
                # transforms), they can extend data_operations.py and add
                # new cases here.
                raise NotImplementedError(f"The normalization '{norm}' is not supported.")
            
            data_modalities.append(data)

        return data_modalities
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        half_window = self.window_size // 2
        start_idx = idx - half_window
        end_idx = idx + half_window + 1  # +1 to make it inclusive
        
        # Handle padding for beginning and ending edge cases
        pad_left = 0
        pad_right = 0
        if start_idx < 0:
            pad_left = abs(start_idx)
            start_idx = 0
        if end_idx > len(self.labels):
            pad_right = end_idx - len(self.labels)
            end_idx = len(self.labels)

        if not self.cfg['data']['flattened_data']:
            modality_data_slices = list(zip(*self.data[start_idx:end_idx]))  # Transpose your list of tuples

            # Pad each modality and apply transformations
            data = []
            for modality_slice in modality_data_slices:
                modality_slice = np.array(modality_slice)
                # Dynamic padding configuration
                if pad_left > 0 or pad_right > 0:
                    padding_dims = [(0, 0) for _ in range(modality_slice.ndim)]
                    padding_dims[0] = (pad_left, pad_right)
                    padding = tuple(padding_dims)
                    modality_slice = np.pad(modality_slice, padding, 'constant', constant_values=0)
                    modality_slice = modality_slice.transpose(1, 2, 0)
                else:
                    modality_slice = modality_slice.transpose(1, 2, 0)
                if self.transform:
                    modality_slice = self.transform(modality_slice)

                data.append(modality_slice)
            data = tuple(data)
        else: 
            data = self.data[start_idx:end_idx, :]
            # Pad the data accordingly
            if pad_left > 0 or pad_right > 0:
                data = np.pad(data, ((pad_left, pad_right), (0, 0)), 'constant', constant_values=0)
            if self.transform:
                data = self.transform(data)
            data = data.squeeze(0)

        return data, label
