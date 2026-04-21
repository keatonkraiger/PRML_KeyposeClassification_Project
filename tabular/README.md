# Python implementation of multi-modal keypose classification

## Requirements
- python 3.8

## Installation
We recommend using a virtual environment to install the requirements such as [conda](https://docs.conda.io/en/latest/miniconda.html).
```bash
conda create -n keypose python=3.8 -y
conda activate keypose
pip install -e .
```

## Data

We provide a smaller version of the official PSUTMM-100 dataset which includes only 3D MoCap and foot pressure data (the only two required modalities for this project). The link to the data may be found in the project description. If you want to use other modalities (MoCap marker, video, etc.) you will need to download the request access to the full dataset from the [official dataset request page](https://forms.office.com/r/ceCgdJGkE7). After downloading the data, you can generate the processed dataset with the following command:

```bash
python create_dataset.py --data_path PSU_Data
```

where the above uses the smaller version of the dataset. Otherwise you would provide the path to the full dataset you downloaded from the official request page.


## Usage overview

The tabular pipeline has two main driver scripts and YAML configs in `configs/`:

- `scripts/train_test_cpu.py`: runs classic sklearn models (e.g., KNN) on tabular features.
- `scripts/train_test_torch.py`: runs a simple PyTorch MLP (and is boilerplate for students to extend).
- `configs/knn.yaml`: example config for KNN.
- `configs/mlp.yaml`: example config for the PyTorch MLP.

Both scripts expect that you have first created the processed dataset with `create_dataset.py` and that the `default.data_path` in the config points to that folder (e.g., `Taiji_dataset`).

### Config structure (common parts)

Key fields shared by `knn.yaml` and `mlp.yaml`:

- `default.data_path`: root folder containing `Subject*/` folders with `mocap_3d_*.npy` and `pressure_*.npy` files.
- `default.subject`: which subjects to use (`"all"` or comma-separated list like `"1,2,3"`).
- `default.keypose_csv`: CSV with keypose centers per (subject, take).
- `default.pose_info_json`: JSON with class names and a `class_map` used for merged evaluation.
- `default.m`, `default.n`: number of frames before/after each keypose center to label as that class.
- `default.balance_classes`: if `true`, down-samples background frames so class 0 does not dominate.
- `default.enable_merged_eval`: if `true`, also evaluates on merged classes using the `class_map`.
- `data.modals`: list of modalities to use (e.g., `pressure`, `mocap_3d`).
- `data.pressure.lod_level`: level-of-detail downsampling for pressure maps (0 = none).
- `data.*.norm`: per-modality normalization (`null`, `MINMAX`, or `ZSCORE`). By default, `mocap_3d` uses `ZSCORE` and `pressure` uses `null`.

### CPU driver

`scripts/train_test_cpu.py` uses sklearn classifiers defined in `classifier_mapping` (currently KNN):

- Config section `classify` controls the model:
  - `classifier`: model name (e.g., `knn`).
  - `n_neighbors`, `weights`, etc.: passed directly into the sklearn constructor.
- Config section `training` controls data handling:
  - `shuffle_train`: if `true`, shuffles the LOSO train split before fitting.

To train a KNN with the default arguments using pressure and mocap_3d modalities:

```bash
python scripts/train_test_cpu.py --cfg configs/knn.yaml
```

This script:

- Loads per-frame data via `create_datasets` (early-fuses modalities if requested).
- Generates frame labels from the keypose CSV (`gen_labels_from_keypose_df`).
- Performs leave-one-subject-out (LOSO) training/testing.
- Optionally evaluates both on the original classes and the merged classes (if `enable_merged_eval: true`).

You of course are not limited to KNNs and may implement our use any existing classifiers. The current code works well with sklearn-style APIs but the code may be easily adapted to work with other classifiers (XGBoost, etc.).

### PyTorch MLP driver

`scripts/train_test_torch.py` is a minimal PyTorch example built on top of `GenericClassifier` in `tabular/models/generic.py`.

- Config section `training` controls the device and hyperparameters:
  - `device`: `'cuda'` to use a GPU if available, otherwise falls back to CPU.
  - `lr`: learning rate.
  - `optimizer`: one of `SGD`, `Adam`, or `AdamW`.
  - `batch_size`: mini-batch size.
  - `epochs`: number of epochs.
  - `shuffle_train`: same meaning as in the CPU script.
- The default model is `SimpleMLP` in `tabular/models/mlp.py`, a one-layer MLP over the fused feature vector.
- Students can swap in their own `torch.nn.Module` by editing the `_build_model` function in `train_test_torch.py`.

To run the example MLP configuration:

```bash
python scripts/train_test_torch.py --cfg configs/mlp.yaml
```

The torch script mirrors the CPU pipeline: it uses the same dataset creation, labeling, LOSO splitting, and merged/unmerged evaluation code; only the classifier and training loop differ.

Again, we provide a simple MLP as a starting point, but you are encouraged to implement or use existing architectures.
