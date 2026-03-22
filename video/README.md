# KeyposeClassificatio# Installation

## Dataset Preparation
Please run the following script to generate the video dataset:

```
python scripts/prepare_data.py --psutmm_root path/to/PSUU100 --crop_videos --video_splice --create_labels
```

The above script will crop each video around the pelvis, splice the videos around keyframes, and create training and testing labels for LOSO cross validation. 

## Requirements

We suggest first creating a virtual environment using `conda`:

```
conda create -n SlowFast python=3.10 -y
conda activate SlowFast
```

This repository depends on Meta's [PySlowFast](https://github.com/facebookresearch/slowfast). This will require installing several dependencies. Please follow the official SlowFast installation instructions [here](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md).

I've had success with the following commands (from KeyposeClassification/video):

```
conda create -n SlowFast python=3.10 -y
conda activate SlowFast
pip install -U pip setuptools wheel
pip install -U torch torchvision numpy cython ninja
conda install -y -c conda-forge av ffmpeg
pip install -U simplejson psutil opencv-python tensorboard
pip install -U iopath
pip install -U 'git+https://github.com/facebookresearch/fvcore.git'
pip install -U 'git+https://github.com/facebookresearch/fairscale.git'
pip install -U "git+https://github.com/facebookresearch/pytorchvideo.git"
pip install --no-build-isolation -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
cd ..
git clone https://github.com/facebookresearch/detectron2 detectron2
pip install --no-build-isolation -e detectron2
cd video
pip install -r requirements.txt
pip install -e .
```

The installation can be tricky and you may need to troubleshoot dependency issues on your machine.

## Dataset Generation

You first need to download PSUTMM-100. You can then run the following script to generate the video dataset:

```
python scripts/prepare_data.py --psutmm_root path/to/PSUU100 --crop_videos --video_splice --create_labels
```

## Training

Training experiments are defined in `configs/`. You can run the following command to train a model:

```python tools/benchmark.py --cfg configs/PSUTMM/X3D_S.yaml
```

