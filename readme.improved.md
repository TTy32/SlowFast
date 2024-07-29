
# Slowfast overview

Pre-trained Models:

- **Kinetics**: Capture a wide range of human actions.
- **AVA (Atomic Visual Actions)**: Detects spatio-temporal human actions in videos. Models are fine-tuned for tasks requiring detailed action localization.
- **X3D (Expandable 3D Networks)**: Efficient video models that expand along multiple axes (temporal, spatial, depth) for optimized video understanding.

Model Hierarchy:

```
        ImageNet
           |
  -------------------
  |        |        |
Kinetics   AVA     X3D
  |        |
  AVA      |
           X3D
```

Note that `Detectron2` (also developed by Facebook AI Research) handles object segmentation in training and runtime inference.

# Model URLs

Some were hard to find, listing them here.

* `https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_32x2_R101_50_50.pkl`
* `https://dl.fbaipublicfiles.com/detectron2/COCODetection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl`
* `https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_64x2_R101_50_50.pkl`



# Scaleway GPU VPS details

Install 555 driver instead of 535:

```bash
sudo ubuntu-drivers install nvidia:555
```

Install / [Download CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)

```bash
sudo apt-get -y install cuda-12-1
```

# Monitoring and Utilities

## GPU Utilization (htop for GPUs)

```bash
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
```

# Check NVIDIA Driver Version
```bash
cat /proc/driver/nvidia/version
cat /sys/module/nvidia/version
```

## Detectron2 Environment Check
- To find out inconsistent CUDA versions:

```bash
python -m detectron2.utils.collect_env
```









## Compile FFmpeg with CUDA Support

- Follow the guide for [compiling for Linux](https://docs.nvidia.com/video-technologies/video-codec-sdk/11.1/ffmpeg-with-nvidia-gpu/index.html#compiling-for-linux).






# Installing NVIDIA and CUDA Drivers



## Install CUDA 12.1 (12.5 is not supported by pytorch!)

Uninstall previous drivers

```
sudo apt-get --purge remove '*cublas*' 'cuda*' 'nsight*' 
sudo apt-get autoremove
sudo apt-get autoclean
```

Cuda 12.1

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub
sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-get update
sudo apt-get install cuda-12-1
```

Below might be needed according to some SO posts

```
sudo apt-get install nvidia-modprobe
sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm
```


### Set CUDA Paths

```bash
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
```




# Install miniforge

Use miniforge (NOT anaconda, its slow).
Mamba is included in mini forge. Mamba is reimplementation of anaconda in C++ which is designed to be much faster (supports multi threading) for solving complex environments like pytorch.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
  bash Miniforge3-Linux-x86_64.sh
```

Verify installation:

```bash
conda --version
mamba --version
```

# Create env

```bash
conda create -n myenv39 python=3.9
```

# Activate env

```bash
conda activate myenv39
```

# Install pytorch

`pytorch == 1.10` (requirement for detectron2 https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)

```bash
mamba install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

`pytroch == 1.3.1` also seems to work:

```bash
mamba install pytorch==1.3.1 torchvision==0.4.2 torchaudio==0.3.2 cudatoolkit=10.1 -c pytorch -c conda-forge
```

## Necessary dependency fixes

Then after pytorch install:

```bash
# solves error: ImportError: cannot import name 'packaging' from 'pkg_resources'
python -m pip install setuptools==69.5.1

# https://github.com/pytorch/pytorch/issues/123097
mamba install mkl==2024.0

# https://stackoverflow.com/a/71991964/1545362
pip install numpy==1.23.0

# https://github.com/pytorch/pytorch/issues/69894
pip install setuptools==59.5.0
```



## Test pytorch in venv

```bash
$ python
```

```python
import torch
print(torch.rand(3,3).cuda()) 
print(torch.version.cuda) # linked CUDA version
```

Should return no errors.







# Slowfast dependencies


Slowfast dependencies

```bash
pip install simplejson psutil opencv-python tensorboard moviepy pytorchvideo && \
    pip install -U iopath cython
```

Detectron2

```bash
# Correct fvcore dep: https://github.com/facebookresearch/detectron2/issues/4386#issuecomment-1204994435
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
```

Newer Detectron2 version:

```bash
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

Others:

```bash
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
conda install av -c conda-forge
pip3 install scipy
```

Pytorchvideo

https://github.com/facebookresearch/SlowFast/issues/636

```bash
git clone https://github.com/facebookresearch/pytorchvideo.git && \
    cd pytorchvideo && \
    pip install -e .
```

Clone & build slowfast:

```bash
git clone https://github.com/facebookresearch/slowfast && \
    cd slowfast && \
    python setup.py build develop
```

```bash
# In original docs export pythonpath is mentioned, however it doesn't seem to be necessary:
export PYTHONPATH=~/slowfast:$PYTHONPATH
```


# Run Slowfast

Demo:

```bash
python tools/run_net.py --cfg c3.yaml
```


