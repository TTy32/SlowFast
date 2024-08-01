<!-- TOC -->

- [Slowfast overview](#slowfast-overview)
    - [Slowfast Model URLs](#slowfast-model-urls)
- [Monitoring and Utilities](#monitoring-and-utilities)
    - [GPU Utilization (htop for GPUs)](#gpu-utilization-htop-for-gpus)
    - [Detectron2 Environment Check](#detectron2-environment-check)
- [Setup](#setup)
    - [❗ Determining appropiate hardware](#-determining-appropiate-hardware)
        - [Used hardware](#used-hardware)
    - [Install NVIDIA and CUDA Drivers](#install-nvidia-and-cuda-drivers)
        - [Scaleway.com Ubuntu 22 Jammy Installation details](#scalewaycom-ubuntu-22-jammy-installation-details)
        - [Verify installation / driver version](#verify-installation--driver-version)
        - [Generic installation details (kept for reference)](#generic-installation-details-kept-for-reference)
- [Uninstall previous drivers](#uninstall-previous-drivers)
- [Uninstall previous drivers](#uninstall-previous-drivers)
- [Cuda 12.1](#cuda-121)
- [Below might be needed according to some SO posts](#below-might-be-needed-according-to-some-so-posts)
    - [Compile FFmpeg with CUDA Support](#compile-ffmpeg-with-cuda-support)
    - [Slowfast dependencies](#slowfast-dependencies)
- [Run Slowfast](#run-slowfast)

<!-- /TOC -->

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

## Slowfast Model URLs

Some were hard to find, listing them here.

* `https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_32x2_R101_50_50.pkl`
* `https://dl.fbaipublicfiles.com/detectron2/COCODetection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl`
* `https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_64x2_R101_50_50.pkl`





# Monitoring and Utilities

## GPU Utilization (htop for GPUs)

```bash
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
```



## Detectron2 Environment Check
- To find out inconsistent CUDA versions:

```bash
python -m detectron2.utils.collect_env
```














# Setup

## ❗ Determining appropiate hardware

The developers of the SlowFast model used NVIDIA Tesla V100 GPUs (likely 8 as observed from config files).

From docs it appears that Slowfast repo uses `pytorch 1.10`.

This is also a requirement for detectron2 (https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

However I wasn't able to install 1.10.1 from mamba. `pytorch 1.3.1` is installable, and also seems to work.

There is a matrix that describes which pytorch version works with which SM_ (Nvidia Compute capability) > thus CUDA version and in turn which Nvidia Card.

There doesn't seem to be official lists for this, however below are some resources:

* [forum post listing many pytorch <> cuda <> sm_ compute capabalities](https://discuss.pytorch.org/t/gpu-compute-capability-support-for-each-pytorch-version/62434/6)
* [NVIDIA docs: sm_ compute capabalities <> hardware](https://developer.nvidia.com/cuda-gpus#compute)
* Regarding SM compatibilities [SO Post](https://stackoverflow.com/a/39116582/1545362):
  ```
  As a general rule, you can run code compiled for lower compute capabilities on higher compute capability hardware, but not the other way around. 
  ```


### Used hardware

Current (2024) VPS / GPU providers often offer newer cards, and usually limited choice.

I used the H100 NVidia 80 GB instance on Scaleway. Sadly this requires SM_90, which in turn requires Pytorch 2.x.

Luckily, I was able to get detectron, and slowfast working with Pytorch 2.x, using several community and custom patches.



## Install NVIDIA and CUDA Drivers


### Scaleway.com Ubuntu 22 Jammy Installation details

Install 555 driver instead of 535:

* `sudo ubuntu-drivers install nvidia:555`

Install CUDA 12.1 :

* `sudo apt-get -y install cuda-12-1`
* Download option, not necessary most of the time: [Download CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)


Now reboot.


Add CUDA 12.1 paths to env / bashrc:
```bash
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
```

Run ldconfig
```bash
sudo ldconfig
```


### Verify installation / driver version
```bash
cat /proc/driver/nvidia/version ; cat /sys/module/nvidia/version
```





### Generic installation details (kept for reference)

```bash
# Uninstall previous drivers

sudo apt-get --purge remove '*cublas*' 'cuda*' 'nsight*' 
sudo apt-get autoremove
sudo apt-get autoclean

# Cuda 12.1

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub
sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-get update
sudo apt-get install cuda-12-1

#Below might be needed according to some SO posts

sudo apt-get install nvidia-modprobe
sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm
```







## Compile FFmpeg with CUDA Support

- Follow the guide for [compiling for Linux](https://docs.nvidia.com/video-technologies/video-codec-sdk/11.1/ffmpeg-with-nvidia-gpu/index.html#compiling-for-linux).









## Slowfast dependencies

Use miniforge (not anaconda, its slower).
Mamba is included in mini forge. Mamba is reimplementation of anaconda in C++ which is designed to be much faster (supports multi threading) for solving complex environments like pytorch.

* Install miniforge
    ```bash
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh
    ```
    * Verify installation: `conda --version ; mamba --version`
* Create env: `conda create -n myenv39 python=3.9`
* Activate env: `conda activate myenv39`
* Install pytroch 2.x (for detectron2): `pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu121`
    * Previous installations:
        * `mamba install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge pytorch == 1.10`
        * `mamba install pytorch==1.3.1 torchvision==0.4.2 torchaudio==0.3.2 cudatoolkit=10.1 -c pytorch -c conda-forge`
* Test pytorch in venv at this point
    ```bash
    $ python
    ```
    ```python
    import torch
    print(torch.rand(3,3).cuda()) 
    print(torch.version.cuda) # linked CUDA version
    ```
    Should return no errors.
* Slowfast dependencies: `pip install simplejson psutil opencv-python tensorboard moviepy pytorchvideo && pip install -U iopath==0.1.9 cython`
* **[Not needed?]** Dependency fixes:
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
* Detectron2: Compile from source for Pytroch 2.x support:
    ```bash
    git clone https://github.com/facebookresearch/detectron2.git ; \
    cd detectron2 ; \
    python -m pip install -e .
    ```
    * Previous installation methods:
        ```bash
        # Correct fvcore dep: https://github.com/facebookresearch/detectron2/issues/4386#issuecomment-1204994435
        pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

        # or
        python -m pip install detectron2 -f \
        https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
        ```
* Dependencies:
    * `conda install av -c conda-forge`
    * `pip3 install scipy`
    * **[Not needed?]**  `pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
* Pytorchvideo (https://github.com/facebookresearch/SlowFast/issues/636):
    ```bash
    git clone https://github.com/facebookresearch/pytorchvideo.git && \
    cd pytorchvideo && \
    pip install -e .
    ```

Now clone & build slowfast:

```bash
git clone https://github.com/TTy32/SlowFast && \
cd SlowFast && \
python setup.py build develop
```

In original docs export pythonpath is mentioned, however it doesn't seem to be necessary: `export PYTHONPATH=~/slowfast:$PYTHONPATH`


# Run Slowfast

Demo:

```bash
python tools/run_net.py --cfg c3.yaml
```


