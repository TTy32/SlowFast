<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Slowfast overview](#slowfast-overview)
   * [Slowfast Model URLs](#slowfast-model-urls)
   * [Transfer learning](#transfer-learning)
- [Monitoring and Utilities](#monitoring-and-utilities)
   * [GPU Utilization (htop for GPUs)](#gpu-utilization-htop-for-gpus)
   * [Detectron2 Environment Check](#detectron2-environment-check)
- [Setup](#setup)
   * [❗ Determining appropiate hardware](#-determining-appropiate-hardware)
      + [Used hardware](#used-hardware)
   * [Install NVIDIA and CUDA Drivers](#install-nvidia-and-cuda-drivers)
      + [Scaleway.com Ubuntu 22 Jammy Installation details](#scalewaycom-ubuntu-22-jammy-installation-details)
      + [Verify installation / driver version](#verify-installation-driver-version)
      + [Generic installation details (kept for reference)](#generic-installation-details-kept-for-reference)
   * [Compile FFmpeg with CUDA Support](#compile-ffmpeg-with-cuda-support)
   * [Slowfast dependencies](#slowfast-dependencies)
- [Run Slowfast](#run-slowfast)
   * [Common issues](#common-issues)


<!-- TOC end -->


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


## Transfer learning



The SlowFast repository includes configurations and checkpoints for models that have undergone a two-stage training process (Kinetics pre-training followed by AVA fine-tuning).

It is not explicitly mentioned, but the [Model Zoo](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md) suggests this.


## Breakdown of config filename meanings


**Model Type (e.g., SLOWFAST, SLOW, I3D, C2D, X3D, MVIT, VIT, R2PLUS1D, CSN)**:

- These are different models or architectures used for video classification.
- **SLOWFAST**: Refers to the SlowFast network, which processes video at different frame rates.
- **SLOW**: Refers to the Slow pathway of the SlowFast network.
- **I3D**: Inflated 3D ConvNet, an architecture for video classification.
- **C2D**: 2D convolution applied to video.
- **X3D**: A family of efficient video models.
- **MVIT**: Multiscale Vision Transformers.
- **VIT**: Vision Transformers.
- **R2PLUS1D**: 3D convolution factorized into 2D spatial and 1D temporal convolutions.
- **CSN**: Channel-Separated Convolutional Networks.

**Frame Sampling and Temporal Stride (e.g., 32x2, 8x8, 16x8, 64x2, 4x16)**:

- These numbers represent the number of frames and the temporal stride between them.
- The first number (e.g., 32, 8, 16) typically indicates the number of frames sampled.
- The second number (e.g., 2, 8, 4, 16) indicates the temporal stride, or how far apart in time the sampled frames are.
- For example, **32x2** means 32 frames are sampled with a temporal stride of 2.

**Backbone Network (e.g., R50, R101)**:

- This part specifies the backbone network used for feature extraction.
- **R50**: ResNet-50, a ResNet with 50 layers.
- **R101**: ResNet-101, a ResNet with 101 layers.

**Additional Modifiers and Variants (e.g., SHORT, IN1K, NLN, MULTIGRID, v2.1, stepwise)**:

- **SHORT**: Usually indicates a shorter training duration or a smaller dataset subset.
- **IN1K**: Implies the model is pre-trained or tested on the ImageNet-1K dataset.
- **NLN**: Non-local networks, a type of neural network module that can capture long-range dependencies.
- **MULTIGRID**: Refers to multigrid training, a method to train models faster and more effectively.
- **v2.1**: Indicates a version number of the configuration.
- **stepwise**: Refers to stepwise training, a technique where training is done in phases with changing parameters.

Here are some examples:

- **SLOWFAST_32x2_R50_SHORT.yaml**:
    - **SLOWFAST** model.
    - 32 frames sampled with a stride of 2.
    - **ResNet-50** backbone.
    - Shorter training duration or smaller subset.

- **SLOW_8x8_R50.yaml**:
    - **SLOW** pathway of the SlowFast network.
    - 8 frames sampled with a stride of 8.
    - **ResNet-50** backbone.

- **I3D_8x8_R101.yaml**:
    - **I3D** model.
    - 8 frames sampled with a stride of 8.
    - **ResNet-101** backbone.

- **MVIT_B_16_CONV.yaml**:
    - **MVIT** model.
    - B variant.
    - 16 frames.
    - Convolutional layers included.

- **SLOWFAST_8x8_R50_stepwise.yaml**:
    - **SLOWFAST** model.
    - 8 frames sampled with a stride of 8.
    - **ResNet-50** backbone.
    - Stepwise training.




**Configuration Files**

The configuration files for training on AVA often refer to using Kinetics pre-trained weights as a base. This is typically done by setting the `TRAIN.CHECKPOINT_FILE_PATH` parameter to point to a Kinetics pre-trained model:
```yaml
TRAIN:
  ENABLE: True
  DATASET: ava
  BATCH_SIZE: 8
  EVAL_PERIOD: 1
  CHECKPOINT_FILE_PATH: "path_to_kinetics_pretrained_model"
```
See config files: https://github.com/facebookresearch/SlowFast/tree/main/configs/AVA









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


## Common issues

**ValueError: loaded state dict has a different number of parameter groups**

Details of this issue reported in below issues:
https://github.com/facebookresearch/SlowFast/issues/108
https://github.com/facebookresearch/SlowFast/issues/478

I've made a change to utils/checkpoint.py that skips loading the optimizer when param groups differ. The code implies we don't strictly need the optimzer when fine-tuning.


**UnboundLocalError: local variable 'inputs' referenced before assignment**

```bash
[08/07 09:36:01][INFO] train_net.py:  614: Start epoch: 1
Traceback (most recent call last):
  File "/home/u/SlowFast/tools/run_net.py", line 50, in <module>
    main()
  File "/home/u/SlowFast/tools/run_net.py", line 26, in main
    launch_job(cfg=cfg, init_method=args.init_method, func=train)
  File "/home/u/SlowFast/slowfast/utils/misc.py", line 421, in launch_job
    func(cfg=cfg)
  File "/home/u/SlowFast/tools/train_net.py", line 662, in train
    train_epoch(
  File "/home/u/SlowFast/tools/train_net.py", line 275, in train_epoch
    del inputs
UnboundLocalError: local variable 'inputs' referenced before assignment
```

Fix: the number of training videos is smaller than `batch_size`, changing `TRAIN.BATCH_SIZE` to be smaller than the number of training videos in the yaml file or config/defaults.py can solve the problem.

Source: https://github.com/facebookresearch/SlowFast/issues/547#issuecomment-1541990588
Source: https://github.com/facebookresearch/SlowFast/issues/257



# Datasets

## AVA

```
├── annotations                                             AVA.ANNOTATION_DIR
│   ├── ava_action_list_v2.1_for_activitynet_2018.pbtxt
│   ├── ava_action_list_v2.2.pbtxt
│   ├── ava_action_list_v2.2_for_activitynet_2019.pbtxt      AVA.LABEL_MAP_FILE # not used ?
│   ├── ava_action_list_v2.2_for_activitynet_2019.pbtxt.1
│   ├── ava_included_timestamps_v2.2.txt
│   ├── ava_test_excluded_timestamps_v2.1.csv
│   ├── ava_test_excluded_timestamps_v2.2.csv
│   ├── ava_test_predicted_boxes.csv
│   ├── ava_test_v2.2.csv
│   ├── ava_train_excluded_timestamps_v2.1.csv
│   ├── ava_train_excluded_timestamps_v2.2.csv
│   ├── ava_train_predicted_boxes.csv
│   ├── ava_train_v2.1.csv
│   ├── ava_train_v2.2.csv                                   AVA.TRAIN_GT_BOX_LISTS (default = "ava_train_v2.2.csv") (used for train only) # bbox for train
│   ├── ava_val_excluded_timestamps_v2.1.csv                 AVA.EXCLUSION_FILE
│   ├── ava_val_excluded_timestamps_v2.2.csv
│   ├── ava_val_predicted_boxes.csv                          AVA.TEST_PREDICT_BOX_LISTS # bbox for test
│   ├── ava_val_v2.1.csv                                     AVA.GROUNDTRUTH_FILE
│   ├── ava_val_v2.2.csv
│   ├── person_box_67091280_iou75
│   ├── person_box_67091280_iou90
│   ├── test.csv
│   ├── train.csv
│   └── val.csv
├── ava_file_names_trainval_v2.1.txt
├── frame_lists                                             AVA.FRAME_LIST_DIR
│   ├── train.csv                                           ├── AVA.TRAIN_LISTS (default = train.csv)
│   └── val.csv                                             └── AVA.TEST_LIST   (default = val.csv)
└── frames                                                  AVA.FRAME_DIR
    ├── -5KQ66BBWC4
    ├── _-Z6wFjXtGQ
    ├── _145Aa_xkuE
    ├── _7oWZq_s_Sk
    ├── _Ca3gOdOHxU
    ├── _a9SWtcaNj8
    ├── _eBah6c5kyA
    ├── _ithRWANKB0
    └── _mAfwH6i90E
```