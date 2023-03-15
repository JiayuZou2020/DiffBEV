# DiffBEV
Official PyTorch implementation of introducing conditional diffusion model to BEV perception

BEV perception is of great importance in the field of autonomous driving, serving as the cornerstone of planning, controlling, and motion prediction. The quality of the BEV feature highly affects the performance of BEV perception. However, taking the noises in camera parameters and LiDAR scans into consideration, we usually obtain BEV representation with harmful noises. Diffusion models naturally have the ability to denoise noisy samples to the ideal data, which motivates us to utilize the diffusion model to get a better BEV representation. In this work, we propose an end-to-end framework, named DiffBEV, to exploit the potential of diffusion model to generate a more comprehensive BEV representation. To the best of our knowledge, we are the first to apply diffusion model to BEV perception. In practice, we design three types of conditions to guide the training of the diffusion model which denoises the coarse samples and refines the semantic feature in a progressive way. What's more, a cross-attention module is leveraged to fuse the context of BEV feature and the semantic content of conditional diffusion model. DiffBEV achieves a 25.9% mIoU on the nuScenes dataset, which is 6.2% higher than the best-performing existing approach. Quantitative and qualitative results on multiple benchmarks demonstrate the effectiveness of DiffBEV in BEV semantic segmentation and 3D object detection tasks. The code will be available soon.

## Dataset
Please prepare the datasets as follows.

### nuScenes

### KITTI Raw

### KITTI Odometry

### KITTI 3D Object

## Installation
DiffBEV is tested on:
* Python 3.7/3.8
* CUDA 11.1
* Torch 1.9.1

Please check [install](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation.
* Create a conda environment for the project.
```python
conda create -n diffbev python=3.7
conda activate diffbev
```
* Install Pytorch following the [instruction](https://pytorch.org/get-started/locally/).
`conda install pytorch torchvision -c pytorch`
* Install [mmcv](https://github.com/open-mmlab/mmcv)

```python
pip install -U openmim
mim install mmcv-full
```
* Git clone this repository
* Install and compile the required packages.
```python
cd mmsegmentation
pip install -v -e .
```
## Citation
If you find our work is helpful for your research, please consider citing as follows.

## Acknowledgement
Our work is partially based on the following open-sourced projects: [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [VPN](https://github.com/pbw-Berwin/View-Parsing-Network), [PYVA](https://github.com/JonDoe-297/cross-view), [PON](https://github.com/tom-roddick/mono-semantic-maps), [LSS](https://github.com/nv-tlabs/lift-splat-shoot). 
Thanks for their contribution to the research community.
