# Residual-Conditioned Optimal Transport (RCOT)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2405.02843)

<hr />

> **Abstract:** *Deep learning-based image restoration methods generally struggle with faithfully preserving the structures of the original image.  In this work, we propose a novel Residual-Conditioned Optimal Transport (RCOT) approach, which models image restoration as an optimal transport (OT) problem for both unpaired and paired settings,  introducing the transport residual as a unique degradation-specific cue for both the transport cost and the transport map. Specifically, we first formalize a Fourier residual-guided OT objective by incorporating the degradation-specific information of the residual into the transport cost. We further design the transport map as a two-pass RCOT map that comprises a base model and a refinement process, in which the transport residual is computed by the base model in the first pass and then encoded as a degradation-specific embedding to condition the second-pass restoration. By duality, the RCOT problem is transformed into a minimax optimization problem, which can be solved by adversarially training neural networks. Extensive experiments on multiple restoration tasks show that RCOT achieves competitive performance in terms of both distortion measures and perceptual quality, restoring images with more faithful structures as compared with state-of-the-art methods.* 
<hr />

This is the official `Python` implementation of the [ICML 2024](https://icml.cc/) paper [**Residual-Conditioned Optimal Transport: Towards Structure-Preserving Unpaired and Paired Image Restoration**](https://arxiv.org/pdf/2405.02843).

The repository contains reproducible `PyTorch` source code for computing **residual-conditioned optimal transport** (RCOT)  map for structure-preserving and degradation-aware restoration.
The key idea is to **integrate the transport residual as a degradation-specific cue into the transport cost, and more crucially, into the transport map via a two-pass conditioning mechanism**.
<p align="center"><img src="pics/concept3.png" width="1200" /></p>
<p align="center"><img src="pics/OTModel.png" width="1200" /></p>



##  Setup and Pretrained Weights
This repository is built in PyTorch 2.1.1 and tested on Ubuntu 18.04 environment (Python3.8, CUDA11.8). The LR images undergo bicubic rescaling to match the dimensions of their respective high-resolution counterparts.
Follow these instructions.
###  Dependencies Installation


1. Clone our repository
```
git clone https://github.com/xl-tang3/RCOT.git
cd RCOT
```

2. Create conda environment
The Conda environment used can be recreated using the env.yml file
```
conda env create -f env.yml
```

or


```
conda create -n RCOT python=3.8
conda activate RCOT
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install scikit-image
pip install einops
pip install h5py
pip install opencv-python
pip install tqdm
pip install lpips
pip install matplotlib
```

###  Dataset Download and Preperation

All the 5 datasets used in the paper can be downloaded from the following locations:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (OTS)

Super-Resolution: [DIV2K x4](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
 The test directory after setup:
```
├───dehaze
│   ├───input
│   └───target
├───denoise
│   ├───bsd68
│   └───Kodak24
└───derain
    └───Rain100L
        ├───input
        └───target
```
###  Pretrained Weights

Here are [Pretrained weights](https://drive.google.com/drive/folders/16-D1VHGLlkK3DShQVBsDN2WyumlK0jSi) to reproduce the results in our paper.









