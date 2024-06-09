# Residual-Conditioned Optimal Transport (RCOT)
This is the official `Python` implementation of the [ICML 2024]([https://icml.cc/]) paper **Residual-Conditioned Optimal Transport: Towards Structure-Preserving Unpaired and Paired Image Restoration** (https://arxiv.org/abs/2405.02843).

The repository contains reproducible `PyTorch` source code for computing **residual-conditioned optimal transport** (RCOT)  map for structure-preserving and degradation-aware restoration.
The key idea is to integrate the transport residual as a degradation-specific cue into the transport cost, and more crucially, into the transport map via a two-pass conditioning mechanism.
<p align="center"><img src="pics/stochastic_OT_map.png" width="400" /></p>

## Repository structure

## Setup and Pretrained Weights
All the experiments are conducted on Pytorch 2.1.0 with cuda 11.8 an NVIDIA 4090 GPU. For super-resolution, there is an extra preprocessing step. The LR images undergo bicubic rescaling to match the dimensions of their respective high-resolution counterparts.

### Environment setup
```console
conda create -n rcot python=3.8
conda activate rcot
pip install scikit-image
pip install einops
pip install h5py
pip install opencv-python
pip install tqdm
pip install lpips
pip install matplotlib
```
Pretrained weights to reproduce the results in our paper: (https://drive.google.com/drive/folders/16-D1VHGLlkK3DShQVBsDN2WyumlK0jSi)

```console
pip install -r requirements.txt
```

Finally, make sure to install `torch` and `torchvision`. It is advisable to install these packages based on your system and `CUDA` version. Please refer to the [official website](https://pytorch.org) for detailed installation instructions.






