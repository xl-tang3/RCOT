# Residual-Conditioned Optimal Transport (RCOT)
This is the official `Python` implementation of the [ICML 2024]([https://iclr.cc](https://icml.cc/)) paper **Residual-Conditioned Optimal Transport: Towards Structure-Preserving Unpaired and Paired Image Restoration** (https://arxiv.org/abs/2405.02843).

The repository contains reproducible `PyTorch` source code for computing **residual-conditioned optimal transport** (RCOT)  map for structure-preserving and degradation-aware restoration.
The key idea is to integrate the transport residual as a degradation-specific cue into the transport cost, and more crucially, into the transport map via a two-pass conditioning mechanism.
<p align="center"><img src="pics/stochastic_OT_map.png" width="400" /></p>

## Repository structure


## Setup

To run the notebooks, it is recommended to create a virtual environment using either [`conda`](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [`venv`](https://docs.python.org/3/library/venv.html). Once the virtual environment is set up, install the required dependencies by running the following command:

```console
pip install -r requirements.txt
```

Finally, make sure to install `torch` and `torchvision`. It is advisable to install these packages based on your system and `CUDA` version. Please refer to the [official website](https://pytorch.org) for detailed installation instructions.






