import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from torch.autograd import grad
import math
import time

from models.generator.Offset_NVF import Offset_NVF
from .cnf import build_flow as build_cnf_flow
from .acl import build_flow as build_acl_flow
from .iter_denoise import IterDenoise


def generate_generator(config):
    if config.name == 'cnf':
        return build_cnf_flow(config.other_cfg, **config.parameters) if config.parameters is not None else build_cnf_flow()
        
    elif config.name == 'acl':
        return build_acl_flow(config.other_cfg, **config.parameters) if config.parameters is not None else build_acl_flow()   

    elif config.name == 'offset':
        return Offset_NVF(**config.parameters) if config.parameters is not None else Offset_NVF()

    elif config.name == 'iter_denoise':
        return IterDenoise()
    
    else:
        print('Generator', config.name, 'Not implemented.')
        return None

