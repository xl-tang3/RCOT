from torch_scatter import scatter
import torch
from torch import nn
from .DGCNN import DGCNN_knn
from .feature import FeatureExtraction


def generate_backbone(config):
    if config.name == 'DGCNN':
        return DGCNN_knn(**config.parameters) if config.parameters is not None else DGCNN_knn()

    if config.name == 'DeFeat':
        return FeatureExtraction(**config.parameters) if config.parameters is not None else FeatureExtraction()

    else:
        print('Backbone', config.name, 'Not implemented.')
        return None
