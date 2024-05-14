import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import grad
import math
import numpy as np
import torch.nn.init as init

#### The knn function used in graph_feature ####
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
#### The edge_feature used in DGCNN ####
def get_graph_feature(x, k=4):
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature

#### The DGCNN network ####
class DGCNN_knn(nn.Module):
    def __init__(self, emb_dims=512, if_bn=False):
        super(DGCNN_knn, self).__init__()
        self.if_bn = if_bn
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv1.weight, gain=1.0)
        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv2.weight, gain=1.0)
        self.conv3 = nn.Conv2d(64*2, 128, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv3.weight, gain=1.0)
        self.conv4 = nn.Conv2d(128*2, 256, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv4.weight, gain=1.0)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv5.weight, gain=1.0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)
    def forward(self, x, if_relu_atlast = False):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x) # This sub model get the graph-based features for the following 2D convs
        # The x is similar with 2D image
        if self.if_bn == True: x = F.relu(self.bn1(self.conv1(x)))
        else: x = F.relu(self.conv1(x))
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1)
        if self.if_bn == True: x = F.relu(self.bn2(self.conv2(x)))
        else: x = F.relu(self.conv2(x))
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2)
        if self.if_bn == True: x = F.relu(self.bn3(self.conv3(x)))
        else: x = F.relu(self.conv3(x))
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3)
        if self.if_bn == True: x = F.relu(self.bn4(self.conv4(x)))
        else: x = F.relu(self.conv4(x))
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1).unsqueeze(3)
        if if_relu_atlast == False:
            return torch.tanh(self.conv5(x)).view(batch_size, -1, num_points)
        x = F.relu(self.conv5(x)).view(batch_size, -1, num_points)
        return x

