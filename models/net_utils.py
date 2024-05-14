import math
import numpy as np
import torch
from torch.nn import Module
from torch import nn
import pytorch3d

class FCLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        if activation is None:
            self.activation = torch.nn.Identity()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        return self.activation(self.linear(x))

def standard_normal_logprob(z):
    # dim = z.size(-1)
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

## 一般log似然
def standard_logprob(z, s):
    logZ = -0.5 * math.log(2 * math.pi) - math.log(s)
    return logZ - z.pow(2) / (2 * s**2)


## std 也相应的改为s
def truncated_normal_(tensor, mean=0, std=1, trunc_std=0.8):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape  ## (B, N):(196608, 3)
    ## tensor.new_empty(size): 返回大小为size的未初始化的数据,tensor.normal_(): 用根据mean和std采样得到的点填充tensor
    tmp = tensor.new_empty(size + (4,)).normal_() ## Tensor(196608, 3, 4)
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def normalize_sphere(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    return pc, center, scale

def pc_normalize(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    center = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] # (B, N, 1)
    pc = pc / scale
    return pc, center, scale

def normalize_sphere_v2(pc, pc_h, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    pc_h = pc_h - center
    pc_h = pc_h / scale
    return pc, pc_h, center, scale


def normalize_std(pc, std=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    center = pc.mean(dim=-2, keepdim=True)   # (B, 1, 3)
    pc = pc - center
    scale = pc.view(pc.size(0), -1).std(dim=-1).view(pc.size(0), 1, 1) / std
    pc = pc / scale
    return pc, center, scale


def normalize_pcl(pc, center, scale):
    return (pc - center) / scale


def denormalize_pcl(pc, center, scale):
    return pc * scale + center

def normalize_box(pc):
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2  # (B, 1, 3)
    scale = (p_max - p_min).max()
    pc = pc - center
    pc = pc / scale
    return pc, center, scale

def farthest_point_sampling_pair(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    unsampled = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts] # (24,)
        idx = idx.cpu().detach().numpy()
        idx_ori = np.arange(0, pcls.size(1), 1)
        idx_re = np.delete(idx_ori, idx)
        sampled.append(pcls[i:i+1, idx, :])
        unsampled.append(pcls[i:i+1, idx_re, :])
    sampled = torch.cat(sampled, dim=0)
    unsampled = torch.cat(unsampled, dim=0)
    return sampled, unsampled

def farthest_point_sampling(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts] # (24,)
        sampled.append(pcls[i:i+1, idx, :])
    sampled = torch.cat(sampled, dim=0)
    return sampled

def farthest_point_feature_sampling(pcls, feature, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    sample_feature = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts] # (24,)
        sampled.append(pcls[i:i+1, idx, :])
        sample_feature.append(feature[i:i + 1, idx, :])
    sampled = torch.cat(sampled, dim=0)
    sample_feature = torch.cat(sample_feature, dim=0)
    return sampled, sample_feature

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, N)
    return dist

def query_ball_point(xyz, new_xyz, radius, nsample):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(xyz, new_xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:,:,:nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class MLP(Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_CONV(Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv2d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)
      
class MLP_CONV_1d(Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV_1d, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out

def batch_density(pcl_low, pcl_high, k=10):
    """
    Args:
        pcl_low: B,N,3
        pcl_high: B,R*N,3
        k: the KNN of each point, int

    Returns:
        density of the batch: B, N
    """
    pcl_low = pcl_low[:, :, :3]
    pcl_high = pcl_high[:, :, :3]
    B, N, _ = pcl_low.shape
    knn_dst, knn_idx, knn_data = pytorch3d.ops.knn_points(pcl_low, pcl_high, K=k, return_nn=True)  # [B N K]
    # knn_data = pytorch3d.ops.knn_gather(pcl, knn_idx)  # [B N K 3]
    # mean_distance = (knn_data - pcl_low[:, :, None, :]).norm(dim=-1).mean(dim=-1)
    mean_distance = knn_dst.mean(dim=-1)
    dense = k / (mean_distance + 1e-7)
    inf_mask = torch.isinf(dense)
    max_val = dense[~inf_mask].max()
    dense[inf_mask] = max_val
    return dense

def angle_diff(pc, k):
    """
    Args:
        pc: B, N, 3
        k: KNN

    Returns:
        angle: the angle of each point, B, N
    """
    INNER_PRODUCT_THRESHOLD = math.pi / 2
    _, idx, pc_knn = pytorch3d.ops.knn_points(pc, pc, K=k, return_nn=True)

    inner_prod = (pc_knn * pc.unsqueeze(-2)).sum(dim=-1)
    inner_prod[inner_prod > 1] = 1
    inner_prod[inner_prod < -1] = -1
    angle = torch.acos(inner_prod)
    angle[angle > INNER_PRODUCT_THRESHOLD] = math.pi - angle[angle > INNER_PRODUCT_THRESHOLD]
    angle = angle.sum(dim=-1)
    return angle


