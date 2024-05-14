from torch.nn import Conv2d
import pytorch3d.ops

from .net_utils import *
from models.data_loss  import *


def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """
    _, knn_idx, _ = pytorch3d.ops.knn_points(x, y, K=k+offset)
    return knn_idx[:, :, offset:]


def knn_group(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)


class Aggregator(torch.nn.Module):

    def __init__(self, oper):
        super().__init__()
        assert oper in ('mean', 'sum', 'max')
        self.oper = oper

    def forward(self, x, dim=2):
        if self.oper == 'mean':
            return x.mean(dim=dim, keepdim=False)
        elif self.oper == 'sum':
            return x.sum(dim=dim, keepdim=False)
        elif self.oper == 'max':
            ret, _ = x.max(dim=dim, keepdim=False)
            return ret


def dil_knn(x, y, k=16, d=1, use_fsd=False):
    if len(x.shape) > 3:
        x = x.squeeze(2)
    _,idx,_ = pytorch3d.ops.knn_points(x, y, K=k*d)  # [B N K 2]
    if d > 1:
        if use_fsd:
            idx = idx[:, :, k*(d-1):k*d]
        else:
            idx = idx[:, :, ::d]
    return idx

def get_graph_features(x, idx, return_central=True):
    """
    get the features for the neighbors and center points from the x and inx
    :param x: input features
    :param idx: the index for the neighbors and center points
    :return:
    """
    if len(x.shape) > 3:
        x = x.squeeze(2)
    pc_neighbors = pytorch3d.ops.knn_gather(x,idx)
    if return_central:
        pc_central = x.unsqueeze(-2).repeat(1,1,idx.shape[2],1)
        return pc_central, pc_neighbors
    else:
        return pc_neighbors


def  point_shuffler(input,scale=2):
    B,N,M,C = input.shape
    outputs = input.reshape(B,N,1,C//scale,scale)
    outputs = outputs.permute(0,1,4,3,2)
    outputs = outputs.reshape(B,N*scale,1,C//scale)
    return outputs
 
class edge_sim_conv(Module):
    def __init__(self, out_channel=64, scale=2, k=4, cho_k=4, d=1, n_layer=2):
        super(edge_sim_conv, self).__init__()
        self.out_channel = out_channel
        self.k = k
        self.d = d
        self.n_layer = n_layer
        self.cho_k = cho_k
        self.bn = nn.BatchNorm1d(out_channel)
        self.mlp_conv1 = MLP_CONV_1d(3 * out_channel, [out_channel, out_channel], bn=True)
        self.mlp_conv2 = MLP_CONV(6, [256, out_channel], bn=True)
        for i in range(n_layer - 1):
            self.add_module('d' + str(i), Conv2d(out_channel * scale, out_channel * scale, kernel_size=1))

    def forward(self, pcl, pcl_noise, feature, R=1, idx=None):
        B, N, C = pcl_noise.shape
        _, M, _ = pcl.shape
        R = int(N / M)
        if idx is None:
            _, close_idx, close_point = pytorch3d.ops.knn_points(pcl_noise, pcl, K=1, return_nn=True)
            idx = dil_knn(close_point.squeeze(-2), pcl, self.k, self.d)
        knn_points = pytorch3d.ops.knn_gather(pcl, idx)[:, :, 1:, :]
        knn_feat = pytorch3d.ops.knn_gather(feature, idx)[:, :, 1:, :]
        close_feat = pytorch3d.ops.knn_gather(feature, close_idx).squeeze(-2)
        delta_knn_points = knn_points - pcl_noise.unsqueeze(-2).repeat(1, 1, self.k - 1, 1)
        dst = torch.norm(delta_knn_points, p=2, dim=-1)
        weight = torch.exp(- 10 * dst) / (torch.exp(- 10 * dst).sum(dim=-1, keepdim=True) + 1e-7)
        # delta_points = torch.matmul(weight.unsqueeze(-1).permute(0, 1, 3, 2), delta_knn_points).squeeze(-2)
        delta_points = torch.cat([delta_knn_points, close_point.repeat(1, 1, self.k - 1, 1)], dim=-1)
        point_feature = self.mlp_conv2(delta_points.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        point_feature = torch.matmul(weight.unsqueeze(-1).permute(0, 1, 3, 2), point_feature).squeeze(-2)
        co_feature = torch.matmul(weight.unsqueeze(-1).permute(0, 1, 3, 2), knn_feat).squeeze(-2)
        delta_feature = torch.cat([close_feat, co_feature, point_feature], dim=-1)
        new_feature = self.mlp_conv1(delta_feature.permute(0, 2, 1)).permute(0, 2, 1)
        return new_feature