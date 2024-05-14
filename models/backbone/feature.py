import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList
import pytorch3d.ops
# from .utils import *
from models.backbone.dynamic_edge_conv import DynamicEdgeConv
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.inits import reset

def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """
    _, knn_idx, _ = pytorch3d.ops.knn_points(y, x, K=k+offset)
    return knn_idx[:, :, offset:]

class FeatureExtraction(Module):
    def __init__(self, k=32, input_dim=0, z_dim=0, emb_dims=512, output_dim=1, if_bn=False):
        super(FeatureExtraction, self).__init__()
        self.k = k
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.embedding_dim = emb_dims
        self.output_dim = output_dim

        self.conv1 = DynamicEdgeConv(3, 16)
        self.conv2 = DynamicEdgeConv(16, 48)
        self.conv3 = DynamicEdgeConv(48, 144)
        self.conv4 = DynamicEdgeConv(16+48+144, self.embedding_dim)

        self.linear1 = nn.Linear(self.embedding_dim, 256, bias=False)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, self.output_dim)

        if self.z_dim > 0:
            self.linear_proj = nn.Linear(512, self.z_dim)
            self.dropout_proj = nn.Dropout(0.1)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.conv1)
        reset(self.conv2)
        reset(self.conv3)
        reset(self.conv4)
        reset(self.linear1)
        reset(self.linear2)
        reset(self.linear3)

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def get_edge_index(self, x):
        cols = get_knn_idx(x, x, self.k+1).view(self.batch_size, self.num_points, -1)
        cols = (cols + self.rows_add).view(1, -1)
        edge_index = torch.cat([cols, self.rows], dim=0)
        edge_index, _ = remove_self_loops(edge_index.long())

        return edge_index

    def forward(self, x, disp_feat):
        self.batch_size = x.size(0)
        self.num_points = x.size(1)

        self.rows = torch.arange(0, self.num_points).unsqueeze(0).unsqueeze(2).repeat(self.batch_size, 1, self.k+1).cuda()
        self.rows_add = (self.num_points*torch.arange(0, self.batch_size)).unsqueeze(1).unsqueeze(2).repeat(1, self.num_points, self.k+1).cuda()
        self.rows = (self.rows + self.rows_add).view(1, -1)

        if disp_feat is not None:
            disp_feat = F.relu(self.linear_proj(disp_feat))
            disp_feat = self.dropout_proj(disp_feat)
            x = torch.cat([x, disp_feat], dim=-1)        
        
        edge_index = self.get_edge_index(x)
        x = x.view(self.batch_size*self.num_points, -1)
        x1 = self.conv1(x, edge_index)
        x1 = x1.view(self.batch_size, self.num_points, -1)

        edge_index = self.get_edge_index(x1)
        x1 = x1.view(self.batch_size*self.num_points, -1)
        x2 = self.conv2(x1, edge_index)
        x2 = x2.view(self.batch_size, self.num_points, -1)

        edge_index = self.get_edge_index(x2)
        x2 = x2.view(self.batch_size*self.num_points, -1)
        x3 = self.conv3(x2, edge_index)
        x3 = x3.view(self.batch_size, self.num_points, -1)

        edge_index = self.get_edge_index(x3)
        x3 = x3.view(self.batch_size*self.num_points, -1)
        x_combined = torch.cat((x1, x2, x3), dim=-1)
        x_combined = x_combined.view(self.batch_size*self.num_points, -1)
        x = self.conv4(x_combined, edge_index)
        x = x.view(self.batch_size, self.num_points, -1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        x, x_combined = x.view(self.batch_size, self.num_points, -1), x_combined.view(self.batch_size, self.num_points, -1)

        if self.z_dim > 0:
            return x, x_combined.transpose(2, 1).contiguous()
        else:
            return x, None