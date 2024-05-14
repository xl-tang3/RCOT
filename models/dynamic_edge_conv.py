import torch
from torch.nn import Sequential as Seq, Linear, BatchNorm1d as BN, ReLU
# from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       BN(out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels),
                       BN(out_channels),
                       ReLU())
        self.lin = Seq(Linear(in_channels, out_channels),
                       BN(out_channels),
                       ReLU())

    def forward(self, x, edge_index):

        x_pair = (x, x)
        out_1 = self.propagate(edge_index, x=x_pair[0])
        out_2 = self.lin(x_pair[1])

        return out_1 + out_2

    def message(self, x_i, x_j):

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def forward(self, x, edge_index):
        return super().forward(x, edge_index)