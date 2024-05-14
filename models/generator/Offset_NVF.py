import torch
from torch import nn
from pytorch3d.ops import knn_points, knn_gather


class Offset_NVF(nn.Module):
    def __init__(self, emb_dims=256, k=8, pos_dim=128, out_dim=128):
        super(Offset_NVF, self).__init__()
        self.k_nearest  = k
        self.fc_pos = nn.Linear(9, pos_dim)
        self.fc_0 = nn.Conv1d((out_dim+pos_dim)*k, emb_dims * 2, 1)
        self.fc_1 = nn.Conv1d(emb_dims*2, emb_dims, 1)
        self.fc_2 = nn.Conv1d(emb_dims , emb_dims, 1)
        self.fc_3 = nn.Conv1d(emb_dims , emb_dims, 1)
        self.fc_4 = nn.Conv1d(emb_dims , emb_dims, 1)
        self.fc_out = nn.Conv1d(emb_dims, 1, 1)
        self.actvn = nn.ReLU()

    def forward(self, query, xyz, pcl_feat):
        assert pcl_feat.shape[1] == xyz.shape[1]
        B, Q, _ = query.shape
        _, idx, q_nearest_k = knn_points(query, xyz, K=self.k_nearest, return_nn=True)
        q_nearest_k_feature = knn_gather(pcl_feat, idx)
        qk = query.unsqueeze(2).expand(B, Q, self.k_nearest, 3)
        pos_feature = self.actvn(self.fc_pos(torch.cat([qk, q_nearest_k, q_nearest_k-qk], dim=-1)))
        features = torch.cat([pos_feature, q_nearest_k_feature], dim=-1).view(B, Q, -1).transpose(1,2).contiguous() # [B, Q, K, F] => [B, Q, K*F] => [B, K*F, Q]
        features = self.actvn(self.fc_0(features))
        features = self.actvn(self.fc_1(features))
        features = self.actvn(self.fc_2(features))
        net = self.actvn(self.fc_3(features))
        net = self.actvn(self.fc_4(net))
        net = self.fc_out(net)
        out = net.transpose(1,2)
        return out