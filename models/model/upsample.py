
import pytorch3d.ops
import torch
from torch import nn
from models.feature import edge_sim_conv, MLP_CONV_1d
from models.net_utils import normalize_box, denormalize_pcl, farthest_point_sampling


def get_local_frames(pcl_source, pcl_target, feat, k, scale):
    """
    Args:
        pcl_source: (B, N, 3)
        pcl_target: (B, M, 3)
    Returns:
        (B, N, K, 3)
    """
    _, idx, frames = pytorch3d.ops.knn_points(pcl_source, pcl_target, K=k, return_nn=True)  # frames:(B, N, K, 3)
    # frames = (frames - pcl_source.unsqueeze(-2)) / scale  # (B, N, K, 3)
    # feat = pytorch3d.ops.knn_gather(feat, idx)
    return frames, feat

def local_frames_to_pcl(frames, pcl_source, scale):
    """
    Args:
        frames:     (B, N, K, 3)
        pcl_source: (B, N, 3)
    Returns:
        (B, kN, 3)
    """
    B, N, K, d = frames.size()
    frames_n = (frames * scale).reshape(-1, 3)
    frames_denorm = (frames * scale) + pcl_source.unsqueeze(-2)  # (B, N, K, 3)
    pcl_target = frames_denorm.reshape(B, -1, d)
    return pcl_target


def resample_for_frames(pnts, probs, size):
    """
    Args:
        pnts:  Sampled points in the frames, (B, N, k, 3).
        probs: Log probabilities, (B, N, k [, 1])
    Returns:
        (B, N, size, 3)
    """
    B, N, K, _ = pnts.size()
    probs = probs.view(B, N, K)

    idx_top = torch.argsort(probs, dim=-1, descending=True)
    idx_top_p = idx_top[:, :, :size]

    idx_top = idx_top.unsqueeze(-1).expand_as(pnts)  # (B, N, k, 3)
    idx_top = idx_top[:, :, :size, :]  # (B, N, size, 3)

    probs_sel = torch.gather(probs, dim=2, index=idx_top_p)
    pnts_sel = torch.gather(pnts, dim=2, index=idx_top)
    return pnts_sel, probs_sel


def point_dist(value0):
    dist_matrix0 = torch.sqrt(torch.sum(value0 ** 2, dim=1))
    idx_top = torch.argsort(dist_matrix0, dim=-1, descending=False)
    dist_matrix = dist_matrix0[idx_top[1]]
    return dist_matrix


def pair_wise_distance(pcl, target):
    D = pcl.shape
    if len(D) == 2:
        min_dist = 100
        for i in range(D[0]):
            seed = pcl[i].unsqueeze(0)
            _, _, target_0 = pytorch3d.ops.knn_points(seed.unsqueeze(0), target.unsqueeze(0), K=16, return_nn=True)
            value0 = target_0.squeeze(0).squeeze(0) - seed
            dist_matrix = point_dist(value0)
            min_dist = dist_matrix if dist_matrix < min_dist else min_dist
    elif len(D) == 3:
        min_dist = []
        for i in range(D[0]):
            min_dist_1 = 100
            for j in range(D[1]):
                seed = pcl[i][j].unsqueeze(0)
                _, _, target_0 = pytorch3d.ops.knn_points(seed.unsqueeze(0), target[i].unsqueeze(0), K=16,
                                                          return_nn=True)
                value0 = target_0.squeeze(0).squeeze(0) - seed
                dist_matrix = point_dist(value0)
                min_dist_1 = dist_matrix if dist_matrix < min_dist_1 else min_dist_1
            min_dist.append(min_dist_1)
        min_dist = torch.stack(min_dist)
    else:
        print("Input Error!!!")
    return min_dist

class UpsampleNet(nn.Module):

    def __init__(self, backbone, generator, out_channels=512, k=8, aug_noise=0.01, rate_mult=1, box=0.5):
        super().__init__()
        self.feature_net = backbone
        self.generator = generator
        self.edge_sim_conv = edge_sim_conv(out_channel=out_channels, k=k)
        self.mlp = MLP_CONV_1d(out_channels, [384, 256, 128, 64, 3], bn=True)
        self.aug_noise = aug_noise
        self.rate_mult = rate_mult

    def forward(self, pcl_low, rate=2, noisy_points=None, normalize=True, fps=True):
        B, N, C = pcl_low.size()  ## B, N, _:3
        R = int(rate * self.rate_mult)  ## 64

        import time
        t1 = time.time()
        # Normalize
        if normalize:
            pcl_low, center, scale = normalize_box(pcl_low)
            # pcl_low, pcl_high, center, scale = normalize_sphere_v2(pcl_low, pcl_high)

        # Noise Point
        if noisy_points is None:
            noise = torch.randn(B, N * R, 3)
            while sum(noise[noise > 4]) != 0 or sum(noise[noise < -3]) != 0:
                s1 = noise[noise > 4].shape
                noise[noise > 4] = torch.randn(s1)
                s2 = noise[noise < -3].shape
                noise[noise < -3] = torch.randn(s2)
            noise = noise.to(pcl_low) * self.aug_noise
            pcl_noise = pcl_low.repeat(1, R, 1) + noise
        else:
            pcl_noise = noisy_points


        # Feature extraction
        feat = self.feature_net(pcl_low.permute(0,2,1)).permute(0,2,1)

        ### Flow

        # feat_flow = self.edge_sim_conv(pcl_low, pcl_noise, feat, R)

        # feat_flow_re = feat_flow.reshape(-1, feat.size(-1))  ## 196608, 60
        # pcl_high_re = pcl_noise.reshape(-1, pcl_low.size(-1))

        # new_points = self.generator(pcl_high_re, context=feat_flow_re, reverse=True)


        new_points = self.generator(pcl_noise, feat)

        # Reshape and resample
        new_points = new_points.reshape(B, -1, 3)

        pred_noise = new_points - pcl_noise


        # Denormalize
        if normalize:
            new_points = denormalize_pcl(new_points, center, scale)

        if fps:
            new_points = farthest_point_sampling(new_points, rate*pcl_low.size(1))

        return new_points, pred_noise
