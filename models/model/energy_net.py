
import pytorch3d.ops
import torch
from torch import nn
from models.feature import edge_sim_conv, MLP_CONV_1d
from models.net_utils import normalize_box, denormalize_pcl, farthest_point_sampling
from torch.autograd import grad


class EnergyNet(nn.Module):

    def __init__(self, backbone, generator, out_channels=512, k=8, aug_noise=0.01, rate_mult=1, box=0.5, num_steps=2):
        super().__init__()
        self.feature_net = backbone
        self.generator = generator
        self.edge_sim_conv = edge_sim_conv(out_channel=out_channels, k=k)
        self.mlp = MLP_CONV_1d(out_channels, [384, 256, 128, 64, 3], bn=True)
        self.aug_noise = aug_noise
        self.rate_mult = rate_mult
        self.box = box
        self.num_steps = num_steps
        # self.alpha = torch.nn.Parameter(torch.Tensor((1,)))

    def point_filter(self, xyz, min_bound, max_bound, num_points):
        B, N, C = xyz.shape
        filter_xyz = []
        for i in range(B):
            xyz0 = xyz[i]
            for ci in range(3):
                xyz0[xyz[i][:, ci] <= min_bound, :] = 1000
                xyz0[xyz[i][:, ci] >= max_bound, :] = 1000
            valid_inds = xyz0[:, 0] != 1000
            xyz0 = xyz0[valid_inds, :]
            valid_inds = xyz0[:, 1] != 1000
            xyz0 = xyz0[valid_inds, :]
            valid_inds = xyz0[:, 2] != 1000
            xyz0 = xyz0[valid_inds, :]
            xyz0 = xyz0[:num_points, :].unsqueeze(0)
            filter_xyz.append(xyz0)
        xyz = torch.cat(filter_xyz, dim=0)
        return xyz

    def get_energy_and_grad(self, pcl_noise, pcl_low, feat, create_graph, retain_graph, only_inputs):
        pcl_noise.requires_grad=True
        pred_energy = self.generator(pcl_noise, pcl_low, feat)
        d_points = torch.ones_like(pred_energy, requires_grad=False, device=pcl_noise.device)
        pred_energy.requires_grad_(True)
        points_grad = grad(
            outputs=pred_energy,
            inputs=pcl_noise,
            grad_outputs=d_points,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=only_inputs, )[0]
        return pred_energy, points_grad

    def forward(self, pcl_low, rate=2, noisy_points=None, normalize=True, fps=True, state='train'):
        B, N, C = pcl_low.size()  ## B, N, _:3
        R = int(rate * self.rate_mult)  ## 64

        import time
        t1 = time.time()
        # Normalize
        if normalize:
            pcl_low, center, scale = normalize_box(pcl_low)

        # Noise Point
        box_min = -self.box
        box_max = self.box
        if noisy_points is None:
            noise = torch.randn(B, N * R, 3)
            while sum(noise[noise > 4]) != 0 or sum(noise[noise < -3]) != 0:
                s1 = noise[noise > 4].shape
                noise[noise > 4] = torch.randn(s1)
                s2 = noise[noise < -3].shape
                noise[noise < -3] = torch.randn(s2)
            pcl_noise_0 = pcl_low.repeat(1, R, 1) + noise.to(pcl_low) * self.aug_noise
            pcl_noise_0 = self.point_filter(pcl_noise_0, box_min, box_max, N)
        else:
            pcl_noise_0 = noisy_points

        feat = self.feature_net(pcl_low.transpose(1,2)).transpose(1,2)
        ## Feature extraction
        # feat = self.feature_net(pcl_low.permute(0,2,1)).permute(0,2,1)
        if state =='train':
            create_graph = True
            retain_graph = True
            only_inputs = True
        elif state == 'test':
            create_graph = False
            retain_graph = False
            only_inputs = False

        pcl_denoise_all = []
        for _ in range(self.num_steps):
            pcl_denoise = pcl_noise_0.detach()
            energy_score, pcl_grad = self.get_energy_and_grad(pcl_noise=pcl_denoise, pcl_low=pcl_low, feat=feat, \
                                    create_graph=create_graph, retain_graph=retain_graph, only_inputs=only_inputs)
            # energy_score = torch.abs(energy_score)
            pcl_denoise = pcl_denoise - energy_score * pcl_grad
            pcl_denoise_all.append(pcl_denoise)


        pcl_denoise_all = torch.cat(pcl_denoise_all, dim=0)
        energy_score_low, _ = self.get_energy_and_grad(pcl_noise=pcl_low, pcl_low=pcl_low, feat=feat,\
                                create_graph=create_graph, retain_graph=retain_graph, only_inputs=only_inputs)

        # Denormalize
        if normalize:
            for i in range(self.num_steps):
                pcl_denoise_all[i*B:(i+1)*B] = denormalize_pcl(pcl_denoise_all[i*B:(i+1)*B], center, scale)

        return pcl_denoise_all, energy_score_low
