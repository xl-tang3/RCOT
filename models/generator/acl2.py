from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, context, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, ctx=context, logpx=logpx, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, ctx=context, logpx=logpx, reverse=reverse)
            return x, logpx


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SharedDot(nn.Module):
    def __init__(self, in_features, out_features, n_channels, bias=False,
                 init_weight=None, init_bias=None):
        super(SharedDot, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels
        self.init_weight = init_weight
        self.init_bias = init_bias
        self.weight = nn.Parameter(torch.Tensor(n_channels, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_channels, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weight:
            nn.init.uniform_(self.weight.data, a=-self.init_weight, b=self.init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight.data, a=0.)
        if self.bias is not None:
            if self.init_bias:
                nn.init.constant_(self.bias.data, self.init_bias)
            else:
                nn.init.constant_(self.bias.data, 0.)

    def forward(self, input):
        output = torch.matmul(self.weight, input.unsqueeze(1))
        if self.bias is not None:
            output.add_(self.bias.unsqueeze(0).unsqueeze(3))
        output.squeeze_(1)
        return output


class CondRealNVPFlow3D(nn.Module):
    def __init__(self, f_n_features, g_n_features,
                 weight_std=0.01, warp_inds=[0],
                 centered_translation=False, eps=1e-6):
        """
        Args:
            f_n_features: Hidden dim.
            g_n_features: Context dim.
        """
        super().__init__()
        self.f_n_features = f_n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.warp_inds = warp_inds
        self.keep_inds = [0, 1, 2]
        self.centered_translation = centered_translation
        self.register_buffer('eps', torch.from_numpy(np.array([eps], dtype=np.float32)))
        for ind in self.warp_inds:
            self.keep_inds.remove(ind)

        self.T_mu_0 = nn.Sequential(OrderedDict([
            ('mu_sd0', SharedDot(len(self.keep_inds), self.f_n_features, 1)),
            ('mu_sd0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('mu_sd0_relu', nn.ReLU(inplace=True)),
            ('mu_sd1', SharedDot(self.f_n_features, self.f_n_features, 1)),
            ('mu_sd1_bn', nn.BatchNorm1d(self.f_n_features, affine=False))
        ]))

        self.T_mu_0_cond_w = nn.Sequential(OrderedDict([
            ('mu_sd1_film_w0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('mu_sd1_film_w0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('mu_sd1_film_w0_swish', Swish()),
            ('mu_sd1_film_w1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_mu_0_cond_b = nn.Sequential(OrderedDict([
            ('mu_sd1_film_b0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('mu_sd1_film_b0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('mu_sd1_film_b0_swish', Swish()),
            ('mu_sd1_film_b1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_mu_1 = nn.Sequential(OrderedDict([
            ('mu_sd1_relu', nn.ReLU(inplace=True)),
            ('mu_sd2', SharedDot(self.f_n_features, len(self.warp_inds), 1, bias=True))
        ]))

        with torch.no_grad():
            self.T_mu_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_w[-1].bias.data, 0.0)
            self.T_mu_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_b[-1].bias.data, 0.0)
            self.T_mu_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_1[-1].bias.data, 0.0)

        self.T_logvar_0 = nn.Sequential(OrderedDict([
            ('logvar_sd0', SharedDot(len(self.keep_inds), self.f_n_features, 1)),
            ('logvar_sd0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('logvar_sd0_relu', nn.ReLU(inplace=True)),
            ('logvar_sd1', SharedDot(self.f_n_features, self.f_n_features, 1)),
            ('logvar_sd1_bn', nn.BatchNorm1d(self.f_n_features, affine=False))
        ]))

        self.T_logvar_0_cond_w = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_w0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('logvar_sd1_film_w0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('logvar_sd1_film_w0_swish', Swish()),
            ('logvar_sd1_film_w1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_logvar_0_cond_b = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_b0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('logvar_sd1_film_b0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('logvar_sd1_film_b0_swish', Swish()),
            ('logvar_sd1_film_b1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_logvar_1 = nn.Sequential(OrderedDict([
            ('logvar_sd1_relu', nn.ReLU(inplace=True)),
            ('logvar_sd2', SharedDot(self.f_n_features, len(self.warp_inds), 1, bias=True))
        ]))

        with torch.no_grad():
            self.T_logvar_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_w[-1].bias.data, 0.0)
            self.T_logvar_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_b[-1].bias.data, 0.0)
            self.T_logvar_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_1[-1].bias.data, 0.0)

    def forward(self, p, g, mode='direct'):
        """
        Args:
            p: Input points.
            g: Context.
        """
        logvar = torch.zeros_like(p)
        mu = torch.zeros_like(p)

        logvar[:, self.warp_inds, :] = nn.functional.softsign(self.T_logvar_1(
            torch.add(self.eps, torch.exp(self.T_logvar_0_cond_w(g).unsqueeze(2))) *
            self.T_logvar_0(p[:, self.keep_inds, :].contiguous()) + self.T_logvar_0_cond_b(g).unsqueeze(2)
        ))

        mu[:, self.warp_inds, :] = self.T_mu_1(
            torch.add(self.eps, torch.exp(self.T_mu_0_cond_w(g).unsqueeze(2))) *
            self.T_mu_0(p[:, self.keep_inds, :].contiguous()) + self.T_mu_0_cond_b(g).unsqueeze(2)
        )

        logvar = logvar.contiguous()
        mu = mu.contiguous()

        if mode == 'direct':
            p_out = torch.sqrt(torch.add(self.eps, torch.exp(logvar))) * p + mu
        elif mode == 'inverse':
            p_out = (p - mu) / torch.sqrt(torch.add(self.eps, torch.exp(logvar)))

        return p_out, mu, logvar
