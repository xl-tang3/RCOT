"""
Affine Coupling Layers
"""
import types
import torch
from torch import nn
from torch.nn import functional as F


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


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


class Cond3DACL(nn.Module):
    """
        Conditional 3D affine coupling layer
    """

    def __init__(self, warp_dims, hidden_dim, context_dim):
        super().__init__()
        self.warp_dims = warp_dims
        self.keep_dims = [i for i in range(3) if i not in warp_dims]

        ## Network for scale and translation
        self.fc1 = ConcatSquashLinear(len(self.keep_dims), hidden_dim, context_dim)
        self.fc2 = ConcatSquashLinear(hidden_dim, hidden_dim, context_dim)
        self.fc3 = ConcatSquashLinear(hidden_dim, len(self.warp_dims)*2, context_dim)
        
    def net_s_t(self, x, ctx):
        x = F.relu(self.fc1(ctx, x))
        x = F.relu(self.fc2(ctx, x))
        x = self.fc3(ctx, x)
        return x

    def forward(self, x, ctx, logpx=None, reverse=False):
        """
        Args:
            ctx: Context vectors, must be reshaped into a 2D-tensor, (BNk, F).
            x:  Points, must be reshaped into a 2D-tensor, (BNk, 3)
        """
        assert x.dim() == 2
        # print(x.size(), ctx.size())
        s_t = self.net_s_t(x[:, self.keep_dims], ctx)
        scale = torch.sigmoid(s_t[:, :len(self.warp_dims)] + 2.0)
        shift = s_t[:, len(self.warp_dims):]

        logdetjac = torch.sum(torch.log(scale).view(scale.size(0), -1), dim=1, keepdim=True)

        if not reverse:
            y_warp = x[:, self.warp_dims] * scale + shift
            delta_logp = -logdetjac
        else:
            y_warp = (x[:, self.warp_dims] - shift) / scale
            delta_logp = logdetjac

        ys = [None, None, None]
        for i, d in enumerate(self.warp_dims):
            ys[d] = y_warp[:, i:i+1]
        for i, d in enumerate(self.keep_dims):
            ys[d] = x[:, d:d+1]
        y = torch.cat(ys, dim=1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp


def build_flow(context_dim, hidden_dim=64):
    flow = SequentialFlow([
        Cond3DACL([0], hidden_dim, context_dim),
        Cond3DACL([1,2], hidden_dim, context_dim),
        Cond3DACL([1], hidden_dim, context_dim),
        Cond3DACL([0,2], hidden_dim, context_dim),
        Cond3DACL([2], hidden_dim, context_dim),
        Cond3DACL([0,1], hidden_dim, context_dim),
        Cond3DACL([0], hidden_dim, context_dim),
        Cond3DACL([1,2], hidden_dim, context_dim),
        Cond3DACL([1], hidden_dim, context_dim),
        Cond3DACL([0,2], hidden_dim, context_dim),
        Cond3DACL([2], hidden_dim, context_dim),
        Cond3DACL([0,1], hidden_dim, context_dim),
    ])
    return flow
