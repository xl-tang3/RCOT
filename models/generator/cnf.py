import argparse
import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import Parameter
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal

__all__ = ["CNF", "SequentialFlow", "MovingBatchNorm1d", "ODEnet", "ODEfunc"]


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, context, logpx=None, reverse=False, inds=None, integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, context, logpx, integration_times, reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, context, logpx, integration_times, reverse)
            return x, logpx


class CNF(nn.Module):
    def __init__(self, odefunc, conditional=True, T=1.0, train_T=False, regularization_fns=None,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super(CNF, self).__init__()
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))

        if regularization_fns is not None and len(regularization_fns) > 0:
            raise NotImplementedError("Regularization not supported")
        self.use_adjoint = use_adjoint
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.conditional = conditional

    def forward(self, x, context=None, logpx=None, integration_times=None, reverse=False):
        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x)
        else:
            _logpx = logpx

        if self.conditional:
            assert context is not None
            states = (x, _logpx, context)
            atol = [self.atol] * 3
            rtol = [self.rtol] * 3
        else:
            states = (x, _logpx)
            atol = [self.atol] * 2
            rtol = [self.rtol] * 2

        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack(
                    [torch.tensor(0.0).to(x), self.sqrt_end_time * self.sqrt_end_time]
                ).to(x)
            else:
                integration_times = torch.tensor([0., self.T], requires_grad=False).to(x)

        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        if self.training:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=atol,
                rtol=rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )
        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]

        if logpx is not None:
            return z_t, logpz_t
        else:
            return z_t

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


###################
## Normalization ##
###################

def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt


class MovingBatchNormNd(nn.Module):
    def __init__(self, num_features, eps=1e-4, decay=0.1, bn_lag=0., affine=True, sync=False):
        super(MovingBatchNormNd, self).__init__()
        self.num_features = num_features
        self.sync = sync
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer('step', torch.zeros(1))
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()

    def forward(self, x, c=None, logpx=None, reverse=False):
        if reverse:
            return self._reverse(x, logpx)
        else:
            return self._forward(x, logpx)

    def _forward(self, x, logpx=None):
        num_channels = x.size(-1)
        used_mean = self.running_mean.clone().detach()
        used_var = self.running_var.clone().detach()

        if self.training:
            # compute batch statistics
            x_t = x.transpose(0, 1).reshape(num_channels, -1)
            batch_mean = torch.mean(x_t, dim=1)

            if self.sync:
                batch_ex2 = torch.mean(x_t**2, dim=1)
                batch_mean = reduce_tensor(batch_mean)
                batch_ex2 = reduce_tensor(batch_ex2)
                batch_var = batch_ex2 - batch_mean**2
            else:
                batch_var = torch.var(x_t, dim=1)

            # moving average
            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= (1. - self.bn_lag**(self.step[0] + 1))
                used_var = batch_var - (1 - self.bn_lag) * (batch_var - used_var.detach())
                used_var /= (1. - self.bn_lag**(self.step[0] + 1))

            # update running estimates
            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.running_var -= self.decay * (self.running_var - batch_var.data)
            self.step += 1

        # perform normalization
        used_mean = used_mean.view(*self.shape).expand_as(x)
        used_var = used_var.view(*self.shape).expand_as(x)

        y = (x - used_mean) * torch.exp(-0.5 * torch.log(used_var + self.eps))

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(x)
            bias = self.bias.view(*self.shape).expand_as(x)
            y = y * torch.exp(weight) + bias

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x, used_var).sum(-1, keepdim=True)

    def _reverse(self, y, logpy=None):
        used_mean = self.running_mean
        used_var = self.running_var

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(y)
            bias = self.bias.view(*self.shape).expand_as(y)
            y = (y - bias) * torch.exp(-weight)

        used_mean = used_mean.view(*self.shape).expand_as(y)
        used_var = used_var.view(*self.shape).expand_as(y)
        x = y * torch.exp(0.5 * torch.log(used_var + self.eps)) + used_mean

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x, used_var).sum(-1, keepdim=True)

    def _logdetgrad(self, x, used_var):
        logdetgrad = -0.5 * torch.log(used_var + self.eps)
        if self.affine:
            weight = self.weight.view(*self.shape).expand(*x.size())
            logdetgrad += weight
        return logdetgrad

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag},'
            ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)
        )


def stable_var(x, mean=None, dim=1):
    if mean is None:
        mean = x.mean(dim, keepdim=True)
    mean = mean.view(-1, 1)
    res = torch.pow(x - mean, 2)
    max_sqr = torch.max(res, dim, keepdim=True)[0]
    var = torch.mean(res / max_sqr, 1, keepdim=True) * max_sqr
    var = var.view(-1)
    # change nan to zero
    var[var != var] = 0
    return var


class MovingBatchNorm1d(MovingBatchNormNd):
    @property
    def shape(self):
        return [1, -1]

    def forward(self, x, context=None, logpx=None, integration_times=None, reverse=False):
        ret = super(MovingBatchNorm1d, self).forward(
                x, context, logpx=logpx, reverse=reverse)
        return ret


#############
## ODEfunc ##
#############

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, context, x):
        return self._layer(x)


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1 + dim_c, dim_out)

    def forward(self, context, x, c):
        if x.dim() == 3:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)
        x_context = torch.cat((x, context), dim=2)
        return self._layer(x_context)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatLinear_v2, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)

    def forward(self, context, x):
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            bias = bias.unsqueeze(1)
        return self._layer(x) + bias


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper(context))
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
        return self._layer(x) * gate


class ScaleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(ScaleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = self._hyper(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
        return self._layer(x) * gate


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        nn.init.xavier_uniform_(self._layer.weight)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)
        nn.init.xavier_uniform_(self._hyper_bias.weight)
        self._hyper_gate = nn.Linear(1 + dim_c, dim_out)
        nn.init.xavier_uniform_(self._hyper_gate.weight)

    def forward(self, context, x): ## N* 60+1
        gate = torch.sigmoid(self._hyper_gate(context)) ##(196608, 64)
        bias = self._hyper_bias(context) ##(196608, 64)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret ## N*64


class ConcatScaleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatScaleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = self._hyper_gate(context)
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx.mul(e)

    cnt = 0
    while not e_dzdx_e.requires_grad and cnt < 10:
        # print("RequiresGrad:f=%s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt=%d"
        #       % (f.requires_grad, y.requires_grad, e_dzdx.requires_grad,
        #          e.requires_grad, e_dzdx_e.requires_grad, cnt))
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        cnt += 1

    approx_tr_dzdx = e_dzdx_e.sum(dim=-1)
    assert approx_tr_dzdx.requires_grad, \
        "(failed to add node to graph) f=%s %s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt:%s" \
        % (
        f.size(), f.requires_grad, y.requires_grad, e_dzdx.requires_grad, e.requires_grad, e_dzdx_e.requires_grad, cnt)
    return approx_tr_dzdx


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
}


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(self, hidden_dims, input_shape, context_dim, layer_type="concat", nonlinearity="softplus"):
        super(ODEnet, self).__init__()
        base_layer = {
            "ignore": IgnoreLinear,
            "squash": SquashLinear,
            "scale": ScaleLinear,
            "concat": ConcatLinear,
            "concat_v2": ConcatLinear_v2,
            "concatsquash": ConcatSquashLinear,
            "concatscale": ConcatScaleLinear,
        }[layer_type]

        # build models and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        for dim_out in (hidden_dims + [input_shape[0],]):
            layer_kwargs = {}
            layer = base_layer(hidden_shape[0], dim_out, context_dim, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, context, y):
        dx = y ##(196608,3)
        for l, layer in enumerate(self.layers):
            dx = layer(context, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states): ## states: [x, _logpx, feature]
        y = states[0] ##(196608,3)
        t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)

        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.randn_like(y, requires_grad=True).to(y)

        with torch.set_grad_enabled(True):
            if len(states) == 3:  # conditional CNF
                c = states[2] ## feature
                tc = torch.cat([t, c.view(y.size(0), -1)], dim=1) ##(196608,61)
                dy = self.diffeq(tc, y)
                divergence = self.divergence_fn(dy, y, e=self._e).unsqueeze(-1)
                return dy, -divergence, torch.zeros_like(c).requires_grad_(True)
            elif len(states) == 2:  # unconditional CNF
                dy = self.diffeq(t, y)
                divergence = self.divergence_fn(dy, y, e=self._e).view(-1, 1)
                return dy, -divergence
            else:
                assert 0, "`len(states)` should be 2 or 3"


def build_flow(args, input_dim=3, hidden_dims=('256,256'), context_dim=512, num_blocks=1, conditional=True):
    def build_cnf():
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(input_dim,),
            context_dim=context_dim,
            layer_type=args.layer_type,
            nonlinearity='softplus',
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            conditional=conditional,
            solver=args.solver,
            use_adjoint=args.use_adjoint,
            atol=args.atol,
            rtol=args.rtol,
        )
        return cnf


    chain = [build_cnf() for _ in range(num_blocks)]
    if args.batch_norm:
        bn_layers = [MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag, sync=args.sync_bn)
                     for _ in range(num_blocks)]
        bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag, sync=args.sync_bn)]

        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = SequentialFlow(chain)
    return model
