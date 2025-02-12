import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import init

from pina.model.layers import SpectralConvBlock1D, SpectralConvBlock2D


class _BaseBARNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tol = 1e-18 # numerial stability, see https://github.com/pyg-team/pytorch_geometric/issues/2559
    def kl(self, alpha):
        scale = self.weight.shape.numel() + (self.bias.shape.numel() if self.bias is not None else 0)
        mean_alpha = alpha
        var_alpha = alpha.pow(2)
        mean_beta = mean_alpha.mean(0, keepdim=True)
        var_beta = var_alpha.mean(0, keepdim=True)
        kl = scale * 0.5 * ((mean_alpha-mean_beta).pow(2)/var_beta + (var_alpha/var_beta) - 1 - torch.log(var_alpha/var_beta))
        return kl

class LinearBARNN(torch.nn.Linear, _BaseBARNN):
    def forward(self, input, alpha):
        mean = alpha * (F.linear(input, self.weight) + self.bias)
        var = alpha.pow(2) * (F.linear(input.pow(2), self.weight.pow(2)) + self.bias.pow(2))
        return mean + torch.sqrt(var+self.tol)*torch.randn_like(mean)

class Conv1dBARNN(nn.Conv1d, _BaseBARNN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super(Conv1dBARNN, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, False)
    def forward(self, input, alpha):
        W = self.weight
        mean = alpha * F.conv1d(input, W, self.bias, self.stride, self.padding, self.dilation, self.groups)
        var = alpha.pow(2) * F.conv1d(input.pow(2), W.pow(2), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return mean + torch.sqrt(var+self.tol)*torch.randn_like(mean)
    
class Conv2dBARNN(nn.Conv2d, _BaseBARNN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super(Conv2dBARNN, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, False)
    def forward(self, input, alpha):
        """
        Forward with all regularized connections and random activations (Bayesian mode). Typically used for train
        """
        W = self.weight
        mean = alpha * F.conv2d(input, W, self.bias, self.stride, self.padding, self.dilation, self.groups)
        var = alpha.pow(2) * F.conv2d(input.pow(2), W.pow(2), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return mean + torch.sqrt(var+self.tol)*torch.randn_like(mean)

    
class ARDVariationalDropoutLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.tol = 1e-18 # numerial stability, see https://github.com/pyg-team/pytorch_geometric/issues/2559
        self.in_features = in_features
        self.out_features = out_features
        self.log_alpha = torch.nn.Parameter(torch.randn(1, ))
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        # Get dropout rate
        alpha = self.log_alpha.exp()
        # Compute the mean/std with local reparametrization trick
        mean =  F.linear(input, self.weight) + self.bias
        var =   alpha * (F.linear(input**2, self.weight**2) + self.bias.pow(2))
        # Global reparametrization trick
        return mean + torch.sqrt(var+self.tol)*torch.randn_like(mean)

    def kl(self):
        drop_rate = self.log_alpha.exp()
        kl = 0.5 * torch.log(1. + drop_rate.pow(-1))
        return kl.sum()

class ARDVariationalDropoutConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super(ARDVariationalDropoutConv1d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, False)
        self.bias = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.log_alpha = torch.nn.Parameter(torch.randn(1, ))
        self.tol = 1e-18 # numerial stability, see https://github.com/pyg-team/pytorch_geometric/issues/2559

    def forward(self, input):
        # Get dropout rate
        alpha = self.log_alpha.exp()
        # Local reparam. trick
        W = self.weight
        mean = F.conv1d(input, W, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)
        var = alpha * F.conv1d(input.pow(2), W.pow(2), self.bias,
                                      self.stride,self.padding, self.dilation,
                                      self.groups)
        return mean + torch.sqrt(var+self.tol)*torch.randn_like(mean)

    def kl(self):
        drop_rate = self.log_alpha.exp()
        kl = 0.5 * torch.log(1. + drop_rate.pow(-1))
        return kl.sum()

class ARDVariationalDropoutConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super(ARDVariationalDropoutConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, False)
        self.bias = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.log_alpha = torch.nn.Parameter(torch.randn(1, ))
        self.tol = 1e-18 # numerial stability, see https://github.com/pyg-team/pytorch_geometric/issues/2559

    def forward(self, input):
        # Get dropout rate
        alpha = self.log_alpha.exp()
        # Local reparam. trick
        W = self.weight
        mean = F.conv1d(input, W, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)
        var = alpha * F.conv1d(input.pow(2), W.pow(2), self.bias,
                                      self.stride,self.padding, self.dilation,
                                      self.groups)
        return mean + torch.sqrt(var+self.tol)*torch.randn_like(mean)

    def kl(self):
        drop_rate = self.log_alpha.exp()
        kl = 0.5 * torch.log(1. + drop_rate.pow(-1))
        return kl.sum()
    
class ConditionalSpectralConvBlock1D(SpectralConvBlock1D):
    def forward(self, x, emb):
        return super().forward(x * emb)
    
class ConditionalSpectralConvBlock2D(SpectralConvBlock2D):
    def forward(self, x, emb):
        return super().forward(x * emb)