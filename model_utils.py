import collections
import functools

import torch
import torch.nn as nn
from torch.nn.functional import conv2d


def gaussian_sampler(mean, log_var):
    x = torch.normal(torch.zeros(mean.size()), torch.ones(mean.size())).to(mean.device)
    return mean + x * torch.exp(log_var / 2.)


def conv_cross2d(inputs, weights, bias = None, stride = 1, padding = 0, dilation = 1, groups = 1):
    outputs = []
    for input, weight in zip(inputs, weights):
        output = conv2d(
            input = input.unsqueeze(0),
            weight = weight,
            bias = bias,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups
        )
        outputs.append(output)
    outputs = torch.cat(outputs, dim = 0)
    return outputs


class Conv2dLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
                 normalization='batch', nonlinear='relu'):
        if padding is None:
            padding = (kernel_size - 1) // 2

        bias = (normalization is None or normalization is False)

        modules = [nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )]

        if normalization is not None and normalization is not False:
            if normalization == 'batch':
                modules.append(nn.BatchNorm2d(num_features=out_channels))
            else:
                raise NotImplementedError(
                    'unsupported normalization layer: {0}'.format(normalization))

        if nonlinear is not None and nonlinear is not False:
            if nonlinear == 'relu':
                modules.append(nn.ReLU(inplace=True))
            elif nonlinear == 'leakyrelu':
                modules.append(nn.LeakyReLU(inplace=True))
            else:
                raise NotImplementedError(
                    'unsupported nonlinear activation: {0}'.format(nonlinear))

        super(Conv2dLayer, self).__init__(*modules)


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, normalization='batch', nonlinear='relu'):
        bias = (normalization is None or normalization is False)
        
        modules = [nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )]

        if normalization is not None and normalization is not False:
            if normalization == 'batch':
                modules.append(nn.BatchNorm1d(num_features=out_features))
            else:
                raise NotImplementedError(
                    'unsupported normalization layer: {0}'.format(normalization))

        if nonlinear is not None and nonlinear is not False:
            if nonlinear == 'relu':
                modules.append(nn.ReLU(inplace=True))
            elif nonlinear == 'leakyrelu':
                modules.append(nn.LeakyReLU(inplace=True))
            else:
                raise NotImplementedError(
                    'unsupported nonlinear activation: {0}'.format(nonlinear))

        super(LinearLayer, self).__init__(*modules)


class UBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc = None, submodule = None, outermost = False, innermost = False):
        super(UBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size = 4, stride = 2, padding = 1, bias = False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size = 4, stride = 2, padding = 1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size = 4, stride = 2, padding = 1, bias = False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size = 4, stride = 2, padding = 1, bias = False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, ngf = 64):
        super(UNet, self).__init__()

        unet_block = UBlock(ngf * 8, ngf * 8, input_nc = None, submodule = None, innermost = True)

        for i in range(num_layers - 5):
            unet_block = UBlock(ngf * 8, ngf * 8, input_nc = None, submodule = unet_block)

        unet_block = UBlock(ngf * 4, ngf * 8, input_nc = None, submodule = unet_block)
        unet_block = UBlock(ngf * 2, ngf * 4, input_nc = None, submodule = unet_block)
        unet_block = UBlock(ngf, ngf * 2, input_nc = None, submodule = unet_block)
        unet_block = UBlock(out_channels, ngf, input_nc = in_channels, submodule = unet_block, outermost = True)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if m.weight is not None:
            nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, val = 0)

    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if m.weight is not None:
            nn.init.normal_(m.weight, mean = 1, std = 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, val = 0)


def init_weights(model):
    model.apply(functools.partial(_init_weights))