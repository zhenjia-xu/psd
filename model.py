import collections
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from model_utils import Conv2dLayer, LinearLayer, UNet, init_weights, conv_cross2d, gaussian_sampler

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
          ('conv0', Conv2dLayer(3, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv1', Conv2dLayer(32, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv2', Conv2dLayer(32, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv3', Conv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv4', Conv2dLayer(64, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv5', Conv2dLayer(32, 32, 5, stride=1, normalization=None, nonlinear=None)),
        ]))

    def forward(self, inputs):
        return self.encoder.forward(inputs)


class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
          ('conv0', Conv2dLayer(2, 16, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv1', Conv2dLayer(16, 16, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv2', Conv2dLayer(16, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv3', Conv2dLayer(32, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv4', Conv2dLayer(32, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv5', Conv2dLayer(64, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv6', Conv2dLayer(64, 64, 5, stride=2, normalization=None, nonlinear=None)),
        ]))

    def forward(self, inputs):
        outputs = self.encoder.forward(inputs)
        outputs = outputs.view(inputs.size(0), -1)
        return torch.split(outputs, outputs.size(1) // 2, dim = 1)


class KernelDecoder(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(KernelDecoder, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.decoders = nn.ModuleList()
        for k in range(num_channels):
            self.decoders.append(
                nn.Sequential(OrderedDict([
                    ('fc0', LinearLayer(1, 64, normalization='batch', nonlinear='relu')),
                    ('fc1', LinearLayer(64, 128, normalization='batch', nonlinear='relu')),
                    ('fc2', LinearLayer(128, 64, normalization='batch', nonlinear='relu')),
                    ('fc3', LinearLayer(64, kernel_size * kernel_size, normalization=None, nonlinear=None)),
                ]))
            )

    def forward(self, inputs):
        inputs = inputs.view(-1, self.num_channels, 1)

        outputs = []
        for i in range(self.num_channels):
            output = self.decoders[i].forward(inputs[:, i])
            output = output.view(-1, self.kernel_size, self.kernel_size)
            outputs.append(output)

        outputs = torch.stack(outputs, dim = 1)
        outputs = outputs.view(-1, self.num_channels, 1, self.kernel_size, self.kernel_size)
        return outputs


class MotionDecoder(nn.Module):
    def __init__(self):
        super(MotionDecoder, self).__init__()
        self.decoders = nn.ModuleList()
        for k in range(2):
            self.decoders.append(
                nn.Sequential(OrderedDict([
                    ('conv0', Conv2dLayer(32, 32, 9, stride=1, normalization='batch', nonlinear='leakyrelu', groups=32)),
                    ('conv1', Conv2dLayer(32, 32, 9, stride=1, normalization='batch', nonlinear='leakyrelu', groups=32)),
                    ('conv2', Conv2dLayer(32, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu', groups=32)),
                    ('conv3', Conv2dLayer(32, 32, 5, stride=1, normalization=None, nonlinear=None, groups=32)),
                ]))
            )
    
    def forward(self, inputs):
        motions = torch.stack([decoder.forward(inputs) for decoder in self.decoders], 1)
        return motions


class StructuralDescriptor(nn.Module):
    def __init__(self, num_dimension):
        super(StructuralDescriptor, self).__init__()
        init_structure = -10 * torch.ones(num_dimension, num_dimension)
        
        self.structure = nn.Parameter(init_structure)
    def forward(self, motions_before, structure_mask):
        motions_norm = torch.sqrt(motions_before[:, 0] ** 2 + motions_before[:, 1] ** 2)

        B, C, W, H = motions_norm.size()
        motions_flatten = motions_norm.view(B, C, -1)
        motion_max, _ = torch.max(motions_flatten, 2)
        motion_max = motion_max.view(B, C, 1, 1)

        eps = 0.0001
        masks = motions_norm / (motion_max.detach() + eps)
        
        mask_transpose = torch.transpose(masks, 0, 1).reshape(C, -1)

        extra_flows = []
        for k in range(2):
            flow_flatten = motions_before[:, k, ...].view(B, C, -1)
            _, indices = torch.max(torch.abs(flow_flatten), 2)
            flow_max = torch.gather(flow_flatten, 2, indices.view(B, C, 1)).view(B, C, 1, 1).detach()

            extra_mask_flatten = torch.matmul(structure_mask * torch.sigmoid(self.structure), mask_transpose)
            extra_mask = torch.transpose(extra_mask_flatten.view(C, B, W, H), 0, 1)
            extra_flows.append(extra_mask * flow_max)
        
        extra_motions = torch.stack(extra_flows, 1)
        motions_after = motions_before + extra_motions

        return motions_after, masks

        

class ImageDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(ImageDecoder, self).__init__()

        self.decoder = UNet(
            in_channels = in_channels,
            out_channels = out_channels,
            num_layers = num_layers
        )

    def forward(self, images, flows):
        inputs = torch.cat((images, flows), dim = 1)
        outputs = images + self.decoder.forward(inputs)
        return outputs


class PSD(nn.Module):
    def __init__(self, dimensions, size=128):
        super(PSD, self).__init__()
        self.dimensions = dimensions
        self.size = size

        self.image_encoder = ImageEncoder()

        self.motion_encoder = MotionEncoder()

        self.kernel_decoder = KernelDecoder(
            num_channels = 32,
            kernel_size = 5,
        )
        self.motion_decoder = MotionDecoder()
        
        self.image_decoder = ImageDecoder(
            in_channels = 5,
            out_channels = 3,
            num_layers = 7
        )

        if dimensions is not None:
            self.structural_descriptor = StructuralDescriptor(num_dimension=32)
        
        init_weights(self)

    def forward(self, image_inputs, flow_inputs=None, mean = None, log_var = None, z = None, returns = None):
        if mean is None and log_var is None and z is None:
            mean, log_var = self.motion_encoder.forward(flow_inputs)

        if self.dimensions is not None:
            mean, log_var = mean.detach(), log_var.detach()

        if z is None:
            z = gaussian_sampler(mean, log_var)
        features = self.image_encoder.forward(image_inputs)
        kernels = self.kernel_decoder.forward(z)

        features = conv_cross2d(
            inputs = features,
            weights = kernels,
            padding = (kernels.size(-1) - 1) // 2,
            groups = features.size(1)
        )

        if features.size(-1) != self.size:
            features = interpolate(features, size = self.size, mode = 'nearest')

        motion_outputs = self.motion_decoder.forward(features)

        if self.dimensions is not None:
            structure_mask = torch.zeros(32, 32)
            for x in self.dimensions:
                for y in self.dimensions:
                    structure_mask[x][y] = 1 if x != y else 0
            motions_after, masks = self.structural_descriptor(motion_outputs, structure_mask.to(image_inputs.device))
            flow_outputs = torch.sum(motions_after, 2)
        else:
            flow_outputs = torch.sum(motion_outputs, 2)

        image_outputs = self.image_decoder.forward(image_inputs, flow_outputs.detach())

        outputs = {
            'image_outputs': image_outputs,
            'flow_outputs': flow_outputs,
            'motion_outputs': motion_outputs,
        }

        if self.dimensions is not None:
            outputs['masks'] = masks

        if returns is not None:
            for v in returns:
                outputs[v] = locals()[v]

        return outputs