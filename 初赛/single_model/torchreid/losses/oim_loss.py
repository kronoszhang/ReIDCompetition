from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd


class OIM(autograd.Function):
    def __init__(self, lut, momentum=0.5, use_gpu=True):
        super(OIM, self).__init__()
        self.use_gpu = use_gpu
        self.lut = lut
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        if self.use_gpu:
            self.lut = self.lut.cuda()
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = inputs.mm(self.lut.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        if self.use_gpu:
            self.lut = self.lut.cuda()
            grad_outputs = grad_outputs.cuda()
            inputs, targets = inputs.cuda(), targets.cuda()
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)

        for x, y in zip(inputs, targets):
            self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
            self.lut[y] /= self.lut[y].norm()
        return grad_inputs, None


def oim(inputs, targets, lut, momentum=0.5, use_gpu=True):
    return OIM(lut, momentum=momentum, use_gpu=use_gpu)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
                 weight=None, size_average=True, use_gpu=True):
        super(OIMLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average

        self.register_buffer('lut', torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets):
        inputs = oim(inputs, targets, self.lut, momentum=self.momentum, use_gpu=self.use_gpu)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight,
                               size_average=self.size_average)
        return loss, inputs

