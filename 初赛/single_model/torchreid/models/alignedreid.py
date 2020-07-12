from __future__ import absolute_import
from __future__ import division
from .senet import se_resnext101_32x4d
import torch
import torch.nn as nn
import torch.nn.functional as F


class DimReduceLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Aligned_PCB(nn.Module):
    def __init__(self, num_classes, loss='softmax', pretrained=True, **kwargs):
        super(Aligned_PCB, self).__init__()
        self.loss = loss
        self.parts = 8
        self.horizontal = 16
        # base output shape is horizontal * parts
        self.base = se_resnext101_32x4d(num_classes, loss=loss, pretrained=pretrained, **kwargs)

        local_out_channels, planes, reduced_dim, nonlinear = 128, 2048, 256, 'relu'
        self.local_conv = nn.Conv2d(planes, local_out_channels, 1)
        self.local_bn = nn.BatchNorm2d(local_out_channels)
        self.local_relu = nn.ReLU(inplace=True)
        self.local_bottleneck = nn.BatchNorm1d(local_out_channels * self.horizontal)
        self.local_bottleneck.bias.requires_grad_(False)  # no shift
        self.local_classifier = nn.Linear(local_out_channels * self.horizontal, num_classes, bias=False)

        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(planes, reduced_dim, nonlinear=nonlinear)
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.pcb_bottleneck = nn.BatchNorm1d(reduced_dim)
        self.pcb_bottleneck.bias.requires_grad_(False)  # no shift
        self.pcb_classifier = nn.ModuleList([nn.Linear(reduced_dim, num_classes, bias=False) for _ in range(self.parts)])

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremap(self, x):
        x = self.base.layer0(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

    def forward(self, x):
        """
        Returns:
            global_feat: shape [N, C]
            local_feat: shape [N, H, c]
        """
        feat = self.featuremap(x)

        # #################################
        # global feature
        # ---------------------------------
        # print(feat.shape)  [N, 2048, 16, 8]
        global_feat = self.base.global_avgpool(feat)  # [N, C, H, W]
        global_feat = global_feat.view(global_feat.size(0), -1)  # [N, C]
        global_feat_before = global_feat

        if self.base.bnneck_layer is not None:
            global_feat = self.base.bnneck_layer(global_feat)
            global_feat_after_bn = global_feat
        if self.base.fc is not None:
            global_feat = self.base.fc(global_feat)
            global_feat_after_fc = global_feat
        global_class = self.base.classifier(global_feat)

        # #################################
        # PCB feature
        # ---------------------------------
        pcb_part_feat = self.parts_avgpool(feat)
        pcb_part_feat = self.dropout(pcb_part_feat)
        pcb_part_feat_test = pcb_part_feat
        pcb_part_feat = self.conv5(pcb_part_feat)
        pcb_class = []
        pcb_part_triplet = []
        for i in range(self.parts):
            pcb_part_feat_i = pcb_part_feat[:, :, i, :]
            pcb_part_feat_i = pcb_part_feat_i.view(pcb_part_feat_i.size(0), -1)
            pcb_part_triplet.append(pcb_part_feat_i)
            pcb_part_feat_i = self.pcb_bottleneck(pcb_part_feat_i)
            pcb_class_i = self.pcb_classifier[i](pcb_part_feat_i)
            pcb_class.append(pcb_class_i)

        # #################################
        # local feature
        # ---------------------------------
        local_feat = torch.mean(feat, -1, keepdim=True)  # shape [N, C, H, 1]
        local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
        local_feat_before = local_feat
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)  # shape [N, H, c]
        local_feat_class = local_feat_before.view(local_feat_before.size(0), -1)
        local_feat_class = self.local_bottleneck(local_feat_class)
        local_class = self.local_classifier(local_feat_class)

        if not self.training:
            pcb_part_feat_test = F.normalize(pcb_part_feat_test, p=2, dim=1)
            pcb_part_feat_test = pcb_part_feat_test.view(pcb_part_feat_test.size(0), -1)
            return global_feat_after_bn#, pcb_part_feat_test, local_feat

        if self.loss == 'softmax':
            return global_class, pcb_class, local_class
        elif self.loss == 'triplet':
            return global_class, global_feat_before, pcb_class, pcb_part_triplet, local_class, local_feat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))







