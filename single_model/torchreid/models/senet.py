from __future__ import absolute_import
from __future__ import division

__all__ = [
    'senet154',
    'se_resnet50',
    'se_resnet101',
    'se_resnet152',
    'se_resnext50_32x4d',
    'se_resnext101_32x4d',
    'se_resnext101_32x4d_ibn_a',
    'se_resnet50_fc512'
]

from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torch.nn import functional as F
import torchvision


"""
Code imported from https://github.com/Cadene/pretrained-models.pytorch
"""


pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d_ibn_a': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class SEModule(nn.Module):
    # modified by ZHouKaiYang from `SELayer`
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
        
class SELayer(nn.Module):
    # raw 
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, int(channel/reduction), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(channel/reduction), channel, bias=False),
                nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """ResNeXt bottleneck type C with a Squeeze-and-Excitation module"""
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64.)) * groups)
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


##### diverse attention #######
##### IDE high div #####
class HighDivModule(nn.Module):
    def __init__(self, in_channels, order=1):
        super(HighDivModule, self).__init__()
        self.order = order
        self.inter_channels = in_channels // 8 * 2
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, padding=0, bias=False))
            )
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            setattr(self, name, nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, 1, padding=0, bias=False), nn.Sigmoid()))

    def forward(self, x):
        y=[]
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                layer = getattr(self, name)
                y.append(layer(x))
        y_ = []
        cnt=0
        for j in range(self.order):
            y_temp = 1
            for i in range(j+1):
                y_temp = y_temp * y[cnt]
                cnt += 1
            y_.append(F.relu(y_temp))
        #y_ = F.relu(y_)
        y__ = 0
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            layer = getattr(self, name)
            y__ += layer(y_[i])
        out = x * y__ / self.order
        return out#, y__/ self.order
        

class SENet(nn.Module):
    """Squeeze-and-excitation network.
    
    Reference:
        Hu et al. Squeeze-and-Excitation Networks. CVPR 2018.

    Public keys:
        - ``senet154``: SENet154.
        - ``se_resnet50``: ResNet50 + SE.
        - ``se_resnet101``: ResNet101 + SE.
        - ``se_resnet152``: ResNet152 + SE.
        - ``se_resnext50_32x4d``: ResNeXt50 (groups=32, width=4) + SE.
        - ``se_resnext101_32x4d``: ResNeXt101 (groups=32, width=4) + SE.
        - ``se_resnet50_fc512``: (ResNet50 + SE) + FC.
    """
    
    def __init__(self, num_classes, loss, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1,
                 last_stride=2, fc_dims=None, bnneck=True, bnneck_test='after', **kwargs):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `classifier` layer.
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.loss = loss
        self.bnneck = bnneck
        self.bnneck_test = bnneck_test

        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=last_stride,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.bnneck_layer = self._construct_bnneck_layer(bnneck, bnneck_test, 512 * block.expansion)
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        #self.classifier = nn.Sequential()
        #self.classifier_fine_tune = nn.Linear(self.feature_dim, num_classes)#, bias=False)
        self.classifier = nn.Linear(self.feature_dim, num_classes)#, bias=False)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        self.feature_dim = fc_dims[-1]
        
        return nn.Sequential(*layers)
    def _construct_bnneck_layer(self, bnneck, bnneck_test, dim):
        """Constructs bnneck layer

        Args:
            bnneck (bool): True for add BN and False for not
            bnneck_test (str): options=['after', 'before', 'fc'] means use the feature after/before BN for testing or
                               the fc embedding layer feature
            dim (int): the dim of this layer's inputs
        """

        assert isinstance(bnneck, bool) and bnneck_test in ['before', 'after', 'fc']

        if not bnneck:
            return None
        return nn.Sequential(nn.BatchNorm1d(dim))
        #bottleneck = nn.BatchNorm1d(dim)
        #bottleneck.bias.requires_grad_(False)  # no shift
        #return nn.Sequential(bottleneck)
        

    def featuremaps(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x, return_featuremaps=False):
        f = self.featuremaps(x)
        if return_featuremaps:
            return f
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        v1 = v
        
        if self.bnneck_layer is not None:
            v = self.bnneck_layer(v)
            v2 = v
        
        if self.fc is not None:
            v = self.fc(v)
            v3 = v
        
        if not self.training:
            if self.bnneck_test == 'before':
                return v1
            elif self.bnneck_test == 'after':
                return v2
            elif self.bnneck_test == 'fc':
                return v3
        
        y = self.classifier(v)
        #y = self.classifier_fine_tune(v)
        
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v1
        elif self.loss == 'oim':
            # maybe not correct
            v3 = v1
            return y, v1, v3
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    # pretrain_dict = torch.load('/home/ywchong/.cache/torch/checkpoints/se_resnext101_32x4d-3b2fe3d8.pth')
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def senet154(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(
        num_classes=num_classes,
        loss=loss,
        block=SEBottleneck,
        layers=[3, 8, 36, 3],
        groups=64,
        reduction=16,
        dropout_p=0.2,
        last_stride=2,
        fc_dims=None,
        **kwargs
    )
    if pretrained:
        model_url = pretrained_settings['senet154']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


def se_resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(
        num_classes=num_classes,
        loss=loss,
        block=SEResNetBottleneck,
        layers=[3, 4, 6, 3],
        groups=1,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        last_stride=2,
        fc_dims=None,
        **kwargs
    )
    if pretrained:
        model_url = pretrained_settings['se_resnet50']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


def se_resnet50_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(
        num_classes=num_classes,
        loss=loss,
        block=SEResNetBottleneck,
        layers=[3, 4, 6, 3],
        groups=1,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        last_stride=1,
        fc_dims=[512],
        **kwargs
    )
    if pretrained:
        model_url = pretrained_settings['se_resnet50']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


def se_resnet101(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(
        num_classes=num_classes,
        loss=loss,
        block=SEResNetBottleneck,
        layers=[3, 4, 23, 3],
        groups=1,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        last_stride=2,
        fc_dims=None,
        **kwargs
    )
    if pretrained:
        model_url = pretrained_settings['se_resnet101']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


def se_resnet152(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(
        num_classes=num_classes,
        loss=loss,
        block=SEResNetBottleneck,
        layers=[3, 8, 36, 3],
        groups=1,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        last_stride=2,
        fc_dims=None,
        **kwargs
    )
    if pretrained:
        model_url = pretrained_settings['se_resnet152']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


def se_resnext50_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(
        num_classes=num_classes,
        loss=loss,
        block=SEResNeXtBottleneck,
        layers=[3, 4, 6, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        last_stride=2,
        fc_dims=None,
        **kwargs
    )
    if pretrained:
        model_url = pretrained_settings['se_resnext50_32x4d']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model
                              

class IBN_A(nn.Module):
    def __init__(self, planes):
        super(IBN_A, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out
        
class SEResNeXtBottleneck_IBN_A(Bottleneck):
    """ResNeXt bottleneck type C with a Squeeze-and-Excitation module"""
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4, ibn=True):
        super(SEResNeXtBottleneck_IBN_A, self).__init__()
        width = int(math.floor(planes * (base_width / 64.)) * groups)
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        if ibn:
            self.bn1 = IBN_A(width)
        else:
            self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


def se_resnext101_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(
        num_classes=num_classes,
        loss=loss,
        block=SEResNeXtBottleneck,
        layers=[3, 4, 23, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        last_stride=1,  # default=2
        fc_dims=None,
        **kwargs
    )
    if pretrained:
        model_url = pretrained_settings['se_resnext101_32x4d']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model
    
    
    
def se_resnext101_32x4d_ibn_a(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(
        num_classes=num_classes,
        loss=loss,
        block=SEResNeXtBottleneck_IBN_A,
        layers=[3, 4, 23, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        last_stride=1,  # default=2
        fc_dims=None,  
        **kwargs
    )
    if pretrained:
        model_url = pretrained_settings['se_resnext101_32x4d']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model