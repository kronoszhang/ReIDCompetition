from __future__ import absolute_import

import torch

from .resnet import *
from .senet import *
from .pcb import *
from .alignedreid import Aligned_PCB
from .mgn import  MGN
from .senet_b import senet_MHN


__model_factory = {
    # image classification models
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet50_ibn_old_a': resnet50_ibn_old_a,
    'resnet50_ibn_b': resnet50_ibn_b,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnet101_ibn_old_a': resnet101_ibn_old_a,
    'resnet101_ibn_b': resnet101_ibn_b,
    'resnet152_ibn_a': resnet152_ibn_a,
    'resnet152_ibn_old_a': resnet152_ibn_old_a,
    'resnet152_ibn_b': resnet152_ibn_b,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnext50_ibn_a_32x4d': resnext50_ibn_a_32x4d,
    'resnext101_ibn_a_32x8d': resnext101_ibn_a_32x8d,
    'resnet50_fc512': resnet50_fc512,
    'resnet50_ibn_a_fc512': resnet50_ibn_a_fc512,
    'resnet50_ibn_old_a_fc512': resnet50_ibn_old_a_fc512,
    'resnet50_ibn_b_fc512': resnet50_ibn_b_fc512,
    # they all from class ResNet_IBN_B or ResNet
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'se_resnext101_32x4d_ibn_a': se_resnext101_32x4d_ibn_a,
    # reid special model
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'aligned_pcb': Aligned_PCB,
    'mgn':MGN,
    'senet_mhn': senet_MHN,
}

__model_support_feature_map_return_factory = {
    # this is easy to extend, you can refer to ResNet

    # image classification models
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet50_ibn_old_a': resnet50_ibn_old_a,
    'resnet50_ibn_b': resnet50_ibn_b,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnet101_ibn_old_a': resnet101_ibn_old_a,
    'resnet101_ibn_b': resnet101_ibn_b,
    'resnet152_ibn_a': resnet152_ibn_a,
    'resnet152_ibn_old_a': resnet152_ibn_old_a,
    'resnet152_ibn_b': resnet152_ibn_b,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnext50_ibn_a_32x4d': resnext50_ibn_a_32x4d,
    'resnext101_ibn_a_32x8d': resnext101_ibn_a_32x8d,
    'resnet50_fc512': resnet50_fc512,
    'resnet50_ibn_a_fc512': resnet50_ibn_a_fc512,
    'resnet50_ibn_old_a_fc512': resnet50_ibn_old_a_fc512,
    'resnet50_ibn_b_fc512': resnet50_ibn_b_fc512,
    # they all from class ResNet_IBN_B or ResNet
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'se_resnext101_32x4d_ibn_a': se_resnext101_32x4d_ibn_a,
    'mgn':MGN,
    'senet_mhn': senet_MHN,
}


__model_BNNeck_support_factory = {
    # you can easy to extend but may cost many time

    # image classification models
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet50_ibn_old_a': resnet50_ibn_old_a,
    'resnet50_ibn_b': resnet50_ibn_b,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnet101_ibn_old_a': resnet101_ibn_old_a,
    'resnet101_ibn_b': resnet101_ibn_b,
    'resnet152_ibn_a': resnet152_ibn_a,
    'resnet152_ibn_old_a': resnet152_ibn_old_a,
    'resnet152_ibn_b': resnet152_ibn_b,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnext50_ibn_a_32x4d': resnext50_ibn_a_32x4d,
    'resnext101_ibn_a_32x8d': resnext101_ibn_a_32x8d,
    'resnet50_fc512': resnet50_fc512,
    'resnet50_ibn_a_fc512': resnet50_ibn_a_fc512,
    'resnet50_ibn_old_a_fc512': resnet50_ibn_old_a_fc512,
    'resnet50_ibn_b_fc512': resnet50_ibn_b_fc512,
    # they all from class ResNet_IBN_B or ResNet
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'se_resnext101_32x4d_ibn_a': se_resnext101_32x4d_ibn_a,
    'mgn':MGN,
    'senet_mhn': senet_MHN,
    
}


__model_OIMLoss_support_factory = {
    # you can easy to extend but may cost many time

    # image classification models
    'resnet50_fc512': resnet50_fc512,
    'resnet50_ibn_a_fc512': resnet50_ibn_a_fc512,
    'resnet50_ibn_old_a_fc512': resnet50_ibn_old_a_fc512,
    'resnet50_ibn_b_fc512': resnet50_ibn_b_fc512,
    # they all from class ResNet_IBN_B or ResNet and must with embedding layer
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'se_resnext101_32x4d_ibn_a': se_resnext101_32x4d_ibn_a,
}

def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def show_avai_return_feature_map_models():
    """Displays available models which can return feature maps.

        Examples::
            >>> from torchreid import models
            >>> models.show_avai_return_feature_map_models()
        """
    print(list(__model_support_feature_map_return_factory.keys()))


def show_avai_BNNeck_models():
    """Displays available models which can use BNNeck trick

        Examples::
            >>> from torchreid import models
            >>> models.show_avai_BNNeck_models()
        """
    print(list(__model_BNNeck_support_factory.keys()))


def show_avai_oim_loss_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_oim_loss_models()
    """
    print(list(__model_OIMLoss_support_factory.keys()))


def build_model(name, num_classes, loss='softmax', pretrained=True, use_gpu=True):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
    )