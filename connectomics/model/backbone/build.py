from __future__ import print_function, division
from typing import Optional, List

import torch
import torch.nn as nn
from .resnet import ResNet3D
from .repvgg import RepVGG3D
from .botnet import BotNet3D
from .efficientnet import EfficientNet3D
from .swintr import SwinTransformer2D,SwinTransformer3D
from ..utils.misc import IntermediateLayerGetter

backbone_dict = {
    'resnet': ResNet3D,
    'repvgg': RepVGG3D,
    'botnet': BotNet3D,
    'efficientnet': EfficientNet3D,
    'swintransformer2d': SwinTransformer2D,
    'swintransformer3d': SwinTransformer3D,
}


def build_backbone(backbone_type: str,
                   feat_keys: List[str],
                   **kwargs):
    assert backbone_type in ['resnet', 'repvgg', 'botnet', 'efficientnet','swintransformer2d','swintransformer3d']
    return_layers = {'layer0': feat_keys[0],
                     'layer1': feat_keys[1],
                     'layer2': feat_keys[2],
                     'layer3': feat_keys[3],
                     'layer4': feat_keys[4]}

    backbone = backbone_dict[backbone_type](**kwargs)
    if backbone_type[:15] =='swintransformer':
        if backbone.use_conv:
            assert len(feat_keys) == backbone.num_layers + 2
        else:
            assert len(feat_keys) == backbone.num_layers + 1
    else:
        assert len(feat_keys) == backbone.num_stages
    return IntermediateLayerGetter(backbone, return_layers)
