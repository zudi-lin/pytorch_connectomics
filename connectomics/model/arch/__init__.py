from .unet import UNet3D, UNet2D, UNetPlus3D
from .fpn import FPN3D
from .deeplab import DeepLabV3
from .swintr import SwinTransformer3D,SwinTransformer2D

__all__ = [
    'UNet3D',
    'UNetPlus3D',
    'UNet2D',
    'FPN3D',
    'DeepLabV3',
    'SwinTransformer3D',
    'SwinTransformer2D'
]
