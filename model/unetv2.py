import os,sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

def conv3d_pad(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
               dilation=(1,1,1), padding=(1,1,1), bias=False):
    # the size of the padding should be a 6-tuple        
    padding = tuple([x for x in padding for _ in range(2)][::-1])
    return  nn.Sequential(
                nn.ReplicationPad3d(padding),
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=0, dilation=dilation, bias=bias))     

def conv3d_bn_non(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, 
                       dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes))              

def conv3d_bn_elu(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, 
                       dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes),
            nn.ELU(inplace=True))                                   


class dilated_fusion_block(nn.Module):
    # Basic residual module of unet
    def __init__(self, in_planes, out_planes, channel_reduction=16, spatial_reduction=2, z_reduction=1):
        super(dilated_fusion_block, self).__init__() 
        self.se_layer = SELayer(channel=out_planes, channel_reduction=channel_reduction, 
                                spatial_reduction=spatial_reduction, z_reduction=z_reduction)

        self.inconv = conv3d_bn_elu(in_planes,  out_planes, kernel_size=(3,3,3), 
                                    stride=1, padding=(1,1,1), bias=True)

        self.block1 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,1,1), padding=(0,1,1), bias=False)
        self.block2 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,2,2), padding=(0,2,2), bias=False)
        self.block3 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,4,4), padding=(0,4,4), bias=False)
        self.block4 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,8,8), padding=(0,8,8), bias=False)                                                                                     

    def forward(self, x):
        residual  = self.inconv(x)

        x1 = self.block1(residual)
        x2 = self.block2(F.elu(x1, inplace=True))
        x3 = self.block3(F.elu(x2, inplace=True))
        x4 = self.block4(F.elu(x3, inplace=True))

        out = residual + x1 + x2 + x3 + x4
        out = self.se_layer(out)
        out = F.elu(out, inplace=True)
        return out 


class residual_block_2d(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True):
        super(residual_block_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_bn_elu( in_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1)),
            conv3d_bn_non(out_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1))
        )
        self.projector = conv3d_bn_non(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y  

class residual_block_3d(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True):
        super(residual_block_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_bn_elu( in_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1)),
            conv3d_bn_non(out_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1))
        )
        self.projector = conv3d_bn_non(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y       

class bottleneck_dilated(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True):
        super(bottleneck_dilated, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_bn_elu( in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0)),
            conv3d_bn_elu(out_planes, out_planes, kernel_size=(3,3,3), dilation=(1,2,2), padding=(1,2,2)),
            conv3d_bn_non(out_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        )
        self.projector = conv3d_bn_non(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0))
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y        

class unetv2(nn.Module):
    def __init__(self, in_channel=1, out_channel=3, filters=[32,64,128,256,256]):
        super().__init__()

        # encoding path
        self.layer1_E = dilated_fusion_block(in_channel, filters[0], channel_reduction=2, spatial_reduction=16)
        self.layer2_E = dilated_fusion_block(filters[0], filters[1], channel_reduction=4, spatial_reduction=8)
        self.layer3_E = nn.Sequential(
            residual_block_3d(filters[1], filters[2], projection=True),
            SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )
        self.layer4_E = nn.Sequential(
            bottleneck_dilated(filters[2], filters[3], projection=True),
            SELayer(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # center block
        self.center = nn.Sequential(
            bottleneck_dilated(filters[3], filters[4], projection=True),
            SELayer(channel=filters[4], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # decoding path
        self.layer1_D = dilated_fusion_block(filters[0], filters[0], channel_reduction=2, spatial_reduction=16)
        self.layer2_D = dilated_fusion_block(filters[1], filters[1], channel_reduction=4, spatial_reduction=8)
        self.layer3_D = nn.Sequential(
            residual_block_3d(filters[2], filters[2], projection=True),
            SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )
        self.layer4_D = nn.Sequential(
            bottleneck_dilated(filters[3], filters[3], projection=True),
            SELayer(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # pooling & upsample
        self.down = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)

        # conv + upsample
        self.conv1 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv2 = conv3d_bn_elu(filters[2], filters[1], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_elu(filters[3], filters[2], kernel_size=(1,1,1), padding=(0,0,0)) 
        self.conv4 = conv3d_bn_elu(filters[4], filters[3], kernel_size=(1,1,1), padding=(0,0,0))

        # convert to probability
        self.fconv = conv3d_bn_non(filters[0], out_channel, kernel_size=(3,3,3), padding=(1,1,1))

    def forward(self, x):

        # encoding path
        z1 = self.layer1_E(x)
        x = self.down(z1)
        z2 = self.layer2_E(x)
        x = self.down(z2)
        z3 = self.layer3_E(x)
        x = self.down(z3)
        z4 = self.layer4_E(x)
        x = self.down(z4)

        x = self.center(x)

        # decoding path
        x = self.up(self.conv4(x))
        x = x + z4
        x = self.layer4_D(x)

        x = self.up(self.conv3(x))
        x = x + z3
        x = self.layer3_D(x)

        x = self.up(self.conv2(x))
        x = x + z2
        x = self.layer2_D(x)

        x = self.up(self.conv1(x))
        x = x + z1
        x = self.layer1_D(x)

        x = self.fconv(x)
        x = torch.sigmoid(x)
        return x

class SELayer(nn.Module):
    # Squeeze-and-excitation layer
    def __init__(self, channel, channel_reduction=4, spatial_reduction=4, z_reduction=1):
        super(SELayer, self).__init__()
        self.pool_size = (z_reduction, spatial_reduction, spatial_reduction)
        self.se = nn.Sequential(
                nn.AvgPool3d(kernel_size=self.pool_size, stride=self.pool_size),
                nn.Conv3d(channel, channel // channel_reduction, kernel_size=1),
                SynchronizedBatchNorm3d(channel // channel_reduction),
                nn.ELU(inplace=True),
                nn.Conv3d(channel // channel_reduction, channel, kernel_size=1),
                SynchronizedBatchNorm3d(channel),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=self.pool_size, mode='trilinear', align_corners=False),
                )     

    def forward(self, x):
        y = self.se(x)
        z = x + y*x
        return z 

def test():
    model = unetv2()
    print('model type: ', model.__class__.__name__)
    num_params = sum([p.data.nelement() for p in model.parameters()])
    print('number of trainable parameters: ', num_params)
    x = torch.randn(8, 1, 4, 64, 64)
    y = model(x)
    print(x.size(), y.size())

if __name__ == '__main__':
    test()