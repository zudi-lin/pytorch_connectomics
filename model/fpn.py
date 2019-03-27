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

class fpn(nn.Module):
    def __init__(self, in_channel=1, out_channel=3, filters=[32,64,128,256,256]):
        super().__init__()

        # layers
        self.layer1 = nn.Sequential(
            residual_block_2d(in_channel, filters[0], projection=True),
            residual_block_2d(filters[0], filters[0], projection=False),
            residual_block_2d(filters[0], filters[0], projection=True),
            SELayer(channel=filters[0], channel_reduction=2, spatial_reduction=16)
        )
        self.layer2 = nn.Sequential(
            residual_block_2d(filters[0], filters[1], projection=True),
            residual_block_2d(filters[1], filters[1], projection=False),
            residual_block_2d(filters[1], filters[1], projection=True),
            SELayer(channel=filters[1], channel_reduction=4, spatial_reduction=8)
        )
        self.layer3 = nn.Sequential(
            residual_block_3d(filters[1], filters[2], projection=True),
            residual_block_3d(filters[2], filters[2], projection=False),
            residual_block_3d(filters[2], filters[2], projection=True),
            SELayer(channel=filters[2], channel_reduction=8, spatial_reduction=4)
        )
        self.layer4 = nn.Sequential(
            bottleneck_dilated(filters[2], filters[3], projection=True),
            bottleneck_dilated(filters[3], filters[3], projection=False),
            bottleneck_dilated(filters[3], filters[3], projection=True),
            SELayer(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )
        self.layer5 = nn.Sequential(
            bottleneck_dilated(filters[3], filters[4], projection=True),
            bottleneck_dilated(filters[4], filters[4], projection=False),
            bottleneck_dilated(filters[4], filters[4], projection=True),
            SELayer(channel=filters[3], channel_reduction=16, spatial_reduction=2, z_reduction=2)
        )

        # pooling & upsample
        self.down = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)

        # conv + upsample
        self.conv2 = conv3d_bn_elu(filters[1], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv3 = conv3d_bn_elu(filters[2], filters[0], kernel_size=(1,1,1), padding=(0,0,0)) 
        self.conv4 = conv3d_bn_elu(filters[3], filters[0], kernel_size=(1,1,1), padding=(0,0,0))
        self.conv5 = conv3d_bn_elu(filters[3], filters[0], kernel_size=(1,1,1), padding=(0,0,0))

        # convert to probability
        self.fconv = conv3d_bn_non(filters[0], out_channel, kernel_size=(3,3,3), padding=(1,1,1))

    def forward(self, x):

        z1 = self.layer1(x)
        x = self.down(z1)
        z2 = self.layer2(x)
        x = self.down(z2)
        z3 = self.layer3(x)
        x = self.down(z3)
        z4 = self.layer4(x)
        x = self.down(z4)
        z5 = self.layer5(x)

        y5 = self.conv5(z5)
        y4 = self.conv4(z4)
        y3 = self.conv3(z3)
        y2 = self.conv2(z2)
        y1 = z1

        out = self.up(self.up(self.up(self.up(y5)+y4)+y3)+y2)+y1
        #out = self.up(self.up(y5+y4+y3)+y2)+y1
        out = self.fconv(out)
        out = torch.sigmoid(out)
        return out

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
    model = fpn()
    print('model type: ', model.__class__.__name__)
    num_params = sum([p.data.nelement() for p in model.parameters()])
    print('number of trainable parameters: ', num_params)
    x = torch.randn(8, 1, 4, 64, 64)
    y = model(x)
    print(x.size(), y.size())

if __name__ == '__main__':
    test()