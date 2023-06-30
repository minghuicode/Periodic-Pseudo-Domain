'''
define our twin-feature backbone
'''
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101
# from .backbone import resnet18, resnet34, resnet50, resnet101, resnet152, darknet53

from .layers import ReOrg, ResidualBlock, Upsample, conv_leaky, conv_down
from .detector import HeatmapHead, DetectHead

from utils.r_nms import r_nms


import time


def timer(f):
    def _wrapper(*args, **kwargs):
        start = time.time()
        rv = f(*args, **kwargs)
        print("function {} cost {:.2f} seconds.".format(
            f.__name__, time.time()-start))
        return rv
    return _wrapper


class LargeConv(nn.Module):
    '''
    multi-conv and concatenate
    '''
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, dilation=1, bias=True)
        self.conv3 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv5 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=2, dilation=2, bias=False)
        self.conv9 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=4, dilation=4, bias=False)
        self.conv15 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=8, dilation=8, bias=False)

    def forward(self, x):
        y = self.conv1(x) + self.conv3(x) + self.conv5(x) + self.conv9(x) + self.conv15(x)
        return y 

class NeckModule(nn.Module):
    '''
    all module used in neck
    '''

    def __init__(self,  c1: int, c2: int, c3: int, cls_num: int):
        super().__init__()
        self._cls_num = cls_num
        self.c1 = c1  # at 1/4 resolution
        self.c2 = c2  # at 1/8 resolution
        self.c3 = c3  # at 1/16 resolution
        self.cc = cc = min(c1, c2, c3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(c3, cc, 3, 1, 1),
            nn.BatchNorm2d(cc),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            # conv_leaky(cc, c2, 1),
            Upsample(2.0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cc+c2, cc, 3, 1, 1),
            nn.BatchNorm2d(cc),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            # conv_leaky(cc, c1, 1),
            Upsample(2.0)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(cc+c1, cc, 3, 1, 1),
            nn.BatchNorm2d(cc),
            nn.ReLU()
        )
        # below module was used in father network
        # self.contour1 = nn.Conv2d(cc, cls_num, 1)
        # self.contour2 = nn.Conv2d(cc, cls_num, 1)
        # self.contour3 = nn.Conv2d(cc, cls_num, 1)
        # self.post1 = nn.Conv2d(cc, cls_num+6, 1)
        # self.post2 = nn.Conv2d(cc, cls_num+6, 1)
        # self.post3 = nn.Conv2d(cc, cls_num+6, 1)
        self.contour1 = LargeConv(cc, cls_num)
        self.contour2 = LargeConv(cc, cls_num)
        self.contour3 = LargeConv(cc, cls_num)
        self.post1 = LargeConv(cc, cls_num+6)
        self.post2 = LargeConv(cc, cls_num+6)
        self.post3 = LargeConv(cc, cls_num+6) 

    def forward(self, f1, f2, f3):
        f3 = self.conv3(f3)
        x = self.up3(f3)
        x = torch.cat([x, f2], 1)
        f2 = self.conv2(x)
        x = self.up2(f2)
        x = torch.cat([x, f1], 1)
        f1 = self.conv1(x)
        return f1, f2, f3


class TwinRes(nn.Module):
    '''
    use resnet as backbone
    '''

    def __init__(self, cls_num: int = 2, cfg=None):
        super().__init__() 
        self._cls_num = cls_num
        # backbone = 'resnet50'
        backbone = cfg.get('backbone')
        assert backbone in {'resnet18', 'resnet50', 'resnet101'}
        self.backboneName = backbone
        if backbone == 'resnet18':
            c1, c2, c3 = 128, 256, 512
            self.backbone = resnet18(pretrained=True)
        elif backbone == 'resnet50':
            c1, c2, c3 = 512, 1024, 2048
            self.backbone = resnet50(pretrained=True)
        elif backbone == 'resnet101':
            c1, c2, c3 = 512, 1024, 2048
            self.backbone = resnet101(pretrained=True)
        # elif backbone == 'darknet53':
        #     c1, c2, c3 = 1024, 512, 256
        #     self.backbone = Darknet53()   
        contour1_scale = cfg.get('contour1_scale')   
        contour2_scale = cfg.get('contour2_scale')   
        contour3_scale = cfg.get('contour3_scale') 
        rect1_scale = cfg.get('rect1_scale') 
        rect2_scale = cfg.get('rect2_scale') 
        rect3_scale = cfg.get('rect3_scale')
        rect1_stride = cfg.get('rect1_stride') 
        rect2_stride = cfg.get('rect2_stride') 
        rect3_stride = cfg.get('rect3_stride') 
        gamma = cfg.get('gamma')
        # 6 head
        self.heat1 = HeatmapHead(cls_num, 8.0, contour1_scale)  # at 1/8 resolution
        self.heat2 = HeatmapHead(cls_num, 16.0, contour2_scale)  # at 1/16 resolution
        self.heat3 = HeatmapHead(cls_num, 32.0, contour3_scale)  # at 1/32 resolution
        # detect orient box position
        self.head1 = DetectHead(cls_num, 8.0, rect1_stride, rect1_scale, gamma)   # at 1/8 resolution
        self.head2 = DetectHead(cls_num, 16.0, rect2_stride, rect2_scale, gamma)   # at 1/16 resolution
        self.head3 = DetectHead(cls_num, 32.0, rect3_stride, rect3_scale, gamma)   # at 1/32 resolution
        # post B: branch for detect head
        self.heads = [
            self.heat1, self.heat2, self.heat3,
            self.head1, self.head2, self.head3,
        ]
        self.neck = NeckModule(c1, c2, c3, cls_num)
        # functional: upsample x2
        self.up = Upsample(2.0)
        self.pool = nn.MaxPool2d(3, 1, padding=1)
        # set loss mode 
        self.set_loss_mode() 

    def set_loss_mode(self, mode='all'):
        assert mode in {'all', 'A', 'B', 'C'}
        self.head1.set_loss_mode(mode)
        self.head2.set_loss_mode(mode)
        self.head3.set_loss_mode(mode)
        self.loss_mode = mode 

    def forward(self, x, targets=None):
        # backbone is resnet
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # at 1/2 resolution
        x = self.backbone.layer1(x)  # at 1/4 resolution
        f1 = x = self.backbone.layer2(x)  # at 1/8 resolution
        f2 = x = self.backbone.layer3(x)  # at 1/16 resolution
        f3 = x = self.backbone.layer4(x)  # at 1/32 resolution
        # compute FPN feature
        f1, f2, f3 = self.neck(f1, f2, f3)
        # post A: compute heatmap
        x = self.neck.contour1(f1)
        lossA1, heat1 = self.heat1(x, targets)
        x = self.neck.contour2(f2)
        lossA2, heat2 = self.heat2(x, targets)
        x = self.neck.contour3(f3)
        lossA3, heat3 = self.heat3(x, targets)
        # heat to binary
        # heat3 = self.pool(heat3.unsqueeze(1))
        # heat2 = self.pool(heat2.unsqueeze(1))
        # heat1 = self.pool(heat1.unsqueeze(1))
        # heat3 = (heat3>0.5).float()
        # heat2 = (heat2>0.5).float()
        # heat1 = (heat1>0.5).float()
        # multiply feature by heatmap
        f3 = f3 * heat3.unsqueeze(1)
        f2 = f2 * heat2.unsqueeze(1)
        f1 = f1 * heat1.unsqueeze(1)
        # compute box
        # post B: compute oriented bounding box
        x = self.neck.post1(f1)
        lossB1, box1 = self.head1(x, targets)
        x = self.neck.post2(f2)
        lossB2, box2 = self.head2(x, targets)
        x = self.neck.post3(f3)
        lossB3, box3 = self.head3(x, targets)
        loss = lossA1 + lossA2 + lossA3 + lossB1 + lossB2 + lossB3
        if self.training:
            # only compute output at eval time
            boxes = None
        else:
            boxes = torch.cat([box1, box2, box3], 1).detach().cpu()
            # boxes = torch.cat([box1, box2], 1).detach().cpu()
            # boxes = box3.detach().cpu()
        return loss, boxes

 