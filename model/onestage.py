'''
define one-stage model
with serveral backbone:
    - resnet18
    - resnet50
    - resnet101
    - swin-T
'''


import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
from collections import Counter 


from .layers import ReOrg, ResidualBlock, Upsample, conv_leaky, conv_down
from .detector import HeatmapHead, DetectHead, DetectHeadWithWeight


class LargeConv(nn.Module):
    '''
    multi-conv and concatenate
    '''
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, dilation=1, bias=True)
         
    def forward(self, x):
        y = self.conv1(x)  
        return y  

# class LargeConv(nn.Module):
#     '''
#     multi-conv and concatenate
#     '''
#     def __init__(self, c_in: int, c_out: int):
#         super().__init__()
#         self.conv1 = nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, dilation=1, bias=True)
#         self.conv3 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, dilation=1, bias=False)
#         self.conv5 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=2, dilation=2, bias=False)
#         self.conv9 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=4, dilation=4, bias=False)
#         self.conv15 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=8, dilation=8, bias=False)

#     def forward(self, x):
#         y = self.conv1(x) + self.conv3(x) + self.conv5(x) + self.conv9(x) + self.conv15(x)
#         return y  

class JointDetecotr(nn.Module):
    ''' 
    use five auxilary detector for different height-width-ratio 
    '''
    def __init__(self, channel: int, cls_num: int=2, scale_factor: float = 8.0, stride: int=5, weight: float=1.0, gamma: float=1.0):
        super().__init__()
        var = 2.0
        self.auxilary = nn.ModuleList()
        for center in [2,4,6,8,10]:
            self.auxilary.append(
                AuxilaryDetector(channel, cls_num, scale_factor, stride, weight, gamma, center, 2.0)
            )

    def forward(self, x, targets=None):
        loss = 0
        boxes = []
        for detector in self.auxilary:
            los, box = detecotr(x, targets)
            loss = loss + los 
            boxes.append(box)  
        if self.training:
            # update loss
            self.update_log()
            boxes = None 
        else:
            boxes = torch.cat(boxes, 1) 
        return loss, boxes 

    def update_log(self):
        '''
        update loss in tensorboard
        '''
        self.loss_log = Counter()
        for head in self.auxilary:
            for name in head.loss_log:
                self.loss_log[name] += head.loss_log[name]



class AuxilaryDetector(nn.Module):
    '''
    auxilary detector:
    bench & bamboo
    parameters:
    - channel: input feature channel
    - cls_num: predict oriented boxes channel
    - scale_factor: featuer resolution level
    - stride: center position for grid
    - weight: no obj weight
    - gamma: gamma in ring constrain
    - lb: lower bound for window ratios
    - ub: upper bound for window ratios
    '''
    def __init__(self,  channel: int, cls_num: int=2, scale_factor: float = 8.0, stride: int=5, weight: float=1.0, gamma: float=1.0, center=None, var=2.0):
        super().__init__()
        self.cls_num = 2 
        self.predictor = LargeConv(channel,6+cls_num)
        self.detecotr = DetectHeadWithWeight(cls_num, scale_factor, stride, weight, gamma)
        self.center = center
        self.var = var

    def assign_weight(self, w, h, theta):
        t = max(w,h)/(min(w,h)+1e-6)
        t = self.ratio(w,h,theta)
        t = -(t-self.center)**2
        t = np.exp(t/self.var)
        return t

    def targets_assign_weight(self, targets=None):
        '''
        assign the weight of offset loss
        for each targets
        '''
        if targets is None:
            return targets 
        boxes = targets.detach().cpu()
        tmp = []
        for box in boxes:
            w, h, theta = box[-3:]
            r = self.ratio(w,h,theta)
            if self.center is None:
                # assign weight 1
                v = 1.0
            else:
                v = self.assign_weight(w,h,theta)
            box = list(box)
            box.append(v)  
            zz = np.array(box)
            tmp.append(zz)
        tmp = np.array(tmp)
        tmp = torch.FloatTensor(tmp).to(device=targets.device)
        return tmp
        

    def forward(self, x, targets=None): 
        x = self.predictor(x) 
        if targets is not None:
            targets = self.targets_assign_weight(targets)
        loss, boxes = self.detecotr(x, targets)
        if self.training:
            self.loss_log = self.detecotr.loss_log
        return loss, boxes
 
    def shadow(self, w, h, theta):
        '''
        逆时针旋转theta角度后，
        长方形的投影长度
        长方形初始的四个顶点坐标：(±w,±h)
        (  cos -sin  )  (x)
        (   sin cos  )  (y)
        '''
        candidates = []
        for x in [-w, w]:
            for y in [-h,h]:
                v = np.cos(theta) * x - np.sin(theta) * y
                candidates.append(v)
        return max(candidates) - min(candidates)

    def ratio(self, w, h, theta):
        x = self.shadow(w,h,theta)
        y = self.shadow(w,h,theta+np.pi/2)
        return x/(y+1e-6)

    # def select(self, targets):
    #     '''
    #     choose targets in select range
    #     '''
    #     boxes = targets.detach().cpu()
    #     tmp = []
    #     for box in boxes:
    #         w, h, theta = box[-3:]
    #         r = self.ratio(w,h,theta)
    #         if self.lb is not None and self.lb>r:
    #             continue
    #         if self.ub is not None and self.ub<r:
    #             continue
    #         zz = np.array(box)
    #         tmp.append(zz)
    #     tmp = np.array(tmp)
    #     tmp = torch.FloatTensor(tmp).to(device=targets.device)
    #     return tmp


class NeckModule(nn.Module):
    '''
    FPN Neck
    '''
    def __init__(self, input_feature_channels, channel_out):
        super().__init__()
        self.c1 = c1 = input_feature_channels[0]
        self.c2 = c2 = input_feature_channels[1]
        self.c3 = c3 = input_feature_channels[2]
        self.cc = cc = channel_out
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

    def forward(self, f1, f2, f3): 
        f3 = self.conv3(f3)
        x = self.up3(f3) 
        x = torch.cat([x, f2], 1)
        f2 = self.conv2(x)
        x = self.up2(f2)
        x = torch.cat([x, f1], 1)
        f1 = self.conv1(x)
        return f1, f2, f3


class OneStage(nn.Module):
    '''
    Model with Two auxilary Predictor
    bench & bamboo
    '''
    def __init__(self, cls_num: int=2, cfg=None):
        super().__init__()
        backboneName = cfg.get('backbone')
        assert backboneName in {
            'resnet18', 'resnet50', 'resnet101',
            'swin-t'
        }
        # parse cfg file
        contour_weight = cfg.get('contour_weight')
        rect_weight = cfg.get('rect_weight')
        rect_stride = cfg.get('rect_stride')
        gamma = cfg.get('gamma')   # detect orient box position
        if cfg.get('dataset') == 'hrsc':
            auxilary = True
        else:
            auxilary = False
        # backbone
        self.build_backbone(backboneName)
        # neck
        channel = min(self.feature_channels)
        self.neck = NeckModule(self.feature_channels, channel)
        # head
        self.contour2 = LargeConv(channel, cls_num)
        self.contour3 = LargeConv(channel, cls_num)
        self.contour4 = LargeConv(channel, cls_num)
          # 6 head
        self.heat2 = HeatmapHead(cls_num, 8.0, contour_weight)  # at 1/8 resolution
        self.heat3 = HeatmapHead(cls_num, 16.0, contour_weight)  # at 1/16 resolution
        self.heat4 = HeatmapHead(cls_num, 32.0, contour_weight)  # at 1/32 resolution
        self.head2 = self.build_detecotr(auxilary, channel, cls_num, 8.0, rect_stride, rect_weight, gamma)
        self.head3 = self.build_detecotr(auxilary, channel, cls_num, 16.0, rect_stride, rect_weight, gamma)
        self.head4 = self.build_detecotr(auxilary, channel, cls_num, 32.0, rect_stride, rect_weight, gamma)
        self.heads = [
            self.heat2, self.heat3, self.heat4,
            self.head2, self.head3, self.head4,
        ]
        # done

    def forward(self, x, targets=None):
        # backbone forward
        x = self.layer1(x)
        x = self.layer2(x)
        f2 = self.permute(x)
        x = self.layer3(x)
        f3 = self.permute(x)
        x = self.layer4(x)
        f4 = self.permute(x)
        # fpn neck
        f2, f3, f4 = self.neck(f2,f3,f4)
        # TODO:
        # head at heatmap
        # multiply feature by heatmap
        # head at oriented bounding box
        # loss total
        # post A: compute heatmap
        x = self.contour2(f2)
        lossA2, heat2 = self.heat2(x, targets)
        x = self.contour3(f3)
        lossA3, heat3 = self.heat3(x, targets)
        x = self.contour4(f4)
        lossA4, heat4 = self.heat4(x, targets)
        # multiply feature by heatmap
        f4 = f4 * heat4.unsqueeze(1)
        f3 = f3 * heat3.unsqueeze(1)
        f2 = f2 * heat2.unsqueeze(1)
        # detector
        lossB2, box2 = self.head2(f4, targets)
        lossB3, box3 = self.head3(f3, targets)
        lossB4, box4 = self.head4(f2, targets)
        loss = lossA2 + lossA3 + lossA4 + lossB2 + lossB3 + lossB4
        if self.training:
            # only compute output at eval time
            boxes = None
        else:
            boxes = torch.cat([box2, box3, box4], 1).detach().cpu()
        return loss, boxes


    def build_detecotr(self, auxilary, channel: int, cls_num: int=2, scale_factor: float = 8.0, stride: int=5, weight: float=1.0, gamma: float=1.0):

        '''
        build two auxilary_detecotr
        '''
        if auxilary:
            return JointDetecotr(channel, cls_num, scale_factor, stride, weight, gamma)
        return AuxilaryDetector(channel, cls_num, scale_factor, stride, weight, gamma)


    def build_backbone(self, name):
        if name.startswith('resnet'):
            if name=='resnet18':
                # self.feature_channels = [64, 128, 256, 512]
                self.feature_channels = [128, 256, 512]
                tmp = tv.models.resnet18(pretrained=True)
            elif name=='resnet50':
                # self.feature_channels = [256, 512, 1024, 2048]
                self.feature_channels = [512, 1024, 2048]
                tmp = tv.models.resnet50(pretrained=True)
            else:
                # self.feature_channels = [256, 512, 1024, 2048]
                self.feature_channels = [512, 1024, 2048]
                tmp = tv.models.resnet101(pretrained=True)
            self.layer1= nn.Sequential(
                tmp.conv1,
                tmp.bn1,
                tmp.relu,
                tmp.maxpool,
                tmp.layer1
            )    # at 1/4 resolution
            self.layer2 = tmp.layer2 # at 1/8 resolution
            self.layer3 = tmp.layer3 # at 1/16 resolution
            self.layer4 = tmp.layer4 # at 1/32 resolution
            self.permute = self.resnet_permute
        else:
            # swin-t
            tmp = tv.models.swin_t(pretrained=True).features
            # self.feature_channels = [96, 192, 384, 768]
            self.feature_channels = [192, 384, 768]
            self.layer1 = nn.Sequential(
                tmp[0],
                tmp[1], 
            )  # at 1/4 resolution
            self.layer2 = nn.Sequential(
                tmp[2],
                tmp[3], 
            )   # at 1/8 resolution
            self.layer3 =  nn.Sequential(
                tmp[4],
                tmp[5], 
            )   # at 1/16 resolution
            self.layer4 =nn.Sequential(
                tmp[6],
                tmp[7], 
            )    # at 1/32 resolution
            self.permute = self.swin_permute

    def resnet_permute(self, x):
        return x

    def swin_permute(self, x):
        return x.permute(0,3,1,2).contiguous()

