'''
detector head
create date: 2022-05-17 14:18
'''
import torch 
import torch.nn as nn 
import numpy as np 
from torch.autograd import Variable

# from shapely.geometry import Polygon
# from shapely.geometry.point import Point 
from collections import defaultdict


__all__ = ['DetectHead', 'HeatmapHead', 'DetectHeadWithWeight']
 
import time 

def timer(f):
    def _wrapper(*args, **kwargs):
        start = time.time()
        rv = f(*args, **kwargs)
        print("function {} cost {:.2f} seconds.".format(f.__name__,time.time()-start))
        return rv 
    return _wrapper

'''
Loss Define
'''
class AngleLoss(nn.Module):
    '''
    compute weight L1 loss for a given angle
    '''
    def __init__(self, weight, gamma=1.0):
        super().__init__()
        self.weight = torch.abs(weight) ** gamma

    def forward(self, x, y, angle):
        loss = torch.abs(x-torch.cos(angle)) + torch.abs(y-torch.sin(angle))
        loss = self.weight * loss
        return loss.mean()

class RingLoss(nn.Module):
    '''
    compute ring penalty item
    '''
    def __init__(self, weight, gamma=1.0):
        super().__init__()
        self.weight = torch.abs(weight)   

    def forward(self, x, y): 
        loss = torch.abs(x*x+y*y-1)
        loss = self.weight * loss
        return loss.mean()


'''
head map
'''

def render_map(predict, targets=None):
    '''
    render binary map for a single image 
    ''' 
    nB, nH, nW, _ = predict.shape 
    device = predict.device 
    obj_mask = torch.FloatTensor(nH,nW).to(device).fill_(0)
    noobj_mask = torch.FloatTensor(nH,nW).to(device).fill_(1)
    if targets is None or len(targets)==0:
        return  obj_mask.bool(), noobj_mask.bool(), obj_mask.float()  
    # grid points 
    grid_x = torch.arange(nW).repeat(nH, 1).view(
        [nH, nW]
    ).to(torch.float).to(device) + 0.5 
    grid_y = torch.arange(nH).repeat(nW, 1).t().view(
        [nH, nW]
    ).to(torch.float).to(device) + 0.5 
    for box in targets:
        cx = box[2]
        cy = box[3]
        w = box[4]
        h = box[5]
        ang = box[6]
        # center and offset in ang 
        x = cx * torch.cos(ang) + cy*torch.sin(ang)
        y = -cx*torch.sin(ang) + cy*torch.cos(ang) 
        # x1, x2 = x-1.2*h/2, x+1.2*h/2 
        # y1, y2 = y-1.2*w/2, y+1.2*w/2 
        # x3, x4 = x-0.8*h/2, x+0.8*h/2
        # y3, y4 = y-0.8*w/2, y+0.8*w/2 
        x3, x4 = x-1.0*h/2, x+1.0*h/2
        y3, y4 = y-1.0*w/2, y+1.0*w/2 
        x1, x2 = x3-0.5, x4+0.5
        y1, y2 = y3-0.5, y4+0.5 
        # get value map 
        vx = grid_x * torch.cos(ang) + grid_y*torch.sin(ang)
        vy = -grid_x * torch.sin(ang) + grid_y*torch.cos(ang)  
        mask = (x3<=vx) * (x4>=vx) * (y3<=vy) * (y4>=vy)
        obj_mask[mask] = 1.0   
        mask = (x1<=vx) * (x2>=vx) * (y1<=vy) * (y2>=vy)
        noobj_mask[mask] = 0.0 
    return obj_mask.bool(), noobj_mask.bool(), obj_mask.float() 

def build_heatmap(predict, targets):
    '''
    render heatmap for image batch
    ''' 
    nB, nH, nW, _ = predict.shape 
    device = predict.device 
    objs = []
    noobjs = []
    confs = [] 
    for b in range(nB):
        tmp = targets[targets[:,0]==b]  
        obj, noobj, conf = render_map(predict, tmp)
        objs.append(obj)
        noobjs.append(noobj)
        confs.append(conf)
    obj_mask = torch.stack(objs)
    noobj_mask = torch.stack(noobjs)
    tconf = torch.stack(confs) 
    return obj_mask, noobj_mask, tconf 


class DetectHead(nn.Module):
    '''
    predict object position
    ''' 
    def __init__(self, cls_num: int = 2, scale_factor: float = 8.0, stride: int=5, no_obj_scale: float=1.0, gamma: float=1.0):
        super().__init__()
        self._cls_num = cls_num
        self._scale_factor = scale_factor 
        self._stride = stride # max center point offset 
        self._no_obj_scale = no_obj_scale
        self._gamma = gamma
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss() 
        # tensor board loss
        self.loss_log = {}
        # set loss mode 
        self.set_loss_mode() 

    @property
    def channel(self):
        '''
        expect channel number
        '''
        return 6+self._cls_num 

    def set_loss_mode(self, mode='all'):
        assert mode in {'all', 'A', 'B', 'C'} 
        self.loss_mode = mode 
     
    def forward(self, predict, targets=None):
        '''
        if target is None:
            just show output 
        if training:
            just show loss 
        ''' 
        device = predict.device
        nB, _, nH, nW = predict.shape
        stride = self._stride
        cls_num = self._cls_num
        predict = predict.permute(0,2,3,1).contiguous() 
        # predict boxes active 
        # x = torch.sigmoid(predict[...,0])*2 - 0.5 # center x offset 
        # y = torch.sigmoid(predict[...,1])*2 - 0.5 # center y offset 
        x = torch.sigmoid(predict[...,0])*stride - (stride-1)/2 # center x offset 
        y = torch.sigmoid(predict[...,1])*stride - (stride-1)/2 # center y offset 
        w = predict[..., 2] # short side length 
        h = predict[..., 3] # long side length 
        p = torch.sigmoid(predict[..., 4])*2 - 1 # cos(2t) 
        q = torch.sigmoid(predict[..., 5])*2 - 1 # sin(2t) 
        if targets is None:
            loss = 0 
        else:
            # compute loss 
            # register loss
            self.loss_log['offset/x'] = 0.0
            self.loss_log['offset/y'] = 0.0
            self.loss_log['offset/w'] = 0.0
            self.loss_log['offset/h'] = 0.0 
            self.loss_log['offset/angle'] = 0.0 
            self.loss_log['loss/offset'] = 0.0
            self.loss_log['loss/conf'] = 0.0
            self.loss_log['loss/total'] = 0.0
            self.loss_log['conf/obj'] = 0.0
            self.loss_log['conf/noobj'] = 0.0
            loss = 0  
            loss_A = 0 # angle loss 
            loss_B = 0 # regression loss 
            loss_C = 0 # ring constrain loss 
            targets[:, 2:6] /= self._scale_factor
            for cls_id in range(cls_num):
                # select certain class 
                tmp = targets[targets[:, 1] == cls_id] 
                if len(tmp) == 0:
                    continue 
                conf = torch.sigmoid(predict[..., 6+cls_id])
                # build targets 
                obj_mask, noobj_mask, tx, ty, tw, th, ta, tconf, tweight = self.build_targets(
                    predict, tmp) 
                if torch.sum(obj_mask).item()==0:
                    # do nothing
                    continue   
                # confidence loss 
                bce_loss = self.bce_loss
                loss_conf_obj = bce_loss(conf[obj_mask], tconf[obj_mask])   
                loss_conf_noobj = self.bce_loss(
                    conf[noobj_mask], tconf[noobj_mask])  * self._no_obj_scale
                loss_conf = loss_conf_obj + loss_conf_noobj
                # offset loss   
                loss_x = self.mse_loss(tweight[obj_mask]*x[obj_mask], tweight[obj_mask]*tx[obj_mask])
                loss_y = self.mse_loss(tweight[obj_mask]*y[obj_mask], tweight[obj_mask]*ty[obj_mask]) 
                loss_w = self.mse_loss(tweight[obj_mask]*w[obj_mask], tweight[obj_mask]*tw[obj_mask])
                loss_h = self.mse_loss(tweight[obj_mask]*h[obj_mask], tweight[obj_mask]*th[obj_mask])
                # direction loss 
                weight = th[obj_mask].data - tw[obj_mask].data # 长宽比越小，角度越不重要  ,　长宽比越大，宽度越重要
                weight.requires_grad = False 
                angle_loss = AngleLoss(weight, self._gamma)
                loss_angle = angle_loss(p[obj_mask],q[obj_mask],2*ta[obj_mask]) 
                ring_loss = RingLoss(weight)
                loss_ring = ring_loss(p[obj_mask], q[obj_mask])
                loss_A = loss_A + loss_angle
                loss_B = loss_B + loss_x + loss_y + loss_w + loss_h
                loss_C = loss_C + loss_ring 
                loss_offset = loss_x + loss_y + loss_w + loss_h + loss_angle + loss_ring
                # total loss  
                loss = loss + loss_conf
                # log loss for each type of loss
                self.loss_log['offset/x'] += loss_x.item()
                self.loss_log['offset/y'] += loss_y.item()
                self.loss_log['offset/w'] += loss_w.item()
                self.loss_log['offset/h'] += loss_h.item()
                self.loss_log['offset/angle'] += loss_angle.item()  
                self.loss_log['loss/offset'] += loss_offset.item()
                self.loss_log['loss/conf'] += loss_conf.item()
                #self.loss_log['loss/total'] += loss_total.item()
                self.loss_log['conf/obj'] += loss_conf_obj.item()
                self.loss_log['conf/noobj'] += loss_conf_noobj.item() 
                #self.loss_log['conf/cls'+str(cls_id)] = loss_total.item() 
            targets[:, 2:6] *= self._scale_factor 
            # loss mode 
            if self.loss_mode == 'all':
                loss = loss + loss_A + loss_B + loss_C
            elif self.loss_mode == 'A':
                loss = loss + loss_A
            elif self.loss_mode == 'B':
                loss = loss + loss_B
            elif self.loss_mode == 'C':
                loss = loss + loss_C
            else:
                assert ValueError('UnExpect loss mode: ', self.loss_mode)
        if self.training:
            output = None 
        else:
            # compute output at eval time 
            # predict cls score 
            score = torch.sigmoid(predict[..., 6:])
            conf, cls_id = torch.max(score, dim=-1) 
            # combine the output boxes
            grid_x = torch.arange(nW).repeat(nH, 1).view(
                [1, nH, nW]
            ).to(torch.float).to(device)
            grid_y = torch.arange(nH).repeat(nW, 1).t().view(
                [1, nH, nW]
            ).to(torch.float).to(device)
            boxes = torch.FloatTensor(nB, nH, nW, 7).to(device)
            # multiply the confidence 
            boxes[..., 0] = conf 
            boxes[..., 1] = cls_id 
            boxes[..., 2] = grid_x + x 
            boxes[..., 3] = grid_y + y 
            boxes[..., 4] = torch.exp(w) 
            boxes[..., 5] = torch.exp(h) 
            t = torch.sqrt((1+p)/2 + 1e-16) # cos(t) 
            boxes[..., 6] = torch.atan2(q/(2*t+1e-16), t) 
            output = boxes.view(nB, -1, 7)
            # rescale boxes size to pixel measure 
            output[..., 2:6] *= self._scale_factor
        return loss, output 
 

    def build_targets(self, predict, targets): 
        '''
        over-class object overlap was not considered
        '''
        nB, nH, nW, _ = predict.shape
        device = predict.device 
        stride = self._stride
        lower = 0.5-stride/2
        upper = 0.5+stride/2
        # distribute label in cpu 
        boxes = targets.detach().cpu().numpy() 
        # convert to position relative to box 
        # label prior
        prior = {} 
        # obj_mask
        obj_mask = np.zeros([nB,nH,nW])
        noobj_mask = np.ones([nB,nH,nW])
        # heatmap mask
        heat_mask, cold_mask, _ = build_heatmap(predict, targets) 
        # regressor params: 5
        tx = np.zeros([nB,nH,nW])
        ty = np.zeros([nB,nH,nW])
        tw = np.zeros([nB,nH,nW])
        th = np.zeros([nB,nH,nW]) 
        ta = np.zeros([nB,nH,nW])
        tweight = np.zeros([nB,nH,nW])
        for box in boxes:
            b = int(box[0]) # batch id 
            cx = box[2]
            cy = box[3]
            w = box[4]
            h = box[5]
            ang = box[6]
            p = np.cos(2*box[6])
            q = np.sin(2*box[6]) 
            # compute positive label 
            # for i in  [int(cx)-2, int(cx)-1, int(cx), int(cx)+1,int(cx)+2]:
            #     for j in [int(cy)-2,int(cy)-2, int(cy), int(cy)+1,int(cy)+2]: 
            for i in  range(int(cx+lower),int(cx+upper)+1):  
                for j in range(int(cy+lower),int(cy+upper)+1): 
                    if 0<=i<nW and 0<=j<nH:
                        pass
                    else:
                        continue 
                    # compute x offset and y offset 
                    x = cx - i 
                    y = cy - j 
                    # if -0.5<=x<=1.5 and -0.5<=y<=1.5:
                    #     pass 
                    # else:
                    #     continue 
                    # if -2.0<=x<=3.0 and -2<=y<=3.0: 
                    #     pass 
                    # else:
                    #     continue 
                    if lower<=x<=upper and lower<=y<=upper:
                        pass 
                    else:
                        continue 

                    if not cold_mask[b,j,i]:
                        noobj_mask[b,j,i] = 0.0 
                    # set positive
                    if not heat_mask[b,j,i]:
                        continue 

                    # prior value 
                    # vx = (x-0.5) * np.cos(ang) + (y-0.5) * np.sin(ang)
                    # vy = -(x-0.5) * np.sin(ang) + (y-0.5)*np.cos(ang)
                    # v = min(abs(vx)/h,abs(vy)/w)
                    v = min(abs(x-0.5), abs(y-0.5))
                    position = (b,i,j)
                    if position not in prior:
                        prior[position] = v 
                    elif prior[position]>v:
                        prior[position] = v 
                    else:
                        # do nothing
                        continue 
                    # set positive 
                    obj_mask[b,j,i] = 1.0  
                    noobj_mask[b,j,i] = 0.0  
                    # set offset 
                    tx[b,j,i] = x 
                    ty[b,j,i] = y 
                    # set short side length 
                    tw[b,j,i] = np.log(w+1e-16)
                    # set long side length
                    th[b,j,i] = np.log(h+1e-16)
                    # set rotate angles 
                    ta[b,j,i] = box[6]
                    tweight[b,j,i] = box[7] 
        # convert to torch style
        obj_mask = torch.from_numpy(obj_mask).bool().to(device)
        noobj_mask = torch.from_numpy(noobj_mask).bool().to(device)
        tx = torch.from_numpy(tx).float().to(device) 
        ty = torch.from_numpy(ty).float().to(device) 
        tw = torch.from_numpy(tw).float().to(device) 
        th = torch.from_numpy(th).float().to(device) 
        ta = torch.from_numpy(ta).float().to(device)  
        tweight = torch.from_numpy(tweight).float().to(device)
        return obj_mask, noobj_mask, tx, ty, tw, th, ta, obj_mask.float(), tweight
 
 
class DetectHeadWithWeight(nn.Module):
    '''
    predict object position
    at each offset: (x,y,w,h) 
    with targets weights
    ''' 
    def __init__(self, cls_num: int = 2, scale_factor: float = 8.0, stride: int=5, no_obj_scale: float=1.0, gamma: float=1.0):
        super().__init__()
        self._cls_num = cls_num
        self._scale_factor = scale_factor 
        self._stride = stride # max center point offset 
        self._no_obj_scale = no_obj_scale
        self._gamma = gamma
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss() 
        # tensor board loss
        self.loss_log = {} 

    @property
    def channel(self):
        '''
        expect channel number
        '''
        return 6+self._cls_num 
 
     
    def forward(self, predict, targets=None):
        '''
        if target is None:
            just show output 
        if training:
            just show loss 
        ''' 
        device = predict.device
        nB, _, nH, nW = predict.shape
        stride = self._stride
        cls_num = self._cls_num
        predict = predict.permute(0,2,3,1).contiguous() 
        # predict boxes active 
        # x = torch.sigmoid(predict[...,0])*2 - 0.5 # center x offset 
        # y = torch.sigmoid(predict[...,1])*2 - 0.5 # center y offset 
        x = torch.sigmoid(predict[...,0])*stride - (stride-1)/2 # center x offset 
        y = torch.sigmoid(predict[...,1])*stride - (stride-1)/2 # center y offset 
        w = predict[..., 2] # short side length 
        h = predict[..., 3] # long side length 
        p = torch.sigmoid(predict[..., 4])*2 - 1 # cos(2t) 
        q = torch.sigmoid(predict[..., 5])*2 - 1 # sin(2t) 
        if targets is None:
            loss = 0 
        else:
            # compute loss 
            # register loss
            self.loss_log['offset/x'] = 0.0
            self.loss_log['offset/y'] = 0.0
            self.loss_log['offset/w'] = 0.0
            self.loss_log['offset/h'] = 0.0 
            self.loss_log['offset/angle'] = 0.0 
            self.loss_log['loss/offset'] = 0.0
            self.loss_log['loss/conf'] = 0.0
            self.loss_log['loss/total'] = 0.0
            self.loss_log['conf/obj'] = 0.0
            self.loss_log['conf/noobj'] = 0.0
            loss = 0  
            loss_A = 0 # angle loss 
            loss_B = 0 # regression loss 
            loss_C = 0 # ring constrain loss 
            targets[:, 2:6] /= self._scale_factor
            for cls_id in range(cls_num):
                # select certain class 
                tmp = targets[targets[:, 1] == cls_id] 
                if len(tmp) == 0:
                    continue 
                conf = torch.sigmoid(predict[..., 6+cls_id])
                # build targets 
                obj_mask, noobj_mask, tx, ty, tw, th, ta, tconf = self.build_targets(
                    predict, tmp) 
                if torch.sum(obj_mask).item()==0:
                    # do nothing
                    continue   
                # confidence loss 
                bce_loss = self.bce_loss
                loss_conf_obj = bce_loss(conf[obj_mask], tconf[obj_mask])   
                loss_conf_noobj = self.bce_loss(
                    conf[noobj_mask], tconf[noobj_mask])  * self._no_obj_scale
                loss_conf = loss_conf_obj + loss_conf_noobj
                # offset loss   
                loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
                loss_y = self.mse_loss(y[obj_mask], ty[obj_mask]) 
                loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
                loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
                # direction loss 
                weight = th[obj_mask].data - tw[obj_mask].data # 长宽比越小，角度越不重要  ,　长宽比越大，宽度越重要
                weight.requires_grad = False 
                angle_loss = AngleLoss(weight, self._gamma)
                loss_angle = angle_loss(p[obj_mask],q[obj_mask],2*ta[obj_mask]) 
                ring_loss = RingLoss(weight)
                loss_ring = ring_loss(p[obj_mask], q[obj_mask])
                loss_A = loss_A + loss_angle
                loss_B = loss_B + loss_x + loss_y + loss_w + loss_h
                loss_C = loss_C + loss_ring 
                loss_offset = loss_x + loss_y + loss_w + loss_h + loss_angle + loss_ring
                # total loss  
                loss = loss + loss_conf
                # log loss for each type of loss
                self.loss_log['offset/x'] += loss_x.item()
                self.loss_log['offset/y'] += loss_y.item()
                self.loss_log['offset/w'] += loss_w.item()
                self.loss_log['offset/h'] += loss_h.item()
                self.loss_log['offset/angle'] += loss_angle.item()  
                self.loss_log['loss/offset'] += loss_offset.item()
                self.loss_log['loss/conf'] += loss_conf.item()
                #self.loss_log['loss/total'] += loss_total.item()
                self.loss_log['conf/obj'] += loss_conf_obj.item()
                self.loss_log['conf/noobj'] += loss_conf_noobj.item() 
                #self.loss_log['conf/cls'+str(cls_id)] = loss_total.item() 
            targets[:, 2:6] *= self._scale_factor  
            loss = loss + loss_A + loss_B + loss_C 
        if self.training:
            output = None 
        else:
            # compute output at eval time 
            # predict cls score 
            score = torch.sigmoid(predict[..., 6:])
            conf, cls_id = torch.max(score, dim=-1) 
            # combine the output boxes
            grid_x = torch.arange(nW).repeat(nH, 1).view(
                [1, nH, nW]
            ).to(torch.float).to(device)
            grid_y = torch.arange(nH).repeat(nW, 1).t().view(
                [1, nH, nW]
            ).to(torch.float).to(device)
            boxes = torch.FloatTensor(nB, nH, nW, 7).to(device)
            # multiply the confidence 
            boxes[..., 0] = conf 
            boxes[..., 1] = cls_id 
            boxes[..., 2] = grid_x + x 
            boxes[..., 3] = grid_y + y 
            boxes[..., 4] = torch.exp(w) 
            boxes[..., 5] = torch.exp(h) 
            t = torch.sqrt((1+p)/2 + 1e-16) # cos(t) 
            boxes[..., 6] = torch.atan2(q/(2*t+1e-16), t) 
            output = boxes.view(nB, -1, 7)
            # rescale boxes size to pixel measure 
            output[..., 2:6] *= self._scale_factor
        return loss, output 
 

    def build_targets(self, predict, targets): 
        '''
        over-class object overlap was not considered
        '''
        nB, nH, nW, _ = predict.shape
        device = predict.device 
        stride = self._stride
        lower = 0.5-stride/2
        upper = 0.5+stride/2
        # distribute label in cpu 
        boxes = targets.detach().cpu().numpy() 
        # convert to position relative to box 
        # label prior
        prior = {} 
        # obj_mask
        obj_mask = np.zeros([nB,nH,nW])
        noobj_mask = np.ones([nB,nH,nW])
        # heatmap mask
        heat_mask, cold_mask, _ = build_heatmap(predict, targets) 
        # regressor params: 5
        tx = np.zeros([nB,nH,nW])
        ty = np.zeros([nB,nH,nW])
        tw = np.zeros([nB,nH,nW])
        th = np.zeros([nB,nH,nW]) 
        ta = np.zeros([nB,nH,nW])
        for box in boxes:
            b = int(box[0]) # batch id 
            cx = box[2]
            cy = box[3]
            w = box[4]
            h = box[5]
            ang = box[6]
            p = np.cos(2*box[6])
            q = np.sin(2*box[6])
            # compute positive label 
            # for i in  [int(cx)-2, int(cx)-1, int(cx), int(cx)+1,int(cx)+2]:
            #     for j in [int(cy)-2,int(cy)-2, int(cy), int(cy)+1,int(cy)+2]: 
            for i in  range(int(cx+lower),int(cx+upper)+1):  
                for j in range(int(cy+lower),int(cy+upper)+1): 
                    if 0<=i<nW and 0<=j<nH:
                        pass
                    else:
                        continue 
                    # compute x offset and y offset 
                    x = cx - i 
                    y = cy - j 
                    # if -0.5<=x<=1.5 and -0.5<=y<=1.5:
                    #     pass 
                    # else:
                    #     continue 
                    # if -2.0<=x<=3.0 and -2<=y<=3.0: 
                    #     pass 
                    # else:
                    #     continue 
                    if lower<=x<=upper and lower<=y<=upper:
                        pass 
                    else:
                        continue 

                    if not cold_mask[b,j,i]:
                        noobj_mask[b,j,i] = 0.0 
                    # set positive
                    if not heat_mask[b,j,i]:
                        continue 

                    # prior value 
                    # vx = (x-0.5) * np.cos(ang) + (y-0.5) * np.sin(ang)
                    # vy = -(x-0.5) * np.sin(ang) + (y-0.5)*np.cos(ang)
                    # v = min(abs(vx)/h,abs(vy)/w)
                    v = min(abs(x-0.5), abs(y-0.5))
                    position = (b,i,j)
                    if position not in prior:
                        prior[position] = v 
                    elif prior[position]>v:
                        prior[position] = v 
                    else:
                        # do nothing
                        continue 
                    # set positive 
                    obj_mask[b,j,i] = 1.0  
                    noobj_mask[b,j,i] = 0.0  
                    # set offset 
                    tx[b,j,i] = x 
                    ty[b,j,i] = y 
                    # set short side length 
                    tw[b,j,i] = np.log(w+1e-16)
                    # set long side length
                    th[b,j,i] = np.log(h+1e-16)
                    # set rotate angles 
                    ta[b,j,i] = box[6]
        # convert to torch style
        obj_mask = torch.from_numpy(obj_mask).bool().to(device)
        noobj_mask = torch.from_numpy(noobj_mask).bool().to(device)
        tx = torch.from_numpy(tx).float().to(device) 
        ty = torch.from_numpy(ty).float().to(device) 
        tw = torch.from_numpy(tw).float().to(device) 
        th = torch.from_numpy(th).float().to(device) 
        ta = torch.from_numpy(ta).float().to(device)  
        return obj_mask, noobj_mask, tx, ty, tw, th, ta, obj_mask.float() 
 

class HeatmapHead(nn.Module):
    '''
    only have cls probility for each class
    '''
    def __init__(self, cls_num: int = 2, scale_factor: float = 8.0, no_obj_scale: float=1.0):
        super().__init__()
        self._cls_num = cls_num
        self._scale_factor = scale_factor 
        self.bce_loss = nn.BCELoss() 
        self.no_obj_scale = no_obj_scale
        # tensor board loss
        self.loss_log = {}

    @property
    def channel(self):
        '''
        expect channel number
        '''
        return self._cls_num
 
    def forward(self, predict, targets=None):
        ''' 
        if targets it None:
            just show heatmap 
        otherwise:
            compute loss
        '''
        # device = predict.device
        # nB, _, nH, nW = predict.shape 
        cls_num = self._cls_num
        # predcit score active 
        predict= predict.permute(0,2,3,1).contiguous() 
        conf = torch.sigmoid(predict) 
        heatmap, _ = torch.max(conf, dim=-1) 
        if targets is None:
            return 0, heatmap 
        targets[:, 2:6] /= self._scale_factor
        loss = 0
        for cls_id in range(cls_num):
            # select certain class 
            tmp = targets[targets[:, 1] == cls_id]
            if len(tmp) == 0:
                continue 
            conf = torch.sigmoid(predict[..., cls_id])
            # build targets
            # obj_mask, noobj_mask, tconf = self.build_targets(predict, tmp) 
            obj_mask, noobj_mask, tconf = build_heatmap(predict, tmp) 
            if torch.sum(obj_mask).item()==0:
                # do nothing
                continue 
            loss_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
            loss_noobj = self.no_obj_scale * self.bce_loss(conf[noobj_mask], tconf[noobj_mask])  
            loss_conf = loss_obj + loss_noobj  
            self.loss_log['loss/heatmap'] = loss_conf.item() 
            self.loss_log['loss/total'] = loss_conf.item() 
            loss += loss_conf
        targets[:, 2:6] *= self._scale_factor 
        return loss, heatmap

     
    # def build_targets(self, predict, targets):
    #     nB, nH, nW, _ = predict.shape 
    #     device = predict.device 
    #     objs = []
    #     noobjs = []
    #     confs = [] 
    #     for b in range(nB):
    #         tmp = targets[targets[:,0]==b]  
    #         obj, noobj, conf = self.render_map(predict, tmp)
    #         objs.append(obj)
    #         noobjs.append(noobj)
    #         confs.append(conf)
    #     obj_mask = torch.stack(objs)
    #     noobj_mask = torch.stack(noobjs)
    #     tconf = torch.stack(confs) 
    #     return obj_mask, noobj_mask, tconf 

    # def render_map(self, predict, targets=None):
    #     '''
    #     render binary map for a single image 
    #     ''' 
    #     nB, nH, nW, _ = predict.shape 
    #     device = predict.device 
    #     obj_mask = torch.FloatTensor(nH,nW).to(device).fill_(0)
    #     noobj_mask = torch.FloatTensor(nH,nW).to(device).fill_(1)
    #     if targets is None or len(targets)==0:
    #         return  obj_mask.bool(), noobj_mask.bool(), obj_mask.float()  
    #     # grid points 
    #     grid_x = torch.arange(nW).repeat(nH, 1).view(
    #         [nH, nW]
    #     ).to(torch.float).to(device) + 0.5 
    #     grid_y = torch.arange(nH).repeat(nW, 1).t().view(
    #         [nH, nW]
    #     ).to(torch.float).to(device) + 0.5 
    #     for box in targets:
    #         cx = box[2]
    #         cy = box[3]
    #         w = box[4]
    #         h = box[5]
    #         ang = box[6]
    #         # center and offset in ang 
    #         x = cx * torch.cos(ang) + cy*torch.sin(ang)
    #         y = -cx*torch.sin(ang) + cy*torch.cos(ang) 
    #         x1, x2 = x-1.2*h/2, x+1.2*h/2 
    #         y1, y2 = y-1.2*w/2, y+1.2*w/2 
    #         x3, x4 = x-0.8*h/2, x+0.8*h/2
    #         y3, y4 = y-0.8*w/2, y+0.8*w/2 
    #         # get value map 
    #         vx = grid_x * torch.cos(ang) + grid_y*torch.sin(ang)
    #         vy = -grid_x * torch.sin(ang) + grid_y*torch.cos(ang)  
    #         mask = (x3<=vx) * (x4>=vx) * (y3<=vy) * (y4>=vy)
    #         obj_mask[mask] = 1.0   
    #         mask = (x1<=vx) * (x2>=vx) * (y1<=vy) * (y2>=vy)
    #         noobj_mask[mask] = 0.0 
    #     return obj_mask.bool(), noobj_mask.bool(), obj_mask.float()  