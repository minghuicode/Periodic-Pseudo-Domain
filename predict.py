'''
predict all image from single folder
''' 
import os
import sys 
import cv2
import time  
import shutil
import numpy as np
from xml.dom import minidom
from collections import defaultdict

import torch
from glob import glob
from tqdm import tqdm, trange
from shapely.geometry import Polygon

from model import TwinRes 

device = 'cuda'

class Arg:
    def __init__(self):
        self.backbone = 'resnet50'
        self.stride = 1
        return 

def load_model():
    '''
    load model and params
    '''
    dataset='Munich'
    backbone='resnet50'
    # step0: check params 
    assert backbone in {'resnet18', 'resnet50', 'resnet101'}
    model_name = backbone + dataset
    # step1: model define
    args = Arg() 
    model = TwinRes(1, args).to(device)
    # step2: load weights
    model.load_state_dict(torch.load(os.path.join('checkpoints', model_name+'1_best.pth'))) 
    # model.load_state_dict(torch.load(os.path.join('checkpoints', model_name+'.pth'))) 
    # step3: set eval time
    model.eval()
    return model


def load_image(jpg, resize=None): 
    '''
    load a single image and convert it to torch style
    所有图像都给他放大一倍
    '''
    def near32(w):
        '''
        get the integer near w
        '''
        p, q = divmod(w, 32)
        if q == 0:
            return 32*p
        return 32*p+32
    img = cv2.imread(jpg, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    if resize is not None:
        h = int(resize*height)
        w = int(resize*width)
        img = cv2.resize(img,(w,h))
    img = torch.from_numpy(img).float()
    height, width, _ = img.shape
    # build batch with single image
    h = near32(height)
    w = near32(width)
    imgs = torch.zeros([1, 3, h, w])
    # transfer img to torch style
    imgs[0, 0, :height, :width] = img[:, :, 0] / 255.0  # red
    imgs[0, 1, :height, :width] = img[:, :, 1] / 255.0  # green
    imgs[0, 2, :height, :width] = img[:, :, 2] / 255.0  # blue
    # copy data to devcie
    imgs = imgs.to(device)
    return imgs


def post_process(boxes, conf_thres=0.5, nms_thres=0.3):
    '''
    post process for predict boxes
    step1: threshold and sort
    step2: to points style
    step3: skew-iou based nms
    '''
    # conf_thres = 0.95   
    nms_thres = 0.3
    # step1: threshold and sort it
    boxes = boxes[boxes[:, 0] >= conf_thres]
    score = boxes[:, 0]
    boxes = boxes[(-score).argsort()]
    # if empty, do nothing
    if len(boxes) == 0:
        return np.zeros([0,10])
    # try gpu nms 
    if device == 'cuda':
        gpu_nms = True 
    else:
        gpu_nms = False  
    try:
        import math 
        from utils.r_nms import r_nms 
    except ImportError:
        gpu_nms = False  
    if gpu_nms:
        # select all object from each category 
        rv = np.zeros([0,7])
        while len(boxes)>0:
            cls_id = boxes[0,1]
            tmp = boxes[boxes[:,1]==cls_id]
            boxes = boxes[boxes[:,1]!=cls_id]
            score = tmp[:, 0]
            cx = tmp[:, 2]
            cy = tmp[:, 3]
            w = tmp[:, 4]
            h = tmp[:, 5]
            ang = tmp[:, 6]
            n = len(tmp)
            dets = np.zeros([n,6])
            dets[:, 0] = cx 
            dets[:, 1] = cy 
            dets[:, 2] = w 
            dets[:, 3] = h  
            dets[:, 4] = ang 
            dets[:, 5] = score 
            dets = torch.from_numpy(dets).to(device).float()
            valid = r_nms(dets, nms_thres)
            valid = valid.detach().cpu().numpy() 
            tmp = tmp[valid]
            rv = np.concatenate([rv,tmp],axis=0) 
        # to point style 
        boxes = rv  
        cx = boxes[:, 2]
        cy = boxes[:, 3]
        w = boxes[:, 4]
        h = boxes[:, 5]
        ang = boxes[:, 6]
        n = len(boxes)
        points = np.zeros([n, 10])
        points[:, 0] = boxes[:, 0] # score
        points[:, 1] = boxes[:, 1] # cls_id
        i = 2
        for p,q in [(1,1),(1,-1),(-1,-1),(-1,1)]:
            x = cx + p * h/2 * np.cos(ang) - q * w/2*np.sin(ang)
            y = cy + p * h/2 * np.sin(ang) + q * w/2*np.cos(ang)
            points[:, i] = x
            i += 1
            points[:, i] = y
            i += 1
        return points 
    # step2: to points style
    cx = boxes[:, 2]
    cy = boxes[:, 3]
    w = boxes[:, 4]
    h = boxes[:, 5]
    ang = boxes[:, 6]
    n = len(boxes)
    points = np.zeros([n, 10])
    points[:, 0] = boxes[:, 0] # score
    points[:, 1] = boxes[:, 1] # cls_id
    i = 2
    for p,q in [(1,1),(1,-1),(-1,-1),(-1,1)]:
        x = cx + p * h/2 * np.cos(ang) - q * w/2*np.sin(ang)
        y = cy + p * h/2 * np.sin(ang) + q * w/2*np.cos(ang)
        points[:, i] = x
        i += 1
        points[:, i] = y
        i += 1
    # step3: utils skew-iou based nms
    rv = []
    while len(points)>0:
        box = points[0]
        cls_id = box[1]
        # select certain cls
        tmp = points[points[:, 1]==cls_id]
        points = points[points[:, 1]!=cls_id]
        while len(tmp)>0:
            box = tmp[0]
            rv.append(np.array(box))
            valid = skew_iou(box, tmp) < nms_thres
            tmp = tmp[valid]
    rv = np.array(rv)
    score = rv[:,0]
    rv = rv[(-score).argsort()]
    return rv

def draw_boxes(name, boxes): 
    # candidates colors 
    color =  (255, 255, 0)  
    color2 = (75, 139, 147)
    # step 1: read img
    img = cv2.imread(name)
    # step 2: draw ground truth on first image
    for box in boxes: 
        # step 2.2 draw the points
        points = [int(t) for t in box[2:]]
        points = np.array(points).reshape(4, 2)
        img = cv2.polylines(img, [points], True, color, 3) 
        # draw a center line
        cx = sum(points[:,0]) // 4 
        cy = sum(points[:,1]) // 4 
        # 第一条线的方向
        p, q = points[1][0]-points[-1][0], points[1][1]-points[-1][1] 
        img = cv2.line(img, (cx-3,cy), (cx+3,cy), color2, 2)
        # 第二条线的方向 
        img = cv2.line(img, (cx,cy-3), (cx,cy+3), color2, 2) 
    return img 


def folder_predict(src, dst):
    model = load_model() 
    if not os.path.exists(dst):
        os.mkdir(dst)
    # resize = 2.0 # 所有图像放大一倍
    resize = 1.8
    with torch.no_grad():
        for name in tqdm(glob(os.path.join(src,'*jpg'))):
            # predict
            img = load_image(name, resize)
            _, boxes = model(img) 
            boxes = post_process(boxes[0])
            boxes[:, 2:] /= resize
            # draw boxes
            img = draw_boxes(name, boxes)
            # save img 
            _, short = os.path.split(name) 
            save_name = os.path.join(dst, short)
            cv2.imwrite(save_name, img)
    return 

if __name__ == "__main__":
    folder_predict('../movie-detector/jpgs','../movie-detector/predict')

