''' 
test model accuracy on val or test part 
dataset:
    - DOTA-v1.0 
    - HRSC2016
    - UCAS-AOD
metrics:
    - VOC 2012
    - CNN FPS
    - post process speed
    - entire image process speed
create date: 2022-05-20 14:57
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

from model import TwinModel, TwinRes




import argparse


parser = argparse.ArgumentParser(description='Training args')  
parser.add_argument('-d','--dataset', type=str, default='Ucas', help='dataset, (Ucas, Dota, Hrsc)')
parser.add_argument('-b','--backbone', type=str, default='resnet50', help='backbone, (resnet18, resnet50, resnet101)')

device = 'cuda'

def load_model(dataset='Ucas',backbone='resnet50'):
    '''
    load model and params
    '''
    # step0: check params
    assert dataset in {'Ucas', 'Hrsc', 'Dota'} 
    assert backbone in {'resnet18', 'resnet50', 'resnet101'}
    model_name = backbone + dataset
    # step1: model define
    if dataset=='Ucas':
        cls_num = 2
    elif dataset=='Hrsc':
        cls_num = 1 
    elif dataset=='Dota':
        cls_num = 15
    model = TwinRes(cls_num,backbone).to(device)
    # step2: load weights
    model.load_state_dict(torch.load(os.path.join('checkpoints', model_name+'_best.pth'))) 
    # model.load_state_dict(torch.load(os.path.join('checkpoints', model_name+'.pth'))) 
    # step3: set eval time
    model.eval()
    return model


def load_image(jpg, resize=None): 
    '''
    load a single image and convert it to torch style
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

def load_batch(jpg_list):
    '''
    load image batch
    and set to no more than 1600x960 
    '''
    n = len(jpg_list) 
    rv = torch.zeros([n, 3, 960, 1600]) 
    resize_ratios = [] 
    for i, jpg in enumerate(jpg_list): 
        img = cv2.imread(jpg, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape 
        r1 = 1600/width
        r2 = 960/height
        if r2>r1:
            # target size
            th = int(height*r1)
            tw = 1600
            r = r1
        else:
            th = 960
            tw = int(width*r2)
            r = r2
        if r>1.0:
            r = 1.0  
            tw, th = width, height 
        else:
           img = cv2.resize(img, (tw,th))  
        img = torch.from_numpy(img).float() 
        rv[i, 0, :th, :tw] = img[:,:,0]/255.0 # red
        rv[i, 1, :th, :tw] = img[:,:,1]/255.0 # green
        rv[i, 2, :th, :tw] = img[:,:,2]/255.0 # blue
        resize_ratios.append(r)
    imgs = rv.to(device)
    return imgs, resize_ratios



# def load_image_hrsc(jpg):
#     '''
#     load a single image and convert it to torch style
#     '''
#     img = cv2.imread(jpg, cv2.IMREAD_UNCHANGED)
#     height, width, _ = img.shape
#     img = cv2.resize(img, (800, 512))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = torch.from_numpy(img).float()
#     # build batch with single image
#     imgs = torch.zeros([1, 3, 512, 800])
#     # transfer img to torch style
#     imgs[0, 0, ...] = img[:, :, 0] / 255.0  # red
#     imgs[0, 1, ...] = img[:, :, 1] / 255.0  # green
#     imgs[0, 2, ...] = img[:, :, 2] / 255.0  # blue
#     # copy data to devcie
#     imgs = imgs.to(device)
#     return imgs, width/800, height/512

def load_image_hrsc(jpg):
    '''
    load a single image and convert it to torch style
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
    return imgs, 1.0, 1.0

def save_txt(boxes, txt, dataset='Ucas'):
    # step0: check params
    assert dataset in {'Ucas', 'Hrsc', 'Dota'} 
    if dataset == 'Ucas':
        d = {0: 'car', 1: 'plane'}
    elif dataset == 'Hrsc':
        d = {0: 'ship'}
    elif dataset == 'Dota':
        d = [ 
            'plane',
            'baseball-diamond',
            'bridge',
            'ground-track-field',
            'small-vehicle',
            'large-vehicle',
            'ship',
            'tennis-court',
            'basketball-court',
            'storage-tank',
            'soccer-ball-field',
            'roundabout',
            'harbor', 
            'swimming-pool',
            'helicopter'
        ] 
    ss = ''
    for box in boxes:
        score, cls_id = box[0], box[1]
        cls_id = int(cls_id)
        cls_name = d[cls_id]
        # obb = [max(t,0) for t in box[2:]]
        obb = [int(t) for t in box[2:]]
        ss += ' '.join([cls_name]+[str(t) for t in [float(score)]+obb]) + '\n'
    with open(txt, 'w') as fp:
        fp.writelines(ss)

def read_boxes(txt, cls_id):
    '''
    read boxes from a txt file
    using UCAS_AOD style
    '''
    d = {0: 'car', 1: 'plane'}
    cls_name = d[cls_id]
    with open(txt, 'r') as fp:
        lines = fp.readlines()
    boxes = []
    for line in lines:
        # 13 individual number
        nums = [float(t) for t in line.split()]
        # points
        box = [cls_name] + [int(t) for t in nums[:8]]
        boxes.append(box)
    return boxes 

def hbb_iou(xmin, xmax, ymin, ymax, x1, x2, y1, y2):
    xx = min(x2, xmax) - max(x1,xmin)
    yy = min(y2, ymax) - max(y1, ymin) 
    inter = max(xx,0) * max(yy,0)
    union = (xmax-xmin) * (ymax + ymin) + (x2-x1) * (y2-y1) - inter 
    iou = inter / (union + 1e-16)
    return iou 

def skew_iou(box, boxes):
    '''
    handle box
    target boxes
    '''
    rv = []
    points = np.array(box[2:]).reshape(4,2)
    xmin, xmax = min(points[:,0]), max(points[:,0])
    ymin, ymax = min(points[:,1]), max(points[:,1])
    handle_polygon = Polygon(points)
    for box in boxes:
        points = np.array(box[2:]).reshape(4,2) 
        # # early stop 
        # x1, x2 = min(points[:,0]), max(points[:,0])
        # y1, y2 = min(points[:,1]), max(points[:,1])
        # if hbb_iou(xmin, xmax, ymin, ymax, x1, x2, y1, y2)<0.1:
        #     # seem as intersection 
        #     rv.append(0.0)
        #     continue 
        target_polygon = Polygon(points)
        inter = handle_polygon.intersection(target_polygon).area
        union = handle_polygon.area + target_polygon.area - inter
        iou = inter / (union+1e-16)
        rv.append(iou)
    rv = np.array(rv)
    return rv

def pointStyle(boxes):
    '''
    convert obb boxes to points style
    '''
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


def patch_process(boxes, conf_thres=0.5, nms_thres=0.3):
    '''
    patch process for predict boxes
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
        return []
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
    # select all object from each category 
    assert(gpu_nms)
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
    return rv  

    
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


def test_ucas(backbone='resnet50'):
    '''
    val mode: validation part for 302 images
    test mode: test part for 453 images
    train mode: training part for 755 images
    '''
    dst_path = 'skewIou_mAP/input'
    # step1: get all image list
    root = '/home/buaa/dataset/ucas'
    test_txt = os.path.join(root, 'test.txt')
    # test_txt = os.path.join(root, 'val.txt')
    with open(test_txt, 'r') as fp:
        lines = fp.readlines()
    test_list = []
    for line in lines:
        jpg, txt, cls_id = line.split()
        cls_id = int(cls_id)
        test_list.append([jpg, txt, cls_id])
    # step2: write all ground truth to given path
    gt_path = os.path.join(dst_path, 'ground-truth')
    img_path = os.path.join(dst_path, 'images-optional')
    result_path = os.path.join(dst_path, 'detection-results')
    # clear path and make new dictionary
    if os.path.exists(os.path.join('.', dst_path)):
        shutil.rmtree(os.path.join('.', dst_path))
    os.mkdir(dst_path)
    os.mkdir(gt_path)
    os.mkdir(img_path)
    os.mkdir(result_path)
    print("step 1/2: writting ground truth...")
    # write all ground truth
    new_id = 0
    for jpg, txt, cls_id in tqdm(test_list):
        suffix = jpg.split('.')[-1]
        new_id += 1
        short = '0000' + str(new_id)
        short = short[-4:]
        # 1. copy images
        # shutil.copy(jpg, img_path)
        shutil.copy(jpg, os.path.join(img_path, short+'.'+suffix))
        # 2. read boxes
        boxes = read_boxes(txt, cls_id)
        ss = ''
        for box in boxes:
            ss += ' '.join([str(t) for t in box]) + '\n'
        # 3. save gt to txt
        gt_txt = os.path.join(gt_path, short+'.txt')
        with open(gt_txt, 'w') as fp:
            fp.writelines(ss)
    # step3: forward and test each image
    print('step 2/2: writting predict result...')
    model = load_model('Ucas',backbone)
    # time log 
    time_start = time.time()  
    imread_time = 0.0
    cnn_time = 0.0 
    nms_time = 0.0
    with torch.no_grad():
        new_id = 0
        for jpg, txt, cls_id in tqdm(test_list):
            # load image data
            t1 = time.time() 
            imgs = load_image(jpg)
            # model forward
            t2 = time.time() 
            _, boxes = model(imgs)
            # get boxes
            t3 = time.time() 
            boxes = post_process(boxes[0])
            # save boxes to the txt file
            # _, short = os.path.split(txt)
            t4 = time.time() 
            new_id += 1
            short = '0000' + str(new_id)
            short = short[-4:]
            result_txt = os.path.join(result_path, short+'.txt')
            save_txt(boxes, result_txt, 'Ucas')
            # time log 
            imread_time += t2-t1 
            cnn_time += t3-t2 
            nms_time += t4-t3 
    total_time = time.time() - time_start
    # report the speed 
    cnn_fps = len(test_list) / cnn_time
    nms_fps = len(test_list) / nms_time
    total_fps = len(test_list) / total_time  
    print("--"*12 + 'Speed Board' + '--'*12)
    print("network FPS: {:.2f} per second".format(cnn_fps))
    print("NMS FPS: {:.2f} per second".format(nms_fps))
    print("total img load time: {:.2f}".format(imread_time))
    print("total img forward time: {:.2f}".format(cnn_time))
    print("total nms time: {:.2f}".format(nms_time))
    print("total time spend: {:.2f}".format(total_time))
    print("total image number: {}".format(len(test_list)))
    print("total Process Speed: {:.2f} Image per second ".format(total_fps)) 

 