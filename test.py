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

from model import TwinRes, OneStage
from config.utils import RotateConfig



import argparse


device = 'cuda'


def parseArg():
    parser = argparse.ArgumentParser(description='Training args')
    parser.add_argument('-d','--dataset', type=str, default=None, help='dataset, (Ucas, Munich, Hrsc)')
    parser.add_argument('-b','--backbone', type=str, default=None, help='backbone, (resnet18, resnet50, resnet101)')
    parser.add_argument('-n','--model_name', type=str, default=None, help='weight model name')
    parser.add_argument('-s','--stride', type=int, default=None, help='the max center offset stride of head predict')
    parser.add_argument('-c','--config', type=str, default='config/default.json', help='config file')
   # parser args
    args = parser.parse_args()
    # check params
    cfg = RotateConfig(args.config)
    cfg.modify(args)
    return cfg

def load_model(cfg):
    '''
    load model and params
    '''
    dataset = cfg.get('dataset')
    model_name = cfg.get('model_name')
    # step1: model define
    if dataset=='Ucas':
        cls_num = 2
    elif dataset=='Hrsc':
        cls_num = 1
    elif dataset=='Dota':
        cls_num = 15
    elif dataset=='Munich':
        cls_num = 1
    # model = TwinRes(cls_num, cfg).to(device)
    model = OneStage(cls_num, cfg).to(device)
    # model = TwinRes(cls_num, args.backbone, stride=args.stride).to(device)
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
    assert dataset in {'Ucas', 'Hrsc', 'Munich'}
    if dataset == 'Ucas':
        d = {0: 'car', 1: 'plane'}
    elif dataset == 'Hrsc':
        d = {0: 'ship'}
    elif dataset == 'Munich':
        d = [
            'car'
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


def test_ucas(cfg):
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
    pixels = 0
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
    model = load_model(cfg)
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
            _, _, h, w = imgs.shape
            pixels += h*w
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
    print("total pixels: ", pixels)
    pixels /= 1000*1000
    v = cnn_time *1000.0 / pixels
    print("ms per millsion pixel: {:.2f}".format(v))
    print("total nms time: {:.2f}".format(nms_time))
    print("total time spend: {:.2f}".format(total_time))
    print("total image number: {}".format(len(test_list)))
    print("total fps: {:.2f}".format(len(test_list)/total_time))
    print("total Process Speed: {:.2f} Image per second ".format(total_fps))

def xywhp2points(xcenter,ycenter,swidth,sheight,angle):
    ang = 0 - angle
    points = []
    # # 四个点的坐标
    # # p1
    # points.append(xcenter+swidth*cos(angle)+sheight*cos(angle-90.0))
    # points.append(ycenter+swidth*sin(angle)+sheight*sin(angle-90.0))
    # # p2
    # points.append(xcenter+swidth*cos(angle)+sheight*cos(angle+90.0))
    # points.append(ycenter+swidth*sin(angle)+sheight*sin(angle+90.0))
    # # p4
    # points.append(xcenter+swidth*cos(angle+180.0)+sheight*cos(angle+90.0))
    # points.append(ycenter+swidth*sin(angle+180.0)+sheight*sin(angle+90.0))
    # # p3
    # points.append(xcenter+swidth*cos(angle+180.0)+sheight*cos(angle-90.0))
    # points.append(ycenter+swidth*sin(angle+180.0)+sheight*sin(angle-90.0))
    for p,q in [(1,1),(1,-1),(-1,-1),(-1,1)]:
        x = xcenter + p * swidth * np.cos(ang*np.pi/180) - q * sheight*np.sin(ang*np.pi/180)
        y = ycenter + p * swidth * np.sin(ang*np.pi/180) + q * sheight*np.cos(ang*np.pi/180)
        points.append(int(x))
        points.append(int(y))
    return points



def test_munich(cfg):
    '''
    val mode: validation part for 302 images
    test mode: test part for 453 images
    train mode: training part for 755 images
    '''
    dst_path = 'skewIou_mAP/input'
    # step1: get all image list
    root = '/home/buaa/Data/TGARS2021/dataset/dlr'
    # test_txt = os.path.join(root, 'val.txt')
    test_list = [
        '2012-04-26-Muenchen-Tunnel_4K0G0040',
        '2012-04-26-Muenchen-Tunnel_4K0G0080',
        '2012-04-26-Muenchen-Tunnel_4K0G0030',
        '2012-04-26-Muenchen-Tunnel_4K0G0051',
        '2012-04-26-Muenchen-Tunnel_4K0G0010'
    ]
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
    for t in tqdm(test_list):
        img_name = os.path.join(root, t+'.JPG')
        # 1. copy images
        # shutil.copy(jpg, img_path)
        shutil.copy(img_name, os.path.join(img_path, t+'.JPG'))
        # 2. read boxes
        boxes = []
        for suffix in ['_pkw.samp','_bus.samp','_truck.samp']:
            note_name = os.path.join(root,t+suffix)
            # load labels
            if not os.path.exists(note_name):
                continue
            with open(note_name,'r') as fp:
                lines = fp.readlines()
            for line in lines:
                if line[0] in {'#','@'}:
                    continue
                _, __, xcenter, ycenter, swidth, sheight, angle = [
                    float(s) for s in line.split()]
                points = xywhp2points(xcenter, ycenter, swidth, sheight, angle)
                # convert boxes to point style
                # boxes.append(['car  1.00'] + points)
                boxes.append(['car'] + points)
        ss = ''
        for box in boxes:
            ss += ' '.join([str(t) for t in box]) + '\n'
        # 3. save gt to txt
        gt_txt = os.path.join(gt_path, t+'.txt')
        with open(gt_txt, 'w') as fp:
            fp.writelines(ss)
    # step3: forward and test each image
    print('step 2/2: writting predict result...')
    model = load_model(cfg)
    # time log
    time_start = time.time()
    imread_time = 0.0
    cnn_time = 0.0
    nms_time = 0.0
    pixels = 0
    with torch.no_grad():
        new_id = 0
        for t in tqdm(test_list):
            # load image data
            img_name = os.path.join(root, t+'.JPG')
            t1 = time.time()
            imgs = load_image(img_name)
            _, _, h, w = imgs.shape
            pixels += h*w
            # model forward
            t2 = time.time()
            _, boxes = model(imgs)
            # get boxes
            t3 = time.time()
            boxes = post_process(boxes[0])
            # save boxes to the txt file
            # _, short = os.path.split(txt)
            t4 = time.time()
            result_txt = os.path.join(result_path, t+'.txt')
            save_txt(boxes, result_txt, 'Munich')
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
    pixels /= 1000*1000
    v = cnn_time *1000.0 / pixels
    print("ms per millsion pixel: {:.2f}".format(v))
    print("total nms time: {:.2f}".format(nms_time))
    print("total time spend: {:.2f}".format(total_time))
    print("total image number: {}".format(len(test_list)))
    print("total fps: {:.2f}".format(len(test_list)/total_time))
    print("total Process Speed: {:.2f} Image per second ".format(total_fps))

def global_nms(boxes, nms_thres=0.3):
    '''
    skew-iou based nms
    '''
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
    assert(gpu_nms)
    boxes = np.array(boxes)
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
    # return as point style
    score = rv[:,0]
    rv = rv[(-score).argsort()]
    points = pointStyle(rv)
    return points

def read_xml(xml_name):
    # open xml doc
    file = minidom.parse(xml_name)
    # get the objects number
    objects = file.getElementsByTagName('HRSC_Object')
    n = len(objects)
    obj_list = []
    for i in range(n):
        tmp = objects[i]
        # class id
        cls_name = tmp.getElementsByTagName('Class_ID')[0].childNodes[0].data
        # HBB: horizontal bounding box
        xmin = tmp.getElementsByTagName('box_xmin')[0].childNodes[0].data
        ymin = tmp.getElementsByTagName('box_ymin')[0].childNodes[0].data
        xmax = tmp.getElementsByTagName('box_xmax')[0].childNodes[0].data
        ymax = tmp.getElementsByTagName('box_ymax')[0].childNodes[0].data
        # OBB: orient bounding box
        cx = tmp.getElementsByTagName('mbox_cx')[0].childNodes[0].data
        cy = tmp.getElementsByTagName('mbox_cy')[0].childNodes[0].data
        w = tmp.getElementsByTagName('mbox_w')[0].childNodes[0].data
        h = tmp.getElementsByTagName('mbox_h')[0].childNodes[0].data
        ang = tmp.getElementsByTagName('mbox_ang')[0].childNodes[0].data
        # head id
        x = tmp.getElementsByTagName('header_x')[0].childNodes[0].data
        y = tmp.getElementsByTagName('header_y')[0].childNodes[0].data
        # define the 12-dim boxes
        box = [
            cls_name,
            xmin, ymin, xmax, ymax,
            cx, cy, w, h, ang,
            x, y
        ]
        obj_list.append(box)
    return obj_list


def read_boxes_xml(xml):
    cls_name = 'ship'
    obj_list = read_xml(xml)
    # obj to boxes
    boxes = []
    for box in obj_list:
        cx, cy, w, h, ang = [float(t) for t in box[5:10]]
        if h<w:
            h, w = w, h
        obb = []
        # only single class, set cls_id to be 0
        obb.append(cls_name)
        for p,q in [(1,1),(1,-1),(-1,-1),(-1,1)]:
            x = cx + p * h/2 * np.cos(ang) - q * w/2*np.sin(ang)
            y = cy + p * h/2 * np.sin(ang) + q * w/2*np.cos(ang)
            x = int(x)
            y = int(y)
            obb.append(x)
            obb.append(y)
        boxes.append(obb)
    return boxes

def test_hrsc(cfg):
    '''
    val mode: validation part for 302 images
    test mode: test part for 453 images
    train mode: training part for 755 images
    '''
    batch_size = 8
    dst_path = 'skewIou_mAP/input'
    # step1: get all image list
    root = '/home/buaa/Data/TGARS2021/dataset/HRSC2016'
    test_txt = os.path.join(root, 'ImageSets', 'test.txt')
    # test_txt = os.path.join(root, 'val.txt')
    with open(test_txt, 'r') as fp:
        lines = fp.readlines()
    test_list = []
    for line in lines:
        short = line.split()[0]
        jpg = os.path.join(root, 'Test', 'AllImages', short+'.jpg')
        xml = os.path.join(root, 'Test', 'Annotations', short+'.xml')
        if not os.path.exists(jpg):
            # print(" Not Found: " + jpg)
            continue
        test_list.append([jpg, xml])
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
    for jpg, xml in tqdm(test_list):
        suffix = jpg.split('.')[-1]
        new_id += 1
        short = '0000' + str(new_id)
        short = short[-4:]
        # 1. copy images
        # shutil.copy(jpg, img_path)
        shutil.copy(jpg, os.path.join(img_path, short+'.'+suffix))
        # 2. read boxes
        boxes = read_boxes_xml(xml)
        ss = ''
        for box in boxes:
            ss += ' '.join([str(t) for t in box]) + '\n'
        # 3. save gt to txt
        gt_txt = os.path.join(gt_path, short+'.txt')
        with open(gt_txt, 'w') as fp:
            fp.writelines(ss)
    # step3: forward and test each image
    print('step 2/2: writting predict result...')
    model = load_model(cfg)
    # time log
    time_start = time.time()
    imread_time = 0.0
    cnn_time = 0.0
    nms_time = 0.0
    pixels = 0
    with torch.no_grad():
        new_id = 0
        for jpg, xml in tqdm(test_list):
            # load image data
            t1 = time.time()
            # imgs = load_image(jpg, resize=0.6)
            imgs = load_image(jpg, resize=1.0)
            _, _, h, w = imgs.shape
            pixels += h*w
            # model forward
            t2 = time.time()
            _, boxes = model(imgs)
            # get boxes
            t3 = time.time()
            boxes = post_process(boxes[0])
            # boxes[:, 2:] /= 0.6
            # save boxes to the txt file
            t4 = time.time()
            new_id += 1
            short = '0000' + str(new_id)
            short = short[-4:]
            result_txt = os.path.join(result_path, short+'.txt')
            save_txt(boxes, result_txt, 'Hrsc')
        # for start in trange(0, len(test_list), batch_size):
        #     finish = min(start+batch_size, len(test_list))
        #     mini_list = [name[0] for name in test_list[start:finish]]
        #     # load image data
        #     t1 = time.time()
        #     imgs, resize_ratio = load_batch(mini_list)
        #     # model forward
        #     t2 = time.time()
        #     _, boxes = model(imgs)
        #     # get boxes
        #     t3 = time.time()
        #     boxes = [post_process(box) for box in boxes]
        #     # save boxes to the txt file
        #     # _, short = os.path.split(txt)
        #     t4 = time.time()
        #     for box, r in zip(boxes,resize_ratio):
        #         new_id += 1
        #         short = '0000' + str(new_id)
        #         box[:,2:] *= r
        #         short = short[-4:]
        #         result_txt = os.path.join(result_path, short+'.txt')
        #         save_txt(box, result_txt, 'Hrsc')
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
    pixels /= 1000*1000
    v = cnn_time *1000.0 / pixels
    print("ms per millsion pixel: {:.2f}".format(v))
    print("total nms time: {:.2f}".format(nms_time))
    print("total time spend: {:.2f}".format(total_time))
    print("total image number: {}".format(len(test_list)))
    print("total fps: {:.2f}".format(len(test_list)/total_time))
    print("total Process Speed: {:.2f} Image per second ".format(total_fps))

def read_boxes_dota(txt):
    '''
    read boxes from dota annotation
    '''
    with open(txt, 'r') as fp:
        lines = fp.readlines()
    boxes = []
    # first line: image source
    # second line: ground sample distance
    for line in lines[2:]:
        nums = line.split()
        # points
        points = [int(t) for t in nums[:8]]
        cls_name = nums[8]
        box = [cls_name] + points
        boxes.append(box)
    return boxes

def load_image_batch(task_list):
    '''
    DOTA dataset utils
    load batch images and combine the imgs
    '''
    nB = len(task_list)
    imgs = torch.zeros([nB, 3, 512, 512])
    bias = []
    image = None
    pre_name = None
    for ids, (img_name, n, task_id, x1, y1, x2, y2) in enumerate(task_list):
        # load image or use pre image
        if img_name != pre_name:
            image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pre_name = img_name
        # crop the batch
        img = image[y1:y2, x1:x2, :].copy()
        # resize the image
        h, w = y2-y1, x2-x1
        hw = max(y2-y1, x2-x1)
        # assert size in {512, 1024, 2048}
        # assert size > 256
        if hw<=512:
            # no resize
            scale_factor = 1.0
        elif hw<=1024:
            scale_factor = 2.0
        elif hw<=2048:
            scale_factor = 4.0
        else:
            # patch size too large
            raise ValueError("patch size too large ", img_name, x1,y1,x2,y2)
        th = int(h/scale_factor)
        tw = int(w/scale_factor)
        img = cv2.resize(img, dsize=(tw, th))
        img = torch.from_numpy(img).float()
        # transfer img to torch style
        imgs[ids, 0, :th, :tw] = img[:, :, 0]/255.0 # red
        imgs[ids, 1, :th, :tw] = img[:, :, 1]/255.0 # green
        imgs[ids, 2, :th, :tw] = img[:, :, 2]/255.0 # blue
        bias.append([x1, y1, scale_factor])
    # copy data to devcie
    imgs = imgs.to(device)
    return imgs, bias


def test_dota(backbone='resnet50'):
    '''
    val mode: validation part for 458 images
    train mode: train part for 1413 images
    '''
    batch_size = 8
    dst_path = 'skewIou_mAP/input'
    # step1: get all image list
    root = '/home/buaa/Data/TGARS2021/dataset/DOTA-v1.0'
    src_path = os.path.join(root, 'val')
    # src_path = os.path.join(root, 'train')
    img_src = os.path.join(src_path, 'images')
    txt_src = os.path.join(src_path, 'labelTxt')
    img_list = glob(os.path.join(img_src, 'P????.png'))
    # step2: assign tasks & write all ground truth to given path
    gt_path = os.path.join(dst_path, 'ground-truth')
    # img_path = os.path.join(dst_path, 'images-optional')
    result_path = os.path.join(dst_path, 'detection-results')
    # clear path and make new dictionary
    if os.path.exists(os.path.join('.', dst_path)):
        shutil.rmtree(os.path.join('.', dst_path))
    os.mkdir(dst_path)
    os.mkdir(gt_path)
    # os.mkdir(img_path)
    os.mkdir(result_path)
    tasks = []
    print("step 1/2: writting ground truth...")
    # write all ground truth
    for png in tqdm(img_list):
        _, short = os.path.split(png)
        short = short.split('.', 1)[0]
        txt = os.path.join(txt_src, short+'.txt')
        # read boxes
        boxes = read_boxes_dota(txt)
        ss = ''
        for box in boxes:
            ss += ' '.join([str(t) for t in box]) + '\n'
        # save gt to txt
        gt_txt = os.path.join(gt_path, short+'.txt')
        with open(gt_txt, 'w') as fp:
            fp.writelines(ss)
        # read image
        img = cv2.imread(png)
        try:
            height, width, _ = img.shape
        except ValueError:
            # gray-scale image
            print('convert gray-scale image to RGB image: '+png)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            height, width, _ = img.shape
        # compute task numbers
        candidates = [] # candidates patch location
        # for patch_size in [512, 1024, 2048]:
        for patch_size in [512, 2048]:
            step = patch_size // 2
            for x in range(0, width, step):
                for y in range(0, height, step):
                    x2 = min(x+patch_size, width)
                    y2 = min(y+patch_size, height)
                    w = x2-x
                    h = y2-y
                    if w<patch_size/2 or h<patch_size/2:
                        # do nothing
                        continue
                    # try to padding to 512x512
                    if w<patch_size and width>patch_size:
                        x = x2-patch_size
                        w = patch_size
                    if h<patch_size<height:
                        y = y2-patch_size
                        h = patch_size
                    # add patch
                    candidates.append([x,y,x2,y2])
        # add all task to task list
        n = len(candidates)
        mini_id = 0 # id in this task
        for x1, y1, x2, y2 in candidates:
            # task
            task = [png, n, mini_id, x1, y1, x2, y2]
            tasks.append(task)
            mini_id += 1
    print("Processing {} images, which includes {} patches".format(len(img_list), len(tasks)))
    # step3: forward and test each image
    print('step 2/2: writting predict result...')
    model = load_model("Dota",backbone)
    # time log
    time_start = time.time()
    imread_time = 0.0
    cnn_time = 0.0
    patch_nms_time = 0.0
    global_nms_time = 0.0
    nms_time = 0.0
    with torch.no_grad():
        start = 0
        # result cache for patche detection
        cache_result = defaultdict(list)
        detect_over = []
        for start in trange(0, len(tasks), batch_size):
            # detect patch in this batch
            finish = min(start+batch_size, len(tasks))
            mini_task_list = tasks[start:finish]
            # load imgs
            t1 = time.time()
            imgs, bias = load_image_batch(mini_task_list)
            # model forward
            t2 = time.time()
            _, boxes = model(imgs)
            # get boxes
            t3 = time.time()
            boxes = [patch_process(box) for box in boxes]
            # collect detect result to the cache
            t4 = time.time()
            num = len(mini_task_list)
            for i in range(num):
                img_name, n, task_id, x1, y1, x2, y2 = mini_task_list[i]
                # add bias
                box = boxes[i]
                if len(box)>0:
                    x, y, scale_factor = bias[i]
                    box[:, 2::6] *= scale_factor
                    box[:, 2] += x
                    box[:, 3] += y
                    cache_result[img_name] = cache_result[img_name] + list(box)
                # make sure that all image has been detected
                if task_id+1==n:
                    detect_over.append(img_name)
            for img_name in detect_over:
                # global skew nms
                t5 = time.time()
                boxes = global_nms(cache_result[img_name])
                t6 = time.time()
                _, short = os.path.split(img_name)
                short = short.split('.', 1)[0]
                result_txt = os.path.join(result_path, short+'.txt')
                # save to txt
                save_txt(boxes, result_txt, 'Dota')
                # clear cache
                cache_result.pop(img_name)
                # time log
                global_nms_time += t6-t5
            # clear detect list
            detect_over = []
            # time log
            imread_time += t2-t1
            cnn_time += t3-t2
            patch_nms_time += t4-t3
    nms_time = patch_nms_time + global_nms_time
    total_time = time.time() - time_start
    # report the speed
    cnn_fps = len(tasks) / cnn_time
    patch_nms_fps = len(tasks) / patch_nms_time
    total_speed = total_time / len(img_list)
    print("--"*12 + 'Speed Board' + '--'*12)
    print("network FPS: {:.2f} per second".format(cnn_fps))
    print("patchNMS FPS: {:.2f} per second".format(patch_nms_fps))
    print("total img load time: {:.2f}".format(imread_time))
    print("total img forward time: {:.2f}".format(cnn_time))
    print("total nms time: {:.2f}".format(nms_time))
    print("total time spend: {:.2f}".format(total_time))
    print("total image number: {}".format(len(img_list)))
    print("total fps: {:.2f}".format(len(test_list)/total_time))
    print("total Process Speed: {:.2f} second per Image  ".format(total_speed))


def test(cfg):
    dataset = cfg.get('dataset')
    # step0: check params
    assert dataset in {'Ucas', 'Hrsc', 'Munich'}
    if dataset=='Ucas':
        test_ucas(cfg)
    elif dataset=='Hrsc':
        test_hrsc(cfg)
    elif dataset=='Munich':
        test_munich(cfg)


if __name__ == "__main__":
    cfg = parseArg()
    test(cfg)
