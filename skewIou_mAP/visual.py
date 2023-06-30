'''
make the predict and ground truth visualable
create date: 2022-05-08-20:01
'''
import os
import sys
import cv2
import shutil
import numpy as np

from tqdm import tqdm, trange
from glob import glob

# candidates colors
colors = [
    (0, 255, 255),
    (255, 255, 0),
]

color_maps = {}  # default color for each cls_name


def load_gt_boxes(txt):
    '''
    load boxes from ground truth txt file
    '''
    with open(txt, 'r') as fp:
        lines = fp.readlines()
    boxes = []
    for line in lines:
        tmp = line.split()
        cls_name = tmp[0]
        points = tmp[1:]
        box = [cls_name] + [int(float(t)) for t in points]
        boxes.append(box)
    return boxes


def load_pt_boxes(txt):
    '''
    load boxes from ground truth txt file
    '''
    with open(txt, 'r') as fp:
        lines = fp.readlines()
    boxes = []
    for line in lines:
        tmp = line.split()
        cls_name = tmp[0]
        score = tmp[1]
        points = tmp[2:]
        box = [cls_name] + [float(score)] + [int(float(t)) for t in points]
        boxes.append(box)
    return boxes


def draw_boxes(name, gt_boxes, pt_boxes):
    # step 1: read img
    img1 = cv2.imread(name)
    img2 = img1.copy()
    # step 2: draw ground truth on first image
    for box in gt_boxes:
        # step 2.1 get box color
        cls_name = box[0]
        if cls_name not in color_maps:
            color_maps[cls_name] = len(color_maps)
        color_id = color_maps[cls_name]
        color = colors[color_id]
        # step 2.2 draw the points
        points = box[1:]
        points = np.array(points).reshape(4, 2)
        img1 = cv2.polylines(img1, [points], True, color, 3)
        # step 2.3 show  the cls name
        x, y = points[0]
        x, y = int(x), int(y)
        img1 = cv2.putText(img1, cls_name, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, color)
    # step 3: draw predict result on second image
    for box in pt_boxes:
        # step 2.1 get box color
        cls_name = box[0]
        if cls_name not in color_maps:
            color_maps[cls_name] = len(color_maps)
        color_id = color_maps[cls_name]
        color = colors[color_id]
        # step 2.2 draw the points
        points = box[2:]
        points = np.array(points).reshape(4, 2)
        img2 = cv2.polylines(img2, [points], True, color, 3)
        # step 2.3 show  the predict score
        score = float(box[1])
        x, y = points[0]
        x, y = int(x), int(y)
        img2 = cv2.putText(img2, '%.2f' % score, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, color)
    # concate and return
    img = cv2.vconcat([img1, img2])
    return img


def visual():
    root = os.path.join(os.getcwd(), 'input')
    # src paths
    img_path = os.path.join(root, 'images-optional')   # dictionary of images
    # dictionary of ground truth
    gt_path = os.path.join(root, 'ground-truth')
    # dictionary of detection results
    pt_path = os.path.join(root, 'detection-results')
    # dst paths
    dst_path = os.path.join(os.getcwd(), 'result')
    dst_default = os.path.join(dst_path, 'all')
    # remove dst path and create empty folder
    if os.path.exists(os.path.join('.', dst_path)):
        shutil.rmtree(os.path.join('.', dst_path))
    os.mkdir(dst_path)
    # os.mkdir(dst_default)
    # main processor
    all_imgs = glob(os.path.join(img_path, '*png')) + \
        glob(os.path.join(img_path, '*jpg')) + \
        glob(os.path.join(img_path, '*JPG'))
    for img in tqdm(all_imgs):
        _, short = os.path.split(img)
        short = short.split('.', 1)[0]
        gt = os.path.join(gt_path, short+'.txt')
        pt = os.path.join(pt_path, short+'.txt')
        # load ground truth boxes
        gt_boxes = load_gt_boxes(gt)
        # load predcit boxes
        pt_boxes = load_pt_boxes(pt)
        # draw all boxes on the image
        picture = draw_boxes(img, gt_boxes, pt_boxes)
        # save the boxes in given dictionary
        # cv2.imwrite(os.path.join(dst_default, short+'.png'), picture)
        # copy image to each distinct folder
        candidate_cls = []
        for box in gt_boxes:
            cls_name = box[0]
            candidate_cls.append(cls_name)
        for cls_name in set(candidate_cls):
            cls_folder = os.path.join(dst_path, cls_name)
            if not os.path.exists(cls_folder):
                os.mkdir(cls_folder)
            cv2.imwrite(os.path.join(cls_folder, short+'.png'), picture)
    print('Done')


if __name__ == '__main__':
    visual()
