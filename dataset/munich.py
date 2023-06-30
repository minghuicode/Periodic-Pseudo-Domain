'''
dataset for Munich
random choice 5 training and 5 test images
data augmentation:
'''
import os
import sys
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from PIL import Image
from xml.dom import minidom
from glob import glob
import json

from .augment import OrientAugment


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size,
                          mode='nearest').squeeze(0)
    return image


def near32(x: int):
    '''
    return nearest 32 value
    '''
    rv = int(round(x/32)*32)
    return max(rv, 32)


def resize_padding(image, targets, size):
    '''
    hold height/width ratio
    resize image, padding zero
    '''
    _, src_h, src_w = image.shape
    size = near32(size)
    rv = torch.zeros_like(image)
    dst_h = dst_w = size
    rv = torch.zeros([3, size, size]).to(device=image.device)
    if src_h > src_w:
        dst_w = near32(size*src_w/src_h)
        s = dst_w / size
        if targets is not None:
            # edit cx
            targets[:, 1] = s*(targets[:, 1] - 0.5) + 0.5
            # edit w
            targets[:, 3] = s*targets[:, 3]
    else:
        dst_h = near32(size*src_h/src_w)
        s = dst_h / size
        if targets is not None:
            # edit cy
            targets[:, 2] = s*(targets[:, 2] - 0.5) + 0.5
            # edit h
            targets[:, 4] = s*targets[:, 4]
    image = F.interpolate(image.unsqueeze(0), size=(dst_h, dst_w),
                          mode='nearest').squeeze(0)
    hstart, wstart = (size-dst_h)//2, (size-dst_w)//2
    rv[:, hstart:hstart+dst_h, wstart:wstart+dst_w] = image
    return rv, targets



"""
Munich dataset
"""
class MunichDataset(Dataset):
    def __init__(self, root='/home/buaa/dataset/dlr', augment=True, multiscale=True):
        super().__init__()
        # 5 image for train, other 5 for test
        #            img name             box num
        # 2012-04-26-Muenchen-Tunnel_4K0G0070 590
        # 2012-04-26-Muenchen-Tunnel_4K0G0090 611
        # 2012-04-26-Muenchen-Tunnel_4K0G0020 337
        # 2012-04-26-Muenchen-Tunnel_4K0G0100 411
        # 2012-04-26-Muenchen-Tunnel_4K0G0060 725
        # 2012-04-26-Muenchen-Tunnel_4K0G0040 193
        # 2012-04-26-Muenchen-Tunnel_4K0G0080 277
        # 2012-04-26-Muenchen-Tunnel_4K0G0030 27
        # 2012-04-26-Muenchen-Tunnel_4K0G0051 247
        # 2012-04-26-Muenchen-Tunnel_4K0G0010 67
        self.root = root
        self.short = [
            '2012-04-26-Muenchen-Tunnel_4K0G0070',
            '2012-04-26-Muenchen-Tunnel_4K0G0090',
            '2012-04-26-Muenchen-Tunnel_4K0G0020',
            '2012-04-26-Muenchen-Tunnel_4K0G0100',
            '2012-04-26-Muenchen-Tunnel_4K0G0060'
        ]
        # check all images, if it exists?
        # max object per patch
        self.augment = augment
        self.multiscale = multiscale
        if augment:
            self.aug = OrientAugment(0.2, 0.5, 0.8)
            # self.aug = OrientAugment(0.2, 0.5, -1)
        else:
            self.aug = OrientAugment(-1, -1, -1)
        # 416, 448, 480, 512, 544, 576, 608
        self.img_size = 512
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        # read all annotations
        self.imgs, self.boxes = self.read_all_notes()
        # store every image in memory
        self.img_memory = {}
        for img_name in self.imgs:
            image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.img_memory[img_name] = img
        # copy each image 100 times
        self.imgs = self.imgs * 100


    def xywhp2points(self, xcenter,ycenter,swidth,sheight,angle):
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
            points.append(x)
            points.append(y)
        return points

    def read_all_notes(self):
        imgs = []
        labels = {}
        for t in self.short:
            img_name = os.path.join(self.root, t+'.JPG')
            if os.path.exists(img_name):
                imgs.append(img_name)
            else:
                raise ValueError("image not found: ", t)
            # collect all labels
            labels[img_name] = []
            for suffix in ['_pkw.samp','_bus.samp','_truck.samp']:
                note_name = os.path.join(self.root,t+suffix)
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
                    box = [xcenter, ycenter, swidth, sheight, angle]
                    labels[img_name].append(box)
        return imgs, labels

    def clip1024(self, img_name):
        '''
        clip a 1024x1024 square from the image
        '''
        # load image from CPU memory
        img = self.img_memory[img_name]
        height, width, _ = img.shape
        xmin = random.randint(0, height-1024)
        ymin = random.randint(0, width-1024)
        xmax, ymax = xmin+1024, ymin+1024
        img = img[xmin:xmin+1024, ymin:ymin+1024, :].copy()
        # ------
        #  Box
        # ------
        boxes = []
        # for cx, cy, h, w in self.boxes:
        for cy, cx, w, h, ang in self.boxes[img_name]:
            if xmin <= cx < xmax and ymin <= cy < ymax:
                box = self.xywhp2points(cy-ymin, cx-xmin, w, h, ang)
                boxes.append(box)
        return img, boxes

    def __getitem__(self, index):
        '''
        no augment
        only multi-scale
        '''
        img_name = self.imgs[index]
        img, obbs = self.clip1024(img_name)
        if len(obbs) > 0:
            obbs = np.array(obbs)
        else:
            obbs  = None
        # use augment
        img, obbs = self.aug(img, obbs)
        # crop a 512x512 square
        img, obbs = self.crop(img, obbs)
        if obbs is None or len(obbs)==0:
            targets = None
        else:
            targets = torch.zeros((len(obbs), 10))
            targets[:, 2:] = torch.from_numpy(obbs)
        # image to tensor
        img = transforms.ToTensor()(img)
        return img, targets

    def collate_fn(self, batch):
        imgs, targets = [], []
        # add sample index to targets
        for i, (img, boxes) in enumerate(batch):
            imgs.append(img)
            if boxes is None:
                continue
            boxes[:, 0] = i
            targets.append(boxes)
        if len(targets) > 0:
            targets = torch.cat(targets, 0)
        else:
            targets = None
        img_size = random.choice(
            range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, img_size) for img in imgs])
        # resize targets
        if targets is not None:
            targets[:,2:] *= img_size/512
        # transfer target style
        targets = self.styleTransfer(targets)
        return imgs, targets


    def crop(self, image, obbs=None):
        '''
        crop a square area from given image
        image: opencv shape, i.e. [height x width x 3] dtype: uint8
        obbs: oriented boxes, shape=[x1,y1,x2,y2,x3,y3,x4,y4]
        input image size: rectangle
        output image size: square
        '''
        # step 1/5: determine crop size
        height, width, _ = image.shape
        if min(height, width) < 512:
            # padding to 512
            rv = np.zeros([max(height,512),max(width,512),3], dtype=np.uint8)
            rv[:height, :width, :] = image
            image = rv
        crop_size = 512
        # # step 2/5: determine crop center
        # horizontal edge
        xmin = int(crop_size//2)
        xmax = int(width-crop_size//2)
        # vertical edge
        ymin = int(crop_size//2)
        ymax = int(height-crop_size//2)
        cx = random.randint(xmin, max(xmin, xmax))
        cy = random.randint(ymin, max(ymin, ymax))
        # step 3/5: crop the image
        # left top point
        x_top, y_top = cx-crop_size//2, cy-crop_size//2
        image = image[y_top:y_top+crop_size, x_top:x_top+crop_size]
        # step 4/5: adjust label infomation
        if obbs is not None:
            # check if each box locate in crop area
            obb_rv = []
            for i in range(len(obbs)):
                obb = obbs[i]
                xx = sum(obb[0::2])/len(obb[0::2])
                yy = sum(obb[1::2])/len(obb[1::2])
                xx = int(xx)
                yy = int(yy)
                center_in = x_top <= xx < x_top+crop_size and y_top <= yy < y_top+crop_size
                if center_in:
                    # adjust obb box
                    obb[0::2] = obb[0::2]-x_top
                    obb[1::2] = obb[1::2]-y_top
                    obb_rv.append(obb)
                else:
                    # box not in window
                    continue
            if len(obbs) > 0:
                obbs = np.array(obb_rv)
            else:
                # no box in croped area
                obbs = None
        # step 5/5: return crop image and croped labels
        return image, obbs

    def styleTransfer(self, targets):
        '''
        convert targets style from point style to (x,y,w,h,theta) style
        '''
        if targets is None:
            return targets
        boxes = []
        for target in targets:
            box = []
            # batch id
            box.append(target[0].item())
            # cls id
            box.append(target[1].item())
            # points
            points = target[2:].detach().cpu().numpy()
            points = points.reshape(4,2)
            # select center points
            cx = np.sum(points[:,0]) / 4
            cy = np.sum(points[:,1]) / 4
            box.append(cx)
            box.append(cy)
            # minimal width and maximum width
            nums = []
            for i,j in enumerate([1,2,3,0]):
                p = points[i]
                q = points[j]
                l = np.linalg.norm(q-p)
                x, y = q-p
                # theta = np.arctan2(x, y) # should be (y,x)
                theta = np.arctan2(y, x)
                nums.append([l, theta])
            nums.sort()
            w = (nums[0][0] + nums[1][0])/2
            h = (nums[2][0] + nums[3][0])/2
            theta = nums[-1][-1]
            # covnert to [0,pi]
            while theta<0.0:
                theta += np.pi
            while theta>np.pi:
                theta -= np.pi
            box.append(w)
            box.append(h)
            box.append(theta)
            boxes.append(box)
        boxes = np.array(boxes)
        targets = torch.FloatTensor(boxes).to(device=targets.device)
        return targets

    def __len__(self):
        # each epoch:
        #   # 5 images
        #   # each image:
        #   #   # 4 batch
        #       # each batch:
        #       #   # 16 imgs
        #   5x16 = 80 iters(20 optimizations) per epoch
        return len(self.imgs)


def draw_boxes(img, targets):
    # convert image to cv2 shape
    img = img.detach().cpu().numpy()
    r, g, b = img[0,...], img[1,...], img[2,...]
    img = np.stack([r,g,b],axis=-1)
    img = np.array(img*255.0, np.uint8)
    print(img.shape)
    # convert targets to gt_boxes shape
    gt_boxes = targets.detach().cpu().numpy()
    # candidates colors
    color =  (0, 255, 255) # which color is this
    # step 1: read img
    img1 = img
    # step 2: draw ground truth on first image
    for box in gt_boxes:
        # step 2.2 draw the points
        points = [int(t) for t in box[2:]]
        points = np.array(points).reshape(4, 2)
        img1 = cv2.polylines(img1, [points], True, color, 3)
    return img1

def demo():
    '''
    select single img and single targets
    show the demo
    '''
    dd = MunichDataset()
    print(len(dd))
    i = 0
    for img, targets in dd:
        i += 1
        print(i)
        if targets is None:
            continue
        print(i, img.shape, targets.shape)
        # show the demo
        img = draw_boxes(img, targets)
        # show at plt
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()
        # over!
        return


def test_loader():
    dd = MunichDataset()
    print(len(dd))
    i = 0
    for img, targets in dd:
        i += 1
        print(i)
        if targets is None:
            continue
        print(i, img.shape, targets.shape)



if __name__ == "__main__":
    demo()
