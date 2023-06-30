'''
dataset for UCAS_AOD
random choice 1100 training and 400 test images
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
HRSC2016 dataset
"""


def read_xml(xml_name, x_bias: int = 0, y_bias: int = 0):
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
        # convert all param from str to number
        xmin, ymin, xmax, ymax = [float(t) for t in [xmin, ymin, xmax, ymax]]
        cx, cy, w, h, ang = [float(t) for t in [cx, cy, w, h, ang]]
        x, y = [float(t) for t in [x, y]]
        # define the 12-dim boxes
        box = [
            cls_name,
            xmin-x_bias, ymin-y_bias, xmax-x_bias, ymax-y_bias,
            cx-x_bias, cy-y_bias, w, h, ang,
            x-x_bias, y-y_bias
        ]
        obj_list.append(box)
    return obj_list


def list2box(obj_list):
    '''
    from obj_list format
    to numpy format 
    '''
    n = len(obj_list)
    boxes = []
    obbs = []
    for i, box in enumerate(obj_list):
        # hbb annotation is ignore
        xmin, ymin, xmax, ymax = [float(t) for t in box[1:5]]
        hbb = [(xmin+xmax)/2, (ymin+ymax)/2, (xmax-xmin), (ymax-ymin)]
        # obb annotation
        # 使用 x,y, w, h, ang 的方式
        cx, cy, w, h, ang = [float(t) for t in box[5:10]]
        if h < w:
            h, w = w, h
        # convert to points style
        obb = []
        for p, q in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
            x = cx + p * h/2 * np.cos(ang) - q * w/2*np.sin(ang)
            y = cy + p * h/2 * np.sin(ang) + q * w/2*np.cos(ang)
            obb.append(x)
            obb.append(y)
        # add to list
        boxes.append(hbb)
        obbs.append(obb)
    boxes = np.array(boxes)
    obbs = np.array(obbs)
    return boxes, obbs


class HrscDataset(Dataset):
    '''
    high resolution ship collections 2016 dataset
    all image was resize to about 800x512 
    and crop a 512x512 area 
    '''

    def __init__(self, augment=True, multiscale=True):
        super().__init__()
        self._root = '/home/buaa/Data/TGARS2021/dataset/HRSC2016'
        # self._root = '/home/buaa/dataset/HRSC2016'
        self._augment = augment
        self._multiscale = multiscale
        if augment:
            # self.aug = OrientAugment(0.5, 0.5, 0.5)
            self.aug = OrientAugment(0.2, 0.5, 0.8)
        else:
            self.aug = OrientAugment(-1, -1, -1)
        # load all train
        self._list = self.get_from_txt('train.txt')
        # resize ratio
        self.resize_lb = 0.6
        self.resize_ub = 1.0

    def get_from_txt(self, name):
        file_name = os.path.join(self._root, 'ImageSets', name)
        assert(os.path.exists(file_name))
        with open(file_name, 'r') as fp:
            lines = fp.readlines()
        img_src = os.path.join(self._root, 'Train', 'AllImages')
        xml_src = os.path.join(self._root, 'Train', 'Annotations')
        rv = []
        for line in lines:
            # remove '\n' char
            short = line[:-1]
            img_name = os.path.join(img_src, short+'.jpg')
            if not os.path.exists(img_name):
                img_name = os.path.join(img_src, short+'.bmp')
                if not os.path.exists(img_name):
                    continue
            xml_name = os.path.join(xml_src, short+'.xml')
            if not os.path.exists(xml_name):
                continue
            rv.append([img_name, xml_name])
        return rv

    def __getitem__(self, index):
        img_name, xml_name = self._list[index]
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # load annotation
        obj_list = read_xml(xml_name)
        boxes, obbs = list2box(obj_list)
        if len(boxes) > 0:
            boxes = np.array(boxes)
            obbs = np.array(obbs)
        else:
            obbs = None
        # if image length is larger than 512, resize that
        height, width, _ = img.shape
        resize_ub = 512 / min(height, width)
        resize_lb = self.resize_lb
        resize_ub = min(resize_ub, self.resize_ub)
        # resize the image
        resize = resize_lb + random.random() * max(0, resize_ub-resize_lb)
        resize = 0.6
        resize = 1.0 
        w, h = int(resize*width), int(resize*height)
        img = cv2.resize(img, (w, h))
        obbs = resize * obbs
        # use augment
        img, obbs = self.aug(img, obbs)
        # crop a square area
        img, obbs = self.crop896(img, obbs)
        # combine (cls_id, boxes, obbs, angles) to targets
        if len(obbs) > 0:
            targets = torch.zeros((len(obbs), 2+8))
            # only 1 classes
            # set cls_id to be 0
            cls_id = 0
            targets[:, 1] = cls_id
            targets[:, 2:] = torch.from_numpy(obbs)
        else:
            targets = None
        # image to tensor
        img = transforms.ToTensor()(img)
        return img, targets

    def collate_fn(self, batch):
        # return self.collate_fn_1344(batch)
        imgs, targets = [], []
        # select new image size per batch
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
        # resize images to input shape
        if self._multiscale:
            # resize to [418,448,480,512,544,576,608]
            # t =  random.choice([416, 448, 480, 512, 544, 576, 608])
            t = random.choice([832, 864, 896, 928, 960])
        else:
            t = 896
        scale_factor = t/896
        if targets is not None:
            # convert coordinate to pixel measure
            targets[:, 2:] = scale_factor * targets[:, 2:]
        imgs = torch.stack([resize(img, t) for img in imgs])
        # convert target to [b, cls_id, x, y, w, h, theta] style
        targets = self.styleTransfer(targets)
        return imgs, targets

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
            points = points.reshape(4, 2)
            # select center points
            cx = np.sum(points[:, 0]) / 4
            cy = np.sum(points[:, 1]) / 4
            box.append(cx)
            box.append(cy)
            # minimal width and maximum width
            nums = []
            for i, j in enumerate([1, 2, 3, 0]):
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
            while theta < 0.0:
                theta += np.pi
            while theta > np.pi:
                theta -= np.pi
            box.append(w)
            box.append(h)
            box.append(theta)
            boxes.append(box)
        boxes = np.array(boxes)
        targets = torch.FloatTensor(boxes).to(device=targets.device)
        return targets

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
            rv = np.zeros([max(height, 512), max(
                width, 512), 3], dtype=np.uint8)
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



    def crop896(self, image, obbs=None):
        '''
        crop a square area from given image
        image: opencv shape, i.e. [height x width x 3] dtype: uint8 
        obbs: oriented boxes, shape=[x1,y1,x2,y2,x3,y3,x4,y4] 
        input image size: rectangle 
        output image size: square 
        '''
        # step 1/5: determine crop size
        height, width, _ = image.shape
        if min(height, width) < 896:
            # padding to 512
            rv = np.zeros([max(height, 896), max(
                width, 896), 3], dtype=np.uint8)
            rv[:height, :width, :] = image
            image = rv
        crop_size = 896
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

    def __len__(self):
        return len(self._list)

 

if __name__ == "__main__": 
    dd = HrscDataset() 
    print(len(dd))
    i = 0
    for img, targets in dd:
        i += 1
        print(i)
        if targets is None:
            continue
        print(i, img.shape, targets.shape)
