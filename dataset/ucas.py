'''
dataset for UCAS_AOD
random choice 1100 training and 400 test images 
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

"""
UCAS_AOD dataset
"""

class UcasDataset(Dataset):
    '''
    UCAS_AOD dataset
    total:
        510 car
        1000 plane
        899 negative
    random select 1110 for training, and 400 for test
    '''

    def __init__(self, root='/home/buaa/dataset/ucas', augment=True, multiscale=True):
        '''
        process original image to training part and test part
        random select 1100 for training, and 400 for test
        all image was resized to 1024x512 shape
        if multi-sclae is True:
            crop a square patch instead 1024x512 image
        '''
        super().__init__()
        self._root = root
        # max object per batch
        self._augment = augment 
        self._multiscale = multiscale
        if augment: 
            # self.aug = OrientAugment(0.5, 0.5, 0.5)
            self.aug = OrientAugment(0.2, 0.5, 0.8)
        else:
            self.aug = OrientAugment(-1, -1, -1) 
        # load all train
        self._train_list, self._val_list, self._test_list = self.get_all_list()
        # train mode
        self._list = self._train_list

    def get_all_list(self):
        '''
        split all data to 3 parts:
            50% for training part
            20% for validation part
            30% for test part
        '''
        # load train part and
        if os.path.exists(os.path.join(self._root, 'train.txt')) and os.path.exists(os.path.join(self._root, 'test.txt')) and os.path.exists(os.path.join(self._root, 'val.txt')):
            # already selected
            # read the list from txt
            train_txt = os.path.join(self._root, 'train.txt')
            val_txt = os.path.join(self._root, 'val.txt')
            test_txt = os.path.join(self._root, 'test.txt')
            train_list = []
            with open(train_txt, 'r') as fp:
                ss = fp.readlines()
            for t in ss:
                png, txt, cls_id = t.split()
                cls_id = int(cls_id)
                train_list.append([png, txt, cls_id])
            test_list = []
            with open(test_txt, 'r') as fp:
                ss = fp.readlines()
            for t in ss:
                png, txt, cls_id = t.split()
                cls_id = int(cls_id)
                test_list.append([png, txt, cls_id])
            val_list = []
            with open(val_txt, 'r') as fp:
                ss = fp.readlines()
            for t in ss:
                png, txt, cls_id = t.split()
                cls_id = int(cls_id)
                val_list.append([png, txt, cls_id])
        else:
            # select train list and test list
            # file folder
            car_folder = os.path.join(self._root, 'CAR')
            plane_folder = os.path.join(self._root, 'PLANE')
            all_obj = []
            # all cars
            cls_id = 0
            all_cars = glob(os.path.join(car_folder, 'P????.png'))
            for car in all_cars:
                txt = car[:-3] + 'txt'
                short = os.path.split(car)[-1]
                short = short.split('.')[0]
                all_obj.append([car, txt, cls_id])
            # all airports
            cls_id = 1
            all_planes = glob(os.path.join(plane_folder, 'P????.png'))
            for plane in all_planes:
                txt = plane[:-3] + 'txt'
                short = os.path.split(plane)[-1]
                short = short.split('.')[0]
                all_obj.append([plane, txt, cls_id])
            random.shuffle(all_obj)
            train_list = []
            test_list = []
            val_list = []
            # save the list to txt
            train_txt = os.path.join(self._root, 'train.txt')
            test_txt = os.path.join(self._root, 'test.txt')
            val_txt = os.path.join(self._root, 'val.txt')
            ss = ''
            n1 = int(0.5*len(all_obj))
            n2 = int(0.7*len(all_obj))
            for i in range(n1):
                obj, txt, cls_id = all_obj[i]
                ss += ' '.join([obj, txt, str(cls_id)]) + '\n'
                train_list.append([obj, txt, cls_id])
            with open(train_txt, 'w') as fp:
                fp.writelines(ss)
            ss = ''
            for i in range(n1, n2):
                obj, txt, cls_id = all_obj[i]
                ss += ' '.join([obj, txt, str(cls_id)]) + '\n'
                test_list.append([obj, txt, cls_id])
            with open(val_txt, 'w') as fp:
                fp.writelines(ss)
            ss = ''
            for i in range(n2, len(all_obj)):
                obj, txt, cls_id = all_obj[i]
                ss += ' '.join([obj, txt, str(cls_id)]) + '\n'
                test_list.append([obj, txt, cls_id])
            with open(test_txt, 'w') as fp:
                fp.writelines(ss)
        return train_list, val_list, test_list

    def __getitem__(self, index):
        ''' 
        crop all image patch at size 512x512
        augment & multiscale
        '''
        img_name, txt_name, cls_id = self._list[index]
        # load image
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        # load annotation
        if cls_id < 0:
            lines = None
        else:
            with open(txt_name, 'r') as fp:
                lines = fp.readlines() 
        # load oriented bounding boxes 
        obbs = [] 
        for line in lines:
            box = [float(t) for t in line.split()] 
            # obb boxes
            obb = box[:8]
            obbs.append(obb) 
        if len(obbs) > 0: 
            obbs = np.array(obbs)  
        else:
            obbs  = None  
        # use augment 
        img, obbs = self.aug(img, obbs)  
        # crop a square area 
        img, obbs = self.crop(img, obbs)
        if len(obbs)==0:
            obbs = None 
        # combine (cls_id, obbs) to targets
        if obbs is not None:
            targets = torch.zeros((len(obbs), 2+8))
            targets[:, 1] = cls_id
            # targets[:, 2:] = torch.from_numpy(obbs)
            targets[:, 2:] = torch.from_numpy(obbs) # to pixel measure 
        else:
            targets = None
        # image to tensor
        img = transforms.ToTensor()(img)
        return img, targets

    def collate_fn(self, batch):
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
            t = random.choice([416,448,480,512,544,576,608])
            scale_factor = t/512
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

    def __len__(self):
        return len(self._list)


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


if __name__ == "__main__":
    dd = UcasDataset()  
    print(len(dd))
    i = 0
    for img, targets in dd:
        i += 1
        print(i)
        if targets is None:
            continue
        print(i, img.shape, targets.shape) 