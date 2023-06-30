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
import json 

from augment import Augment, OrientAugment

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


"""
HRSC2016 dataset
"""

def read_xml(xml_name, x_bias: int=0, y_bias: int=0):
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
        xmin, ymin, xmax, ymax = [float(t) for t in [xmin,ymin,xmax,ymax]]
        cx,cy,w,h,ang = [float(t) for t in [cx,cy,w,h,ang]]
        x, y = [float(t) for t in [x,y]]
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
        if h<w:
            h, w = w, h  
        # convert to points style  
        obb = []   
        for p,q in [(1,1),(1,-1),(-1,-1),(-1,1)]:
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
        resize_ub = 512 / min(height,width)
        resize_lb = self.resize_lb
        resize_ub = min(resize_ub,self.resize_ub)
        # resize the image 
        resize = resize_lb + random.random() * max(0,resize_ub-resize_lb)
        resize = 0.6 
        w, h = int(resize*width), int(resize*height)
        img = cv2.resize(img,(w,h))
        obbs = resize * obbs   
        # use augment 
        img, obbs = self.aug(img, obbs) 
        # crop a square area 
        img, obbs = self.crop(img, obbs)
        # combine (cls_id, boxes, obbs, angles) to targets  
        if len(obbs)>0:
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
            t = random.choice([416,448,480,512,544,576,608])
        else:
            t = 512 
        scale_factor = t/512 
        if targets is not None:
            # convert coordinate to pixel measure
            targets[:, 2:] = scale_factor * targets[:, 2:]
        imgs = torch.stack([resize(img, t) for img in imgs]) 
        # convert target to [b, cls_id, x, y, w, h, theta] style
        targets = self.styleTransfer(targets)
        return imgs, targets

    def collate_fn_single(self, batch):
        '''
        only single image for each batch
        '''
        assert(len(batch)==1)
        img, targets = batch[0] 
        _, height, width = img.shape 
        # height
        k, v = divmod(height, 32)
        if v>0:
            h = 32*k + 32 
        else:
            h = 32*k 
        # width 
        k, v = divmod(width, 32)
        if v>0:
            w = 32*k + 32
        else:
            w = 32*k 
        imgs = torch.zeros([1,3,h,w])
        imgs[...,:height, :width] = img 
        return imgs, targets


    def collate_fn_512(self, batch):
        imgs, targets = [], []
        # select new image size per batch
        # resize all image to 512x512
        # add sample index to targets
        candidates_resize_ratio = []
        for i, (img, boxes) in enumerate(batch):
            # assert crop_sqaure in __get_item__
            _, h, w = img.shape
            assert(h==w)
            r = 512/h 
            # resize image and put it into left-top corner  
            img = F.interpolate(img.unsqueeze(0), size=(512, 512),
                                mode='nearest').squeeze(0) 
            imgs.append(img)
            if boxes is None:
                continue
            boxes[:, 0] = i
            boxes[:, 2:6] *= r  
            targets.append(boxes)
        imgs = torch.stack(imgs) 
        if len(targets) > 0:
            targets = torch.cat(targets, 0) 
        else:
            targets = None  
        if self._multiscale:  
            img_size = random.choice([416,448,480,512,544,576,608])
            r = img_size / 512
            imgs = F.interpolate(imgs, size=img_size, mode='nearest') 
            targets[:,2:6] *= r   
        else:
            # do nothing 
            img_size = (512,512)     
        return imgs, targets

    def collate_fn_1344(self, batch):
        imgs, targets = [], []
        # select new image size per batch
        # resize all image to 1344 x 1344
        # add sample index to targets
        candidates_resize_ratio = []
        for i, (img, boxes) in enumerate(batch):
            _, h, w = img.shape
            r1 = 1344/h
            r2 = 1344/w 
            if r1>r2:
                # target size
                th = int(h*r2) 
                tw = 1344
                r = r2 
            else:
                th = 1344
                tw = int(w*r1) 
                r = r1  
            if r>1.0:
                r = 1.0 
                tw, th = w, h 
            # resize image 
            candidates_resize_ratio.append([th, tw, r]) 
            # resize image and put it into left-top corner  
            img = F.interpolate(img.unsqueeze(0), size=(th, tw),
                                mode='nearest').squeeze(0)
            # padding zero to 1344x1344
            image = torch.zeros([3, 1344, 1344])
            image[:, :th, :tw] = img 
            imgs.append(image)
            if boxes is None:
                continue
            boxes[:, 0] = i
            boxes[:, 2:6] *= r  
            targets.append(boxes)
        imgs = torch.stack(imgs)
        # crop the black side as much as possible 
        maxh = maxw = 0
        for h, w, _ in candidates_resize_ratio:
            maxh = max(maxh, h)
            maxw = max(maxw, w)
        if len(targets) > 0:
            targets = torch.cat(targets, 0) 
        else:
            targets = None 
        # utils resize padding 
        # resize images to input shape
        if self._multiscale: 
            # img_size, r = random.choice([
            #     [(1088,1088),0.85],
            #     [(1152,1152),0.9],
            #     [(1216,1216),0.95],
            #     [(1280,1280),1.0],
            #     [(1344,1344),1.05],
            #     [(1408,1408),1.10],
            #     # [(1472,1472),1.15]
            # ])
            # r = random.choice(np.arange(0.85, 1.15, 0.05))
            # img_size = int(1344*r)
            img_size = random.choice([
                1152, 1216, 1280, 1344, 1408, 1472, 1536
            ])
            r = 1344/img_size
            imgs = F.interpolate(imgs, size=img_size, mode='nearest') 
            targets[:,2:6] *= r  
            maxh = int(r*maxh)
            maxw = int(r*maxw)
        else:
            img_size = (1344,1344)   
        # crop the black side as much as possible 
        _, _, th, tw = imgs.shape  
        if maxh<th:
            # crop 
            k, v = divmod(maxh, 32)
            if v>0:
                maxh = 32*k + 32
            else:
                maxh = 32*k 
        if maxw<tw:
            # crop 
            k, v = divmod(maxw, 32)
            if v>0:
                maxw = 32*k + 32
            else:
                maxw = 32*k 
        imgs = imgs[:, :, :maxh, :maxw]  
        return imgs, targets

    def collate_fn_1280(self, batch):
        imgs, targets = [], []
        # select new image size per batch
        # resize all image to 1280 x 800
        # add sample index to targets
        candidates_resize_ratio = []
        for i, (img, boxes) in enumerate(batch):
            _, h, w = img.shape
            r1 = 1280/h
            r2 = 1280/w 
            if r1>r2:
                # target size
                th = int(h*r2) 
                tw = 1280
                r = r2 
            else:
                th = 1280
                tw = int(w*r1) 
                r = r1  
            if r>1.0:
                r = 1.0 
                tw, th = w, h 
            # resize image 
            candidates_resize_ratio.append([th, tw, r]) 
            # resize image and put it into left-top corner  
            img = F.interpolate(img.unsqueeze(0), size=(th, tw),
                                mode='nearest').squeeze(0)
            # padding zero to 1280x1280
            image = torch.zeros([3, 1280, 1280])
            image[:, :th, :tw] = img 
            imgs.append(image)
            if boxes is None:
                continue
            boxes[:, 0] = i
            boxes[:, 2:6] *= r  
            targets.append(boxes)
        imgs = torch.stack(imgs)
        # crop the black side as much as possible 
        maxh = maxw = 0
        for h, w, _ in candidates_resize_ratio:
            maxh = max(maxh, h)
            maxw = max(maxw, w)
        if len(targets) > 0:
            targets = torch.cat(targets, 0) 
        else:
            targets = None 
        # utils resize padding 
        # resize images to input shape
        if self._multiscale: 
            # img_size, r = random.choice([
            #     [(1088,1088),0.85],
            #     [(1152,1152),0.9],
            #     [(1216,1216),0.95],
            #     [(1280,1280),1.0],
            #     [(1344,1344),1.05],
            #     [(1408,1408),1.10],
            #     # [(1472,1472),1.15]
            # ])
            r = random.choice(np.arange(0.85, 1.15, 0.05))
            img_size = int(1280*r)
            imgs = F.interpolate(imgs, size=img_size, mode='nearest') 
            targets[:,2:6] *= r  
            maxh = int(r*maxh)
            maxw = int(r*maxw)
        else:
            img_size = (1280,1280)   
        # crop the black side as much as possible 
        _, _, th, tw = imgs.shape
        if maxh<th:
            # crop 
            k, v = divmod(maxh, 32)
            if v>0:
                maxh = 32*k + 32
            else:
                maxh = 32*k 
        if maxw<tw:
            # crop 
            k, v = divmod(maxw, 32)
            if v>0:
                maxw = 32*k + 32
            else:
                maxw = 32*k 
        imgs = imgs[:, :, :maxh, :maxw]  
        return imgs, targets
        
    def collate_fn_640(self, batch):
        imgs, targets = [], []
        # select new image size per batch
        # resize all image to 1280 x 800
        # add sample index to targets
        candidates_resize_ratio = []
        for i, (img, boxes) in enumerate(batch):
            _, h, w = img.shape
            r1 = 400/h
            r2 = 640/w 
            if r1>r2:
                # target size
                th = int(h*r2) 
                tw = 640
                r = r2 
            else:
                th = 400
                tw = int(w*r1) 
                r = r1  
            # resize image 
            candidates_resize_ratio.append([th, tw, r]) 
            # resize image and put it into left-top corner  
            img = F.interpolate(img.unsqueeze(0), size=(th, tw),
                                mode='nearest').squeeze(0)
            # padding zero to 640x400
            image = torch.zeros([3, 400, 640])
            image[:, :th, :tw] = img 
            imgs.append(image)
            if boxes is None:
                continue
            boxes[:, 0] = i
            boxes[:, 2:6] *= r  
            targets.append(boxes)
        imgs = torch.stack(imgs)
        # crop the black side as much as possible 
        maxh = maxw = 0
        for h, w, _ in candidates_resize_ratio:
            maxh = max(maxh, h)
            maxw = max(maxw, w)
        if len(targets) > 0:
            targets = torch.cat(targets, 0) 
        else:
            targets = None 
        # utils resize padding 
        # resize images to input shape
        if self._multiscale: 
            img_size, r = random.choice([
                [(340,544),0.85],
                [(360,576),0.90],
                [(380,608),0.95],
                [(400,640),1.0],
                [(420,672),1.05],
                [(440,704),1.10],
                [(460,736),1.15]
            ])
            imgs = F.interpolate(imgs, size=img_size, mode='nearest') 
            targets[:,2:6] *= r  
            maxh = int(r*maxh)
            maxw = int(r*maxw)
        else:
            img_size = (400,640)  
        # padding height to 480
        nB, _, nH, nW = imgs.shape 
        img = torch.zeros(nB, 3, 480, nW)
        img[:, :, :nH, :] = imgs 
        imgs = img 
        # crop the black side as much as possible 
        _, _, th, tw = imgs.shape
        if maxh<th:
            # crop 
            k, v = divmod(maxh, 32)
            if v>0:
                maxh = 32*k + 32
            else:
                maxh = 32*k 
        if maxw<tw:
            # crop 
            k, v = divmod(maxw, 32)
            if v>0:
                maxw = 32*k + 32
            else:
                maxw = 32*k 
        imgs = imgs[:, :, :maxh, :maxw]  
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

class DotaDataset:
    pass 

"""
Munich dataset
"""
class VehicleDataset(Dataset):
    def __init__(self, root, img_size=512, augment=True, multiscale=True):
        super(VehicleDataset, self).__init__()
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
        self.short = self.checkImages(root, self.short)
        self.img_size = img_size
        # max object per patch
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        # 416, 448, 480, 512, 544, 576, 608
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        with open('data/boxes.json', 'r') as fp:
            self.json = json.load(fp)
        # get first batch number
        self.batch_count = 0
        self.img, self.boxes = self.wholeImage()

    def checkImages(self, path, names):
        rv = []
        for t in names:
            img_name = os.path.join(path, t+'.JPG')
            if os.path.exists(img_name):
                rv.append(t)
        assert len(
            rv) > 0, 'found no data in train folder, please check dataset folder: {}'.format(path)
        return rv

    def wholeImage(self):
        t = random.choice(self.short)
        # store data to numpy with * B G R *
        image = cv2.imread(os.path.join(
            self.path, t+'.JPG'), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)
        _, self.height, self.width = img.shape
        boxes = self.json[t]
        return img, boxes
 
    def readnote(self, note_name):
        '''
        read car, bus, truck boxes 
        '''
        imgs = []
        for name in os.listdir(dataPath):
            short, ext = os.path.splitext(name)
            if ext=='.JPG':
                imgs.append(short)
        # collect box labels
        labels = {}
        for img in imgs:
            labels[img] = []
            for suffix in ['_pkw.samp','_bus.samp','_truck.samp']:
                note_name = os.path.join(dataPath,img+suffix)
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
                    box = normalBox(
                        xcenter, ycenter, swidth, sheight, angle)  
                    labels[img].append(box) 
        json_name = 'boxes.json'
        with open(json_name,'w') as fp:
            json.dump(labels,fp) 

    def __getitem__(self, index):
        # ------
        # Image
        # ------
        xmin = random.randint(0, self.height-512)
        ymin = random.randint(0, self.width-512)
        xmax, ymax = xmin+512, ymin+512
        img = self.img[:, xmin:xmin+512, ymin:ymin+512]
        # ------
        #  Box
        # ------
        boxes = []
        # for cx, cy, h, w in self.boxes:
        for cy, cx, w, h in self.boxes:
            if xmin <= cx < xmax and ymin <= cy < ymax:
                x1, y1 = cx-h//2 - xmin, cy-w//2 - ymin
                x2, y2 = x1+h, y1+w
                x1, y1 = max(x1/512, 0), max(y1/512, 0)
                x2, y2 = min(x2/512, 1), min(y2/512, 1)
                # box = [(x1+x2)/2, (y1+y2)/2, (x2-x1), y2-y1]
                box = [(y1+y2)/2, (x1+x2)/2, (y2-y1), x2-x1]
                box = [cy-ymin, cx-xmin, w, h, ang]
                boxes.append(box)
        boxes = torch.from_numpy(np.array(boxes))
        if len(boxes) > 0:
            targets = torch.zeros((len(boxes), 5))
            targets[:, 1:] = boxes
        else:
            targets = None
        if self.augment:
            if random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
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
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 4 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        if self.batch_count % 16 == 0:
            self.img, self.boxes = self.wholeImage() 
        return imgs, targets
  

    def __len__(self):
        # each epoch:
        #   # 5 images
        #   # each image:
        #   #   # 4 batch
        #       # each batch:
        #       #   # 16 imgs
        #   5x16 = 80 iters(20 optimizations) per epoch
        return len(self.short)*16*4 


if __name__ == "__main__":
    # dd = UcasDataset() 
    # dd = HrscDataset()
    dd = DotaDataset()
    print(len(dd))
    i = 0
    for img, targets in dd:
        i += 1
        print(i)
        if targets is None:
            continue
        print(i, img.shape, targets.shape) 