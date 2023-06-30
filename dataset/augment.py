'''
augment function:
    R: rotate
    F: flipping
    M: Multi-scale 
Image Enhance function:
    G: graying
    HSV augment
    Blue
    Gamma
    Noise: GaussianNoise
    Sharpen
    Contrast 
'''
import cv2
import random
import numpy as np

__all__ = ['OrientAugment']  
 
class OrientAugment:
    def __init__(self, gray_prob, flip_prob, rotate_prob, cropSlight=False): 
        self.R = OrientAugment.rotate
        self.F = OrientAugment.flipping
        self.G = OrientAugment.graying
        self.cq = OrientAugment.cropSquare
        self.cr = OrientAugment.cropRectangle
        # augment prob  
        self.gray_prob = gray_prob
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob  
        if cropSlight:
            self.slight = True 
            self.crop = self.cropRectangle
        else:
            self.slight = False 
            self.crop = self.cropSquare 

    def __call__(self, image, obbs=None): 
        # augment and crop square 
        # step1/4: graying
        if random.random() < self.gray_prob: 
            image, obbs = self.graying(image, obbs)   
        # step2/4: flipping
        if random.random() < self.flip_prob:
            image, obbs = self.flipping(image, obbs)  
        # step3/4: rotate
        if random.random() < self.rotate_prob:
            image, obbs = self.rotate(image, obbs)    
        return image, obbs 
        # # step4/4: crop a square to the image
        # image, obbs = self.crop(image, obbs)   
        # # image, obbs = self.multiScale(image, obbs)    
        # return image, obbs 
        

    @staticmethod
    def rotate(image, obbs=None):
        '''
        image: opencv shape, i.e. [height x width x 3] dtype: uint8 
        obbs: oriented boxes, shape=[x1,y1,x2,y2,x3,y3,x4,y4].
        ''' 
        # slight = self.slight
        height, width, _ = image.shape 
        # step 1/2: rotate 90, 180, 270 degree
        degree = random.choice(['R', 'RR', 'RRR', 'I']) 
        # if slight:  
        #     # rotate no more than 30 degree
        #     degree = 'I'
        if degree == 'R': 
            # 90 degree
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            if obbs is not None:
                tmp = obbs.copy()
                obbs[:, 0::2], obbs[:, 1::2] = tmp[:, 1::2], width-1-tmp[:, 0::2] 
        elif degree == 'RR': 
            # 180 degree
            image = cv2.rotate(image, cv2.ROTATE_180) 
            if obbs is not None:
                tmp = obbs.copy()
                obbs[:, 0::2], obbs[:, 1::2] = width-1-tmp[:, 0::2], height-1-tmp[:, 1::2] 
        elif degree == 'RRR':
            # 270 degree
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)   
            if obbs is not None:
                tmp = obbs.copy()
                obbs[:, 0::2], obbs[:, 1::2] = height-1-tmp[:, 1::2], tmp[:, 0::2] 
        # early stop 
        # return image, obbs
        # step 2/2: rotate (-30,30) degree
        height, width, _ = image.shape
        cx, cy = width/2, height/2
        degree = 60*random.random()-30
        # degree = 30*random.random()-15
        matrix = cv2.getRotationMatrix2D((cx, cy), degree, 1.0)
        # computing the new bounding dimensions of the image
        nW = int((height*np.abs(matrix[0, 1]))+(width*np.abs(matrix[0, 0])))
        nH = int((width*np.abs(matrix[0, 1]))+(height*np.abs(matrix[0, 0])))
        # move pixel to new image center
        matrix[0, 2] += (nW/2) - cx
        matrix[1, 2] += (nH/2) - cy
        # affine image
        image = cv2.warpAffine(image, matrix, (nW, nH))
        # adjust the label infomations
        if obbs is not None:
            n, k = len(obbs), len(obbs[0])
            k = k//2
            points = np.ones([3, n*k])
            points[:2, :] = obbs.reshape((n*k, 2)).transpose() 
            # affine
            points = np.matmul(matrix, points).transpose() 
            # threshold
            # points[points < 0] = 0.0 
            # points[points[0, :]>nW-1] = nW-1
            # points[points[1, :]>nH-1] = nH-1 
            obbs = points.reshape((n, 2*k)) 
        return image, obbs

    @staticmethod
    def flipping(image, obbs=None):
        '''
        image: opencv shape, i.e. [height x width x 3] dtype: uint8
        obbs: oriented boxes, shape=[x1,y1,x2,y2,x3,y3,x4,y4], all coordinates in (0,1)  
        '''
        if random.random() < 0.5:
            image, obbs = OrientAugment.flippingHorizontal(
                image, obbs)
        else:
            image, obbs = OrientAugment.flippingVertical(
                image, obbs)
        return image, obbs

    @staticmethod
    def flippingHorizontal(image, obbs=None):
        '''
        image: opencv shape, i.e. [height x width x 3] dtype: uint8 
        obbs: oriented boxes, shape=[x1,y1,x2,y2,x3,y3,x4,y4], all coordinates in (0,1)  
        '''
        height, width, _ = image.shape
        image = cv2.flip(image, 1) 
        # obb boxes
        if obbs is not None:
            obbs[:, 0::2] = width-1-obbs[:, 0::2] 
        return image, obbs

    @staticmethod
    def flippingVertical(image, obbs=None):
        '''
        image: opencv shape, i.e. [height x width x 3] dtype: uint8 
        obbs: oriented boxes, shape=[x1,y1,x2,y2,x3,y3,x4,y4], all coordinates in (0,1)  
        '''
        height, width, _ = image.shape
        # image = torch.flip(image, [-2])
        image = cv2.flip(image, 0) 
        # obb boxes
        if obbs is not None:
            obbs[:, 1::2] = height-1-obbs[:, 1::2] 
        return image, obbs

    @staticmethod
    def graying(image, obbs=None):
        '''
        image: opencv shape, i.e. [height x width x 3] dtype: uint8
        boxes: horizontal boxes, shape=[x,y,w,h], all coordinates in (0,1)
        obbs: oriented boxes, shape=[x1,y1,x2,y2,x3,y3,x4,y4], all coordinates in (0,1) 
        if obbs is not None, it should have same length with boxes.
        angles: object angles, note in radian(not degree)
        '''
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image, obbs

    @staticmethod
    def cropSquare(image, obbs=None):
        '''
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
        # # horizontal edge 
        # xmin = int(width*(0.1+crop_scale/2)) 
        # xmax = int(width*(0.9-crop_scale/2))
        # # vertical edge
        # ymin = int(height*(0.1+crop_scale/2))
        # ymax = int(height*(0.9-crop_scale/2)) 
        # cx = random.randint(xmin, max(xmin, xmax))
        # cy = random.randint(ymin, max(ymin, ymax)) 
        # step 2/5: determine crop center 
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

    @staticmethod
    def cropRectangle(image, obbs=None):
        '''
        crop several pixel from one side of picture
        image: opencv shape, i.e. [height x width x 3] dtype: uint8 
        obbs: oriented boxes, shape=[x1,y1,x2,y2,x3,y3,x4,y4] 
        input image size: rectangle 
        output image size: rectangle 
        '''
        maxLen = 8
        # crop image from each side no more than maxLen pixels
        height, width, _ = image.shape
        assert(maxLen*2<height)
        assert(maxLen*2<width)
        # crop numbers
        left = random.randint(0, maxLen)
        right = random.randint(0, maxLen)
        top = random.randint(0, maxLen)
        bottom = random.randint(0, maxLen)
        # crop from the four sides
        image = image[top:height-bottom, left:width-right, :] 
        if obbs is not None:
            obbs[::2] -= left 
            obbs[1::2] -= top 
        return image, obbs 

    @staticmethod
    def multiScale(image, obbs=None):
        '''
        image: opencv shape, i.e. [height x width x 3] dtype: uint8 
        obbs: oriented boxes, shape=[x1,y1,x2,y2,x3,y3,x4,y4]  
        input image size: rectangle
        output image size: square 
        '''
        # step 1/5: determine crop size
        height, width, _ = image.shape 
        # determine the crop_scale 
        if obbs is not None:
            points = np.array(obbs)
            ws = np.max(points[:,::2], axis=1)-np.min(points[:,::2], axis=1)
            hs = np.max(points[:,1::2], axis=1)-np.min(points[:,1::2], axis=1)
            w = np.max(ws)
            h = np.max(hs) 
            s1 = max(512,w,h) / min(height, width) 
            s1 = min(1, s1)
            s2 = min(1, s1+0.3)
            crop_scale = s1 + (s2-s1) * random.random() 
        else:
            # 0.5 -> 0.8 
            crop_scale = 0.5 + 0.3 * random.random()
        crop_size = int(crop_scale*min(height, width)) 
        # step 2/5: determine crop center 
        # horizontal edge 
        xmin = int(width*(0.1+crop_scale/2)) 
        xmax = int(width*(0.9-crop_scale/2))
        # vertical edge
        ymin = int(height*(0.1+crop_scale/2))
        ymax = int(height*(0.9-crop_scale/2))
        # 0.5 -> 1.0 
        # crop_scale = 0.5 + 0.0 * random.random()
        # crop_size = int(crop_scale*min(height, width))
        # # step 2/5: determine crop center 
        # # horizontal edge 
        # xmin = int(width*(0.0+crop_scale/2)) 
        # xmax = int(width*(1.0-crop_scale/2))
        # # vertical edge
        # ymin = int(height*(0.0+crop_scale/2))
        # ymax = int(height*(1.0-crop_scale/2))
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
                xx = sum(obb[0::2])/len(obb[0::2]) * width
                yy = sum(obb[1::2])/len(obb[1::2]) * height
                xx = int(xx)
                yy = int(yy) 
                obb_in = x_top <= xx < x_top+crop_size and y_top <= yy < y_top+crop_size
                # if x_top <= x < x_top+crop_size and y_top <= y < y_top+crop_size:
                if obb_in:  
                    # adjust obb box
                    obb = obbs[i] 
                    obb[0::2] = (obb[0::2]-x_top) 
                    obb[1::2] = (obb[1::2]-y_top) 
                    obb_rv.append(obb) 
                else:
                    # box not in window
                    continue 
        if obb_rv is not None and len(obb_rv)>0:
            obbs = np.array(obb_rv) 
        # step 5/5: return crop image and croped labels 
        return image, obbs 