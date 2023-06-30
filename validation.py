''' 
compute AP50 at validation part 
create date: 2022-05-04-20:39
'''
import os
import cv2
import sys
import math
import random 
import shutil
import numpy as np
from xml.dom import minidom
from collections import defaultdict

import time 
import json
import torch
import operator
from glob import glob
from tqdm import tqdm, trange
from shapely.geometry import Polygon
 
from test import skew_iou


__all__ = ['Validation']

device = 'cuda' 


def log_average_miss_rate(prec, rec, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


"""
 check if the number is a float between 0.0 and 1.0
"""


def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


"""
computer Skew-IoU between two polygon
"""


def SkewIou(bbox1, bbox2):
    polygon1 = Polygon(zip(bbox1[0::2], bbox1[1::2]))
    polygon2 = Polygon(zip(bbox2[0::2], bbox2[1::2]))
    inter = polygon1.intersection(polygon2).area
    union = polygon1.area + polygon2.area - inter
    iou = inter / (union + 1e-6)
    return iou


"""
 throw error and exit
"""


def error(msg):
    print(msg)
    sys.exit(0)


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def get_ap50(tmp_folder= 'skewIou_mAP'):
    MINOVERLAP = 0.5
    specific_iou_flagged = False

    # GT_PATH = os.path.join(tmp_folder, 'input', 'ground-truth')
    # DR_PATH = os.path.join(tmp_folder, 'input', 'detection-results')
    # # if there are no images then no animation can be shown
    # IMG_PATH = os.path.join(tmp_folder, 'input', 'images-optional')
    GT_PATH = os.path.join(tmp_folder, 'ground-truth')
    DR_PATH = os.path.join(tmp_folder, 'detection-results')
    # if there are no images then no animation can be shown
    IMG_PATH = os.path.join(tmp_folder, 'images-optional')
    # try to import OpenCV if the user didn't choose the option --no-animation
    show_animation = False
    # try to import Matplotlib if the user didn't choose the option --no-plot
    draw_plot = False

    """
    Create a ".temp_files/" and "output/" directory
    """
    TEMP_FILES_PATH = os.path.join(tmp_folder, ".temp_files")
    if not os.path.exists(TEMP_FILES_PATH):  # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)

    """
    ground-truth
        Load each of the ground-truth files into a temporary ".json" file.
        Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_files = []
    for txt_file in ground_truth_files_list:
        # print(txt_file)
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                class_name, x1, y1, x2, y2, x3, y3, x4, y4 = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            bbox = ' '.join([x1, y1, x2, y2, x3, y3, x4, y4])
            if is_difficult:
                bounding_boxes.append(
                    {"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append(
                    {"class_name": class_name, "bbox": bbox, "used": False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # dump bounding_boxes into a ".json" file
        new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    # print(gt_classes)
    # print(gt_counter_per_class)

    """
    detection-results
        Load each of the detection-results files into a temporary ".json" file.
    """
    # get a list with the detection-results files
    dr_files_list = glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            # print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    # tmp_class_name, confidence, left, top, right, bottom = line.split()
                    # 不用水平框描述，而是使用　定向框的描述
                    tmp_class_name, confidence, x1, y1, x2, y2, x3, y3, x4, y4 = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    # print("match")
                    # bbox = left + " " + top + " " + right + " " +bottom
                    bbox = ' '.join([x1, y1, x2, y2, x3, y3, x4, y4])
                    bounding_boxes.append(
                        {"confidence": confidence, "file_id": file_id, "bbox": bbox})
                    # print(bounding_boxes)
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
    Calculate the AP for each class
    """
    sum_AP = 0.0 
    count_true_positives = {}
    eachAP = []
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        """
         Load detection-results of that class
        """
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))

        """
         Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name: 
                    # 不用水平框描述，而是使用　定向框的描述
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    # compute inter section of two oriented boxes
                    ov = SkewIou(bb, bbgt)
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = obj
            # set minimum overlap
            min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

        # print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        # print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        # print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        # print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        eachAP.append([class_name, ap])
        # class_name + " AP = {0:.2f}%".format(ap*100)
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP "
        # print(text)

    # remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH) 

    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    # print(text) 
    text = 'mAP: {:.2f}'.format(mAP)  
    ss = [cls_name + '{:.2f}'.format(cls_ap) for cls_name, cls_ap in eachAP]
    text = text + '[' + ','.join(ss) + ']'
    for i in trange(3, leave=False, desc=text):
        time.sleep(1.5)
    # return mAP
    # return the paired list 
    return eachAP

class Validation:
    '''
    model forward only
    '''
    def __init__(self,  dataset='Ucas', size=None):
        self.dataset = dataset 
        if dataset == 'Ucas':
            self.root = '/home/buaa/dataset/ucas' 
        elif dataset == 'Hrsc':
            self.root = '/home/buaa/Data/TGARS2021/dataset/HRSC2016' 
        elif dataset == 'Dota':
            self.root = '/home/buaa/Data/TGARS2021/dataset/DOTA-v1.0'
        elif dataset == 'Munich':
            self.root = '/home/buaa/Data/TGARS2021/dataset/dlr' 
        else:
            raise ValueError("unknown dataset: ", dataset)  
        if size is None:
            self.mini = False
        else:
            self.mini = True 
            self.size = size  
        # validation folder 
        # use for temporal 
        self.tmp_folder = '.val' 

    def __call__(self, model): 
        # make folder 
        self.gt_path = os.path.join(self.tmp_folder, 'ground-truth') 
        self.result_path = os.path.join(self.tmp_folder, 'detection-results')
        if os.path.exists(self.tmp_folder):
            shutil.rmtree(os.path.join('.', self.tmp_folder))
        os.mkdir(self.tmp_folder) 
        os.mkdir(self.gt_path) 
        os.mkdir(self.result_path)
        # write ground truth 
        val_list = self.write_gt()
        # eval model 
        model.eval() 
        self.predict(model, val_list)  
        eachAP = get_ap50(self.tmp_folder) 
        model.train()  
        # clear the folder 
        if os.path.exists(self.tmp_folder):
            shutil.rmtree(os.path.join('.', self.tmp_folder))
        return eachAP 

    def load_image(self, jpg, resize=None): 
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

    def load_image_batch(self, task_list):
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

    def load_label(self, name, cls_id=None):
        '''
        load the given label file
        '''
        if self.dataset=='Ucas':
            d = {0: 'car', 1: 'plane'}
            cls_name = d[cls_id]
            with open(name, 'r') as fp:
                lines = fp.readlines()
            boxes = []
            for line in lines:
                # 13 individual number
                nums = [float(t) for t in line.split()]
                # points
                box = [cls_name] + [int(t) for t in nums[:8]]
                boxes.append(box)
            return boxes 
        elif self.dataset=='Hrsc':
            cls_name = 'ship'
            obj_list = self.read_xml(name)
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
        elif self.dataset=='Dota':
            with open(name, 'r') as fp: 
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
        assert(0) 
    
    def read_xml(self, xml_name):
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


    def patch_process(self, boxes):
        '''
        patch process for predict boxes
        step1: threshold and sort
        step2: to points style
        step3: skew-iou based nms
        ''' 
        conf_thres=0.5
        nms_thres = 0.3
        # step1: threshold and sort it
        boxes = boxes[boxes[:, 0] >= conf_thres]
        score = boxes[:, 0]
        boxes = boxes[(-score).argsort()]
        # no more than 500 for each patch 
        if len(boxes)>500:
            boxes = boxes[:500,:]
        # if empty, do nothing
        if len(boxes) == 0:
            return []
        # try gpu nms  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    
    def post_process(self, boxes):
        '''
        post process for predict boxes
        step1: threshold and sort
        step2: to points style
        step3: skew-iou based nms
        ''' 
        conf_thres=0.5
        nms_thres = 0.3
        # step1: threshold and sort it
        boxes = boxes[boxes[:, 0] >= conf_thres]
        score = boxes[:, 0]
        boxes = boxes[(-score).argsort()]
        # if empty, do nothing
        if len(boxes) == 0:
            return np.zeros([0,10]) 
        # try gpu nms 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def pointStyle(self, boxes):
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

    def global_nms(self, boxes):
        ''' 
        skew-iou based nms
        '''  
        nms_thres=0.3
        # try gpu nms  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        points = self.pointStyle(rv)
        return points  


    def write_gt(self): 
        if self.dataset == 'Ucas':
            val_txt = os.path.join(self.root, 'val.txt')
            with open(val_txt, 'r') as fp:
                lines = fp.readlines()
            val_list = []
            for line in lines:
                jpg, txt, cls_id = line.split()
                cls_id = int(cls_id)
                val_list.append([jpg, txt, cls_id])   
            if self.mini:
                # mini validation 
                # random select 10 images
                random.shuffle(val_list)
                val_list = val_list[:self.size] 
            # write all ground truth
            new_id = 0
            for jpg, txt, cls_id in tqdm(val_list, leave=False):
                suffix = jpg.split('.')[-1]
                new_id += 1
                short = '0000' + str(new_id)
                short = short[-4:]
                # 2. read boxes
                boxes = self.load_label(txt, cls_id)
                ss = ''
                for box in boxes:
                    ss += ' '.join([str(t) for t in box]) + '\n'
                # 3. save gt to txt
                gt_txt = os.path.join(self.gt_path, short+'.txt')
                with open(gt_txt, 'w') as fp:
                    fp.writelines(ss)
            return val_list
        elif self.dataset == 'Hrsc':  
            root = self.root 
            val_txt = os.path.join(root, 'ImageSets', 'val.txt')
            with open(val_txt, 'r') as fp:
                lines = fp.readlines()
            val_list = []
            for line in lines:
                short = line.split()[0]
                jpg = os.path.join(root, 'Train', 'AllImages', short+'.jpg')
                xml = os.path.join(root, 'Train', 'Annotations', short+'.xml')
                if not os.path.exists(jpg):
                    # print(" Not Found: " + jpg)
                    continue
                val_list.append([jpg, xml])
            if self.mini:
                # mini validation 
                # random select 10 images
                random.shuffle(val_list)
                val_list = val_list[:self.size] 
            # write all ground truth
            new_id = 0 
            for jpg, xml in tqdm(val_list, leave=False):
                suffix = jpg.split('.')[-1]
                new_id += 1
                short = '0000' + str(new_id)
                short = short[-4:]
                # 2. read boxes
                boxes = self.load_label(xml)
                ss = ''
                for box in boxes:
                    ss += ' '.join([str(t) for t in box]) + '\n'
                # 3. save gt to txt
                gt_txt = os.path.join(self.gt_path, short+'.txt')
                with open(gt_txt, 'w') as fp:
                    fp.writelines(ss)
            return val_list
        elif self.dataset == 'Dota': 
            root = self.root  
            src_path = os.path.join(root, 'val')
            # src_path = os.path.join(root, 'train')
            img_src = os.path.join(src_path, 'images')
            txt_src = os.path.join(src_path, 'labelTxt')
            img_list = glob(os.path.join(img_src, 'P????.png'))
            # select a mini img list 
            if self.mini: 
                random.shuffle(img_list)
                mini_list = img_list[:self.size]   
                while len(mini_list)==1:
                    # 2152 and 2330 may cause error 
                    name = mini_list[0]
                    _, name = os.path.split(name)
                    if name not in ['P2152.png', 'P2330.png']: 
                        break 
                    random.shuffle(img_list)
                    mini_list = img_list[:self.size]  
            # write all ground truth 
            for png in tqdm(mini_list, leave=False):
                _, short = os.path.split(png) 
                short = short.split('.', 1)[0]
                txt = os.path.join(txt_src, short+'.txt')
                # read boxes  
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
                ss = ''
                for box in boxes: 
                    ss += ' '.join([str(t) for t in box]) + '\n'
                # save gt to txt 
                gt_txt = os.path.join(self.gt_path, short+'.txt')
                with open(gt_txt, 'w') as fp:
                    fp.writelines(ss)
            return mini_list 
        elif self.dataset == 'Munich': 
            val_list = [
                '2012-04-26-Muenchen-Tunnel_4K0G0040',
                '2012-04-26-Muenchen-Tunnel_4K0G0080',
                '2012-04-26-Muenchen-Tunnel_4K0G0030',
                '2012-04-26-Muenchen-Tunnel_4K0G0051',
                '2012-04-26-Muenchen-Tunnel_4K0G0010'
            ]      
            if self.mini:
                # mini validation 
                # random select 10 images
                random.shuffle(val_list)
                val_list = val_list[:self.size] 
            # write all ground truth
            new_id = 0 
            for t in tqdm(val_list, leave=False):
                img_name = os.path.join(self.root, t+'.JPG')  
                # 2. read boxes 
                boxes = [] 
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
                        points = self.xywhp2points(xcenter, ycenter, swidth, sheight, angle)
                        # convert boxes to point style
                        # boxes.append(['car  1.00'] + points)  
                        boxes.append(['car'] + points)  
                ss = ''
                for box in boxes:
                    ss += ' '.join([str(t) for t in box]) + '\n'
                # 3. save gt to txt
                gt_txt = os.path.join(self.gt_path, t+'.txt')
                with open(gt_txt, 'w') as fp:
                    fp.writelines(ss)
            return val_list
        else:
            # other dataset
            assert(0) 

    def save_txt(self, boxes, txt): 
        '''
        save top-N boxes in validation mode 
        '''
        if self.dataset == 'Ucas':
            d = {0: 'car', 1: 'plane'}
            maxNum = 300
        elif self.dataset == 'Hrsc':
            maxNum = 30
            d = {0: 'ship'}
        elif self.dataset == 'Munich':
            maxNum = 1000
            d = {0: 'car'}
        elif self.dataset == 'Dota':
            maxNum = 1000
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
        for box in boxes[:maxNum]:
            score, cls_id = box[0], box[1]
            cls_id = int(cls_id)
            cls_name = d[cls_id]
            # obb = [max(t,0) for t in box[2:]]
            try:
                obb = [int(t) for t in box[2:]]
            except ValueError:
                obb = [0,0, 0,1,1,1,1,0]
            except OverflowError:
                obb = [0,0, 0,1,1,1,1,0]
            ss += ' '.join([cls_name]+[str(t) for t in [float(score)]+obb]) + '\n'
        with open(txt, 'w') as fp:
            fp.writelines(ss)

    def predict(self, model, val_list):
        '''
        predict result and save them to given folder
        '''
        if self.dataset=='Ucas': 
            with torch.no_grad():
                new_id = 0
                for jpg, txt, cls_id in tqdm(val_list, leave=False):
                    # load image data 
                    imgs = self.load_image(jpg)
                    # model forward 
                    _, boxes = model(imgs) 
                    # get boxes 
                    boxes = self.post_process(boxes[0]) 
                    # save boxes to the txt file
                    # _, short = os.path.split(txt) 
                    new_id += 1
                    short = '0000' + str(new_id)
                    short = short[-4:]
                    result_txt = os.path.join(self.result_path, short+'.txt')
                    self.save_txt(boxes, result_txt) 
        elif self.dataset=='Hrsc': 
            with torch.no_grad():
                new_id = 0 
                for jpg, xml in tqdm(val_list, leave=False):
                    # load image data 
                    # if image resize if need ?
                    # imgs = self.load_image(jpg, resize=0.6) 
                    imgs = self.load_image(jpg, resize=1.0) 
                    # model forward 
                    _, boxes = model(imgs)
                    # get boxes
                    boxes = self.post_process(boxes[0])
                    # boxes[:, 2:] /= 0.6 
                    # save boxes to the txt file 
                    new_id += 1
                    short = '0000' + str(new_id)
                    short = short[-4:]
                    result_txt = os.path.join(self.result_path, short+'.txt')
                    self.save_txt(boxes, result_txt)  
        elif self.dataset=='Dota':
            # build tasks 
            tasks = []  
            for png in tqdm(val_list, leave=False): 
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
            # forward  for each image  
            batch_size = 8 
            with torch.no_grad(): 
                # result cache for patche detection
                cache_result = defaultdict(list) 
                detect_over = []  
                for start in trange(0, len(tasks), batch_size, leave=False):
                    # detect patch in this batch 
                    finish = min(start+batch_size, len(tasks))
                    mini_task_list = tasks[start:finish] 
                    # load imgs  
                    imgs, bias = self.load_image_batch(mini_task_list)
                    # model forward 
                    _, boxes = model(imgs) 
                    # get boxes 
                    boxes = [self.patch_process(box) for box in boxes] 
                    # collect detect result to the cache  
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
                        boxes = self.global_nms(cache_result[img_name]) 
                        _, short = os.path.split(img_name)
                        short = short.split('.', 1)[0] 
                        result_txt = os.path.join(self.result_path, short+'.txt')
                        # save to txt 
                        self.save_txt(boxes, result_txt) 
                        # clear cache 
                        cache_result.pop(img_name) 
                    # clear detect list 
                    detect_over = [] 
        elif  self.dataset=='Munich':  
            # forward  for each image   
            with torch.no_grad():
                new_id = 0
                for t in tqdm(val_list, leave=False):
                    # load image data
                    img_name = os.path.join(self.root, t+'.JPG')  
                    imgs = self.load_image(img_name)
                    # model forward 
                    _, boxes = model(imgs)
                    # get boxes 
                    boxes = self.post_process(boxes[0])
                    # save boxes to the txt file
                    # _, short = os.path.split(txt) 
                    result_txt = os.path.join(self.result_path, t+'.txt') 
                    self.save_txt(boxes, result_txt)  
        else:
            # other dataset
            asssert(0)  

             
    def xywhp2points(self, xcenter,ycenter,swidth,sheight,angle):
        ang = 0 - angle
        points = [] 
        # # 四个点的坐标 
        for p,q in [(1,1),(1,-1),(-1,-1),(-1,1)]:
            x = xcenter + p * swidth * np.cos(ang*np.pi/180) - q * sheight*np.sin(ang*np.pi/180)
            y = ycenter + p * swidth * np.sin(ang*np.pi/180) + q * sheight*np.cos(ang*np.pi/180)
            points.append(int(x))
            points.append(int(y))
        return points 