'''
compute mAP via oriented bounding boxes
'''
import os
import json
from glob import glob
from tqdm import tqdm, trange
from collections import defaultdict, Counter
import shapely
from shapely.geometry import Polygon
import numpy as np

# IOU_THRES
IOU_THRES = 0.5
# work in silence?
SILENCE = True
# work folder
work_folder = '.temp'
if not os.path.exists(work_folder):
    os.mkdir(work_folder)

def txt2json(result_folder, gt_folder):
    '''
    convert all predict boxes to a single txt
    convert all gt boxes to a single json file
    '''
    # 1. convert all result txt file
    boxes = defaultdict(list)
    for name in glob(os.path.join(result_folder, '*txt')):
        _, short = os.path.split(name)
        with open(name, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            nums = line.split()
            boxes[short].append(' '.join(nums))
    # save boxes
    with open(os.path.join(work_folder, 'result.json'), 'w') as fp:
        json.dump(boxes, fp)
    # 2. convert all result txt file
    boxes = defaultdict(list)
    for name in glob(os.path.join(gt_folder, '*txt')):
        _, short = os.path.split(name)
        with open(name, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            nums = line.split()
            boxes[short].append(' '.join(nums))
    # save boxes
    with open(os.path.join(work_folder, 'gt.json'), 'w') as fp:
        json.dump(boxes, fp)
    # done

def process_one_file(result, gt):
    ''''
    process result and gt from a single imag e
    '''
    # to each single category
    predict = defaultdict(list)
    answer = defaultdict(list)
    # search all names
    for t in gt:
        nums = t.split()
        cls = nums[0]
        box = [int(_) for _ in nums[1:]]
        answer[cls].append(box)
    for t in result:
        nums = t.split()
        cls = nums[0]
        score = float(nums[1])
        box = [int(_) for _ in nums[2:]]
        predict[cls].append([score] + box)
    matched_score = defaultdict(list)
    unmatched_score = defaultdict(list)
    for cls in answer:
        boxes = predict[cls]
        # sort result by score
        boxes.sort(key=lambda x: x[0], reverse=True)
        # convert all boxes to polygon
        polygons = []
        for box in boxes:
            bbox = [int(_) for _ in box[1:]]
            score = float(box[0])
            poly = Polygon(zip(bbox[::2], bbox[1::2]))
            polygons.append([score, poly])
        # conver all gt to polygon
        poly_gt = []
        boxes = answer[cls]
        for box in boxes:
            bbox = [int(_) for _ in box]
            poly = Polygon(zip(bbox[::2], bbox[1::2]))
            poly_gt.append(poly)
        # valid gt idxs
        valid = set(i for i in range(len(poly_gt)))
        # match all polygon
        for score, poly1 in polygons:
            match = False
            for idx, poly2 in enumerate(poly_gt):
                if idx not in valid:
                    continue
                inter = poly1.intersection(poly2).area
                union = poly1.area + poly2.area - inter
                iou = inter / (union+1e-6)
                if iou>IOU_THRES:
                    # MATCH !
                    valid.remove(idx)
                    match = True
                    break
            if match:
                matched_score[cls].append(score)
            else:
                unmatched_score[cls].append(score)
    # return two dict
    return matched_score, unmatched_score

def process_all_file():
    # load all gt
    with open(os.path.join(work_folder, 'gt.json'), 'r') as fp:
        gt = json.load(fp)
    # load all result
    with open(os.path.join(work_folder, 'result.json'), 'r') as fp:
        result = json.load(fp)
    # process one-by-one
    if SILENCE:
        candidates = gt.keys()
    else:
        candidates = tqdm(gt.keys())
    match = defaultdict(list)
    unmatch = defaultdict(list)
    for name in candidates:
        res = result.get(name, [])
        gts = gt.get(name, [])
        m1, u1 = process_one_file(res, gts)
        # update
        for cls in m1:
            match[cls] += m1[cls]
        for cls in u1:
            unmatch[cls] += u1[cls]
    return match, unmatch

def tp_triple():
    '''
    compute score-tp-fp amoun each category
    '''
    match, unmatch  = process_all_file()
    catetory = set()
    catetory.update(match.keys())
    catetory.update(unmatch.keys())
    # compute for each cateogry
    triple = defaultdict(list)
    for cls in catetory:
        mm = match[cls]
        uu = unmatch[cls]
        mm.sort(reverse=True)
        uu.sort(reverse=True)
        # (score, tp, fp)
        nums = []
        tp = fp = 0
        i = j = 0
        while i<len(mm) and j<len(uu):
            if mm[i]>uu[j]:
                tp += 1
                score = mm[i]
                i += 1
            else:
                fp += 1
                score = uu[j]
                j += 1
            nums.append([score, tp, fp])
        while i<len(mm):
            tp += 1
            score = mm[i]
            i += 1
            nums.append([score, tp, fp])
        while j<len(uu):
            fp += 1
            score = uu[j]
            j += 1
            nums.append([score, tp, fp])
        triple[cls] = nums
    return triple

def precision_recall():
    '''
    compute precision and recall curves via score-tp-fp triple
    '''
    # load all gt
    with open(os.path.join(work_folder, 'gt.json'), 'r') as fp:
        gt = json.load(fp)
    gt_count = Counter()
    for name in gt:
        for t in gt[name]:
            nums = t.split()
            cls = nums[0]
            gt_count[cls] += 1
    # compute map via (score,tp,fp) triple
    triple = tp_triple()
    candidates = gt_count.keys()
    precision = defaultdict(list)
    recall = defaultdict(list)
    for cls in candidates:
        gt_num = gt_count[cls]
        for score, tp, fp in triple[cls]:
            # ignore score
            p = tp/(tp+fp)
            r = tp/gt_num
            precision[cls].append(p)
            recall[cls].append(r)
    # write to json
    with open(os.path.join(work_folder, 'precision.json'), 'w') as fp:
        json.dump(precision,fp)
    with open(os.path.join(work_folder, 'recall.json'), 'w') as fp:
        json.dump(recall,fp)
    # done

def map_07():
    '''
    mAP @ VOC 07 metric
    use 11 point
    '''
    # load precision and reall
    with open(os.path.join(work_folder, 'precision.json'), 'r') as fp:
        precision = json.load(fp)
    with open(os.path.join(work_folder, 'recall.json'), 'r') as fp:
        recall = json.load(fp)
    AP = defaultdict(list)
    for cls in precision:
        nums = []
        for t in np.arange(0, 1.1, 0.1):
            v = 0
            for p,r in zip(precision[cls], recall[cls]):
                if r>=t:
                    v = max(v,p)
            nums.append(v)
        AP[cls] = sum(nums)/len(nums)
    return AP

def map_12():
    '''
    mAP @ VOC 12 metric
    '''
    # load precision and reall
    with open(os.path.join(work_folder, 'precision.json'), 'r') as fp:
        precision = json.load(fp)
    with open(os.path.join(work_folder, 'recall.json'), 'r') as fp:
        recall = json.load(fp)
    AP = defaultdict(list)
    for cls in precision:
        pre = precision[cls]
        rec = recall[cls]
        pre = [0] + pre + [0]
        rec = [0] + rec + [1]
        # precision 转化为凸曲线
        for i in reversed(range(len(pre)-1)):
            pre[i] = max(pre[i],pre[i+1])
        # 累积pr曲线下方的值
        v = 0
        for i in range(1, len(pre)):
            if rec[i]>rec[i-1]:
                v += (rec[i]-rec[i-1]) * pre[i]
        AP[cls] = v
    return AP


def main():
    result_folder = os.path.join('input', 'detection-results')
    gt_folder = os.path.join('input', 'ground-truth')
    # step1: txt to json
    txt2json(result_folder, gt_folder)
    # step2: compute precision recall
    precision_recall()
    # step3: compute map07 and map12
    m7 = map_07()
    print("\n mAP at VOC 2007:\n")
    for cls in m7:
        print(cls, " : ", m7[cls])
    map = sum(m7.values())/len(m7)
    print("total mAP: ", map)
    m2 = map_12()
    print("\n mAP at VOC 2012:\n")
    for cls in m2:
        print(cls, " : ", m2[cls])
    map = sum(m2.values())/len(m2)
    print("total mAP: ", map)


def check_box():
    '''
    do nothing but check box
    '''
    result_folder = os.path.join('input', 'detection-results')
    gt_folder = os.path.join('input', 'ground-truth')
    txt2json(result_folder, gt_folder)
    # load all gt
    with open(os.path.join(work_folder, 'gt.json'), 'r') as fp:
        gt = json.load(fp)
    # load all result
    with open(os.path.join(work_folder, 'result.json'), 'r') as fp:
        result = json.load(fp)
    # check ground-truth boxes
    print("check ground truth boxes...")
    for name in gt:
        for t in gt[name]:
            nums = t.split()
            cls = nums[0]
            box = [int(_) for _ in nums[1:]]
            poly = Polygon(zip(box[::2], box[1::2]))
            try:
                poly.intersection(poly)
                poly.area
            except shapely.errors.GEOSException:
                print(name, t)
    # check predict boxes
    print("check detection boxes...")
    for name in result:
        for t in result[name]:
            nums = t.split()
            score = nums[1]
            if score.startswith("0.5210502"):
                print(name, t)
            box = [int(_) for _ in nums[2:]]
            poly = Polygon(zip(box[::2], box[1::2]))
            try:
                poly.intersection(poly)
                poly.area
            except shapely.errors.GEOSException:
                print(name, t)


if __name__ == '__main__':
    main()
