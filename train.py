'''
train a naive model
'''
import os
import time
import torch
import random
import numpy as np
from collections import Counter

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from dataset import UcasDataset, HrscDataset, MunichDataset
from model import TwinRes, OneStage

from validation import Validation

import argparse
from config.utils import RotateConfig


device = 'cuda'


def parseArg():
    parser = argparse.ArgumentParser(description='Training args')
    parser.add_argument('-e','--epoch', type=int, default=None, help='training epochs, (default:100)')
    parser.add_argument('-d','--dataset', type=str, default=None, help='dataset, (Ucas, Munich, Hrsc)')
    parser.add_argument('-b','--backbone', type=str, default=None, help='backbone, (resnet18, resnet50, resnet101)')
    parser.add_argument('-n','--model_name', type=str, default=None, help='weight model name')
    parser.add_argument('-bs','--batch_size', type=int, default=None, help='training batch size, (default:8)')
    parser.add_argument('-g','--gamma', type=float, default=None, help='gamma in Angle Loss, (default:1.0)')
    # parser.add_argument('-s','--stride', type=int, default=3, help='the max center offset stride of head predict')
    parser.add_argument('-c','--config', type=str, default='config/default.json', help='config file')
    # parser args
    args = parser.parse_args()
    # check params
    cfg = RotateConfig(args.config)
    cfg.modify(args)
    return cfg

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def valid(model, dataset, writer, global_step):
    # mini validation
    # val = Validation(dataset, size=10)
    val = Validation(dataset)
    eachAP = val(model)
    mAP = 0
    for name, ap in eachAP:
        mAP += ap
        writer.add_scalar('ap50/'+name, ap, global_step)
    mAP /= len(eachAP)
    return  mAP


def train_epoch(model, optimizer, train_dataloader, writer, global_step):
    for img, targets in tqdm(train_dataloader, leave=False):
        if targets is None:
            continue
        img, targets = img.to(device), targets.to(device)
        # 5.1 forward
        loss, boxes = model(img, targets)
        if loss==0:
            continue
        # 5.2 backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 5.3 write log to tensorboard
        cc = Counter()
        for head in model.heads:
            for name in  head.loss_log:
                cc[name] += head.loss_log[name]
        for name in cc:
            writer.add_scalar(name, cc[name], global_step)
        global_step += 1
    return model

class LossChanger:
    def __init__(self, k=10):
        self.modes = ['all','A','B','C']
        self.max = k*len(self.modes)
        self.step = 0
        self.k = k

    def next(self, model):
        if self.step>=self.max:
            self.step = 0
        v, _ = divmod(self.step, self.k)
        mode = self.modes[v]
        model.set_loss_mode(mode)
        self.step += 1

    def all(self, model):
        mode = 'all'
        model.set_loss_mode(mode)


def train(cfg):
    dataset = cfg.get('dataset')
    assert dataset in {'Ucas', 'Hrsc', 'Munich'}
    if  dataset=='Ucas':
        TheDataset = UcasDataset
        cls_num = 2
    elif dataset=='Hrsc':
        TheDataset = HrscDataset
        cls_num = 1
    elif dataset=='Munich':
        TheDataset = MunichDataset
        cls_num = 1
    model_name = cfg.get('model_name')
    batch_size = cfg.get('batch_size')
    epoch = cfg.get('epoch')
    val_epoch = cfg.get('val_epoch')
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    # step1: init seed
    random_seed(4399)
    # model = TwinRes(cls_num,cfg).to(device)
    model = OneStage(cls_num,cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # step5: train
    global_step = 0
    # tensorboard log
    writer = SummaryWriter()
    best_ap = 0.0
    save_time = time.time()
    loss_changer = LossChanger()
    for j in trange(epoch):
        # get training data
        train_dataset = TheDataset()
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=4)
        # TODO: get optimizer
        # if j==0:
        #     lr = 0.0003
        #     optimizer = torch.optim.Adam(model.parameters(), lr)
        # elif j == 60:
        #     lr = 0.00006
        #     # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-6)
        #     optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=1e-4)
        # elif j == 80:
        #     lr = 0.00001
        #     # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-6)
        #     optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=1e-4)
        # train one epoch
        # set no change
        # loss_changer.next(model)
        # if (epoch-j)<=20:
        #     loss_changer.all(model)
        model = train_epoch(model, optimizer, train_dataloader, writer, global_step)
        global_step += len(train_dataloader)
        # validation
        if (j+1)% val_epoch==0:
            ap = valid(model,  dataset, writer, global_step)
            if ap>best_ap:
                # save the best model
                best_ap = ap
                torch.save(model.state_dict(), f"checkpoints/%s_best.pth" % model_name)
    # save the final model
    torch.save(model.state_dict(), f"checkpoints/%s.pth" % model_name)


if __name__ == "__main__":
    cfg = parseArg()
    train(cfg)
