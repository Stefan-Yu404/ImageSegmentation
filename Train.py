import os
from tqdm import tqdm
from collections import OrderedDict

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from dataset import Dataset
from dataset.Transforms import *
from models import UNet
from utils import Metrics, Loss, logger, Common




def train(model, train_loader, optimizer, loss_func, n_labels, device, epoch):  
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    # if epoch == 0:
    #     loss = 9999
    # model.train()
    # scheduler.step(loss)

    train_loss = Metrics.AverageMeter()
    train_dice = Metrics.DiceAverage(n_labels)
    train_acc = Metrics.AccuracyAverage(n_labels)
    train_pre = Metrics.PrecisionAverage(n_labels)
    train_spe = Metrics.SpecificityAverage(n_labels)
    train_iou = Metrics.IOUAverage(n_labels)
    # train_MCC = Metrics.MatthewsAverage(n_labels)
    
    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        data, target = batchAugmentation(data.numpy(), target.numpy(), augment = True)
        data, target = torch.from_numpy(data).float(), torch.from_numpy(target).long()
        data = data.permute((0,4,1,2,3)) # batchSize, n_classes/n_channels, w, h, d
        target = Common.to_one_hot_3d(target, n_labels) # batchSize, n_classes/n_channels, w, h, d
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape)
        # pred = torch.argmax(output.data, dim=1)
        # target = torch.argmax(target.data, dim = 1)
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), data.size(0))
        train_dice.update(output, target)
        train_acc.update(output, target)
        train_pre.update(output, target)
        train_spe.update(output, target)
        train_iou.update(output, target)
        # train_MCC.update(output, target)
        
    val_logs = []
    for i in range(n_labels):

        # val_logs.append(OrderedDict({'Train_dice%d'%i: train_dice.avg[i], 'Train_accuracy%d'%i: train_acc.avg[i],
        #                     'Train_precision%d'%i: train_pre.avg[i], 'Train_specificity%d'%i: train_spe.avg[i],
        #                     'Train_iou%d'%i: train_iou.avg[i], 'Train_Matthews correlation coefficient%d'%i: train_MCC.avg[i]}))
        val_logs.append(OrderedDict({'Train_dice%d'%i: train_dice.avg[i], 'Train_accuracy%d'%i: train_acc.avg[i],
                            'Train_precision%d'%i: train_pre.avg[i], 'Train_specificity%d'%i: train_spe.avg[i],
                            'Train_iou%d'%i: train_iou.avg[i]}))

    return val_logs

def val(model, val_loader, loss_func, n_labels,device):
    model.eval()
    val_loss = Metrics.AverageMeter()
    val_dice = Metrics.DiceAverage(n_labels)
    val_acc = Metrics.AccuracyAverage(n_labels)
    val_pre = Metrics.PrecisionAverage(n_labels)
    val_spe = Metrics.SpecificityAverage(n_labels)
    val_iou = Metrics.IOUAverage(n_labels)
    # val_MCC = Metrics.MatthewsAverage(n_labels)

    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            data = data.unsqueeze(1)
            target = Common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            # pred = torch.argmax(output.data, dim=1)
            # target = torch.argmax(target.data, dim = 1)
            loss = loss_func(output, target)

            
            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
            val_acc.update(output, target)
            val_pre.update(output, target)
            val_spe.update(output, target)
            val_iou.update(output, target)
            # val_MCC.update(output, target)
            
    val_logs = []
    for i in range(n_labels):

        # val_logs.append(OrderedDict({'Val_dice%d'%i: val_dice.avg[i], 'Val_accuracy%d'%i: val_acc.avg[i],
        #                     'Val_precision%d'%i: val_pre.avg[i], 'Val_specificity%d'%i: val_spe.avg[i],
        #                     'Val_iou%d'%i: val_iou.avg[i], 'Val_Matthews correlation coefficient%d'%i: val_MCC.avg[i]}))
        val_logs.append(OrderedDict({'Val_dice%d'%i: val_dice.avg[i], 'Val_accuracy%d'%i: val_acc.avg[i],
                            'Val_precision%d'%i: val_pre.avg[i], 'Val_specificity%d'%i: val_spe.avg[i],
                            'Val_iou%d'%i: val_iou.avg[i]}))


    return val_logs

if __name__ =="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 1
    for epoch in range(epochs):
        pass
