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
from models import UNet, SwinUnet
from utils import Metrics, Loss, logger, Common
from Train import train,val


if __name__ == "__main__":
    target = "C:\\Users\\zyu\\Desktop\\ZY2\\Codes"
    os.chdir(target)
    
    print(os.getcwd())
    batchSize = 2
    patchSize = (160, 160, 160)
    epochs = 600
    lr = 1e-3
    wd = 3e-4
    n_labels = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinUnet.swinUnet_t_3D().to(device)
    Common.print_network(model)
    losslist = [Loss.TverskyLoss(), Loss.DiceLoss()]
    loss_func = losslist[1]
    
    savePath = "./logs/"
    if not os.path.exists(savePath): os.mkdir(savePath)
    
    for i in range(n_labels):
        exec("log%d = logger.Train_Logger(savePath, 'train_log%d')"%(i,i))
    
    
    # Knee_dataset_train = Dataset.KneeDataset(patchSize= patchSize, Path = "../Data/Resize/Train/")

    # Knee_dataset_val = Dataset.KneeDataset(patchSize= patchSize, Path = "../Data/Resize/Test/")
    Knee_dataset_train = Dataset.KneeDataset(patchSize= patchSize, Path = "../Data/Resize/Train/")

    Knee_dataset_val = Dataset.KneeDataset(patchSize= patchSize, Path = "../Data/Resize/Test/")
    
    train_loader = torch.utils.data.DataLoader(dataset=Knee_dataset_train, 
                                            batch_size=batchSize, 
                                            shuffle=True,
                                            num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=Knee_dataset_val, 
                                            batch_size=batchSize, 
                                            shuffle=True,
                                            num_workers=0)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor=0.5, patience = 10, threshold = 0.0001, threshold_mode = "abs", eps = 1e-8)
    
    best = [0, 0]  # Initialize the epoch and performance of the optimal model.
    

    for epoch in range(1, epochs + 1):
        Common.adjust_learning_rate(optimizer, epoch, lr)
        train_logs = train(model, train_loader, optimizer, loss_func = loss_func, n_labels=12, device = device, epoch = epoch)
        val_logs = val(model, val_loader, loss_func = loss_func, n_labels=12, device = device)

        for i in range(n_labels):
            train_log = train_logs[i]
            val_log = val_logs[i]
            exec("log%d.update(epoch, train_log, val_log)"%(i))

        # Save checkpoint.
        if epoch%10 ==0:
            torch.save(model.state_dict(), "%scheckpoint/"%savePath + str(epoch) + '.pth')
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(savePath, 'checkpoint/latest_model.pth'))


    