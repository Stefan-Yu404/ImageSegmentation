import torch
import torch.nn as nn
import numpy as np

class TverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1
        Tversky = 0.
        a = 0.3
        b = 0.7
        bce_weight = 1.0

        # the definition of Tversky: TP / (TP + a * FN + b * FP)
        for i in range(pred.size(1)):
            TP = (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1)
            FP = ((1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1)
            FN = (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1)

            Tversky += TP / (TP + a * FN + b * FP + smooth)

        Tversky = Tversky / pred.size(1)
        TverskyLoss = torch.clamp((1 - Tversky).mean(), 0, 1)

        sigmoid = torch.nn.Sigmoid()
        out = sigmoid(pred)
        BinaryCrossEnergyLoss = torch.nn.BCELoss()
        BECLoss = BinaryCrossEnergyLoss(out, target)

        return TverskyLoss + bce_weight * BECLoss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1
        dice = 0.

        # the definition of dice: 2 * TP / (2 * TP + FP + FN)
        for i in range(pred.size(1)):
            TP = (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1)
            FP = ((1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1)
            FN = (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1)

            dice += 2 * TP / (2 * TP + FP + FN + smooth)

        dice = dice / pred.size(1)
        diceloss = torch.clamp((1 - dice).mean(), 0, 1)

        return diceloss

def dice_coefficient(y_true, y_pred, label):
    y_true = np.asarray(y_true == label).astype(np.bool)
    y_pred = np.asarray(y_pred == label).astype(np.bool)
    intersection = np.logical_and(y_true, y_pred)
    return (2. * intersection.sum()) / (y_true.sum() + y_pred.sum())

def dice_coef_multilabel(y_true, y_pred, numLabels):
    dices = []
    for index in range(numLabels):
        dice = dice_coefficient(y_true, y_pred, index)
        dices.append(dice)
    diceAverage = np.sum(dice)/numLabels
    return dice, diceAverage