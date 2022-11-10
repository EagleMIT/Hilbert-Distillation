import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import gradcheck
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)) :
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list) :
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2 :
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1, target.to(torch.int64))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None :
            if self.alpha.type() != input.data.type() :
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1).to(torch.int64))
            logpt = logpt * Variable(at)

        loss = -(1 - pt) ** self.gamma * logpt
        if self.size_average :
            return loss.mean()
        else :
            return loss.sum()


def dice_loss(prediction, target) :
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0
    n = prediction.shape[1]
    prediction = torch.softmax(prediction, dim=1)[:, 1:].contiguous()
    target = F.one_hot(target, num_classes=n)
    if target.ndim == 5:
        target = target.permute(0, 4, 1, 2, 3)
    elif target.ndim == 4:
        target = target.permute(0, 3, 1, 2)
    target = target[:, 1:].contiguous()

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, ce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        ce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """

    loss_func = FocalLoss(gamma=2, alpha=torch.FloatTensor([1., 1., 1., 1.]))
    ce = loss_func(prediction, target)

    dice = dice_loss(prediction, target)

    loss = ce * ce_weight + dice * (1 - ce_weight)

    return loss
