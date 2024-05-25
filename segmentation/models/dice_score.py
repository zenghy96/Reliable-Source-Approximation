import numpy as np
import torch
from torch import Tensor


def dice_coeff(pred: Tensor, target: Tensor):
    # Average of Dice coefficient for all batches, or for a single mask
    smooth = 1e-6
    m1 = pred.view(-1)  # Flatten
    m2 = target.view(-1)  # Flatten
    intersection = (m1 * m2).sum()
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    return dice


def dice_loss(inp: Tensor, target: Tensor):
    return 1 - dice_coeff(inp, target)


def dice_coeff_batch(pred: Tensor, target: Tensor):
    # Average of Dice coefficient for all batches, or for a single mask
    batch_size = pred.shape[0]
    scores = torch.zeros((batch_size,))
    for i in range(batch_size):
        scores[i] = dice_coeff(pred[i], target[i])
    return scores


def dice_coeff_metrics(preds, targets, empty_score=1.0):
    """Calculates the dice coefficient for the images"""
    batch_size = preds.shape[0]
    scores = torch.zeros((batch_size,))
    for i in range(batch_size):
        pred = preds[i]
        target = targets[i]
        if type(preds) == 'tensor':
            im1 = pred.view(-1)  # Flatten
            im2 = target.view(-1)  # Flatten
        else:
            im1 = pred.reshape(-1)
            im2 = target.reshape(-1)
        intersection = (im1 * im2).sum()
        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            scores[i] = empty_score
        else:
            scores[i] = 2. * intersection.sum() / im_sum
    return scores.cpu().tolist()