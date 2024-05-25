import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import cal_dice


@torch.inference_mode()
def validate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for images, masks_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            # move images and labels to correct device and type
            images = images.to(device=device)
            masks_true = masks_true.to(device=device)

            # predict the mask
            masks_pred, v, alpha, beta = net(images)
            if net.n_classes == 1:
                assert masks_true.min() >= 0 and masks_true.max() <= 1, 'True mask indices should be in [0, 1]'
                masks_pred = (F.sigmoid(masks_pred) > 0.5).float() 
            else:
                masks_pred = F.one_hot(masks_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                assert masks_true.min() >= 0 and masks_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            
            # compute the Dice score
            for pred, gt in zip(masks_pred, masks_true):
                dice_score.append(cal_dice(pred, gt).item())
    net.train()
    dice_score = np.array(dice_score)
    return dice_score.mean()
