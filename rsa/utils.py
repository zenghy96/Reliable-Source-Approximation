import numpy as np
import matplotlib.pyplot as plt
import random
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_mean(matrix, mask_true):
    rows, cols = matrix.shape
    total_sum = 0
    count = 0
    for i in range(rows):
        for j in range(cols):
            if mask_true[i, j] == 1:
                total_sum += matrix[i, j]
                count += 1
    if count > 0:
        mean = total_sum / count
        return mean
    else:
        return 0  # 如果没有掩码覆盖区域，可以返回None或其他适当的值


def cal_var(masks):
    # delete no mask_true pred
    masks_filtered = masks[np.sum(masks, axis=(1, 2)) > 0]
    if masks_filtered.shape[0] < 2:
        # at least get two predicted mask
        return 1
    # iou
    intersection = np.logical_and.reduce(masks_filtered, axis=0)
    union = np.logical_or.reduce(masks_filtered, axis=0)
    iou = np.sum(intersection) / np.sum(union)
    return 1 - iou


def plot_img(ax, img, txt=None, cmap='gray'):
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    if txt is not None:
        ax.set_title(txt)
    

def cal_dice(im1, im2):
    # eval 
    im1 = im1.reshape(-1)
    im2 = im2.reshape(-1)
    intersection = (im1 * im2).sum()
    im_sum = im1.sum() + im2.sum()
    dice_score = 2. * intersection.sum() / im_sum
    return dice_score


def get_new_pred(pred, un):
    if pred.max() > 0:
        xx, yy = np.where(un==1)
        for x, y in zip(xx, yy):
            if pred[x, y] == 0:
                pred[x, y] = 1
    return pred


def find_best(all_var, var_thresh, masks_pred):
    best_var = all_var.min()
    idx1 = int(all_var.argmin())
    if best_var <= var_thresh:
        preds = masks_pred[idx1]
        best_pred = np.logical_or.reduce(preds, axis=0)
        best_pred = np.float32(best_pred)
        nums = preds.reshape((preds.shape[0], -1))
        nums = nums.sum(axis=1)
        idx2 = int(nums.argmax())
        return best_var, best_pred, idx1, idx2
    else:
        return None, None, None, None