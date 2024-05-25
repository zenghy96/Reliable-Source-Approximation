

def cal_dice(pred, gt):
    """Calculates the dice coefficient for the masks"""
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    im_sum = pred.sum() + gt.sum()
    intersection = (pred * gt).sum()
    dice = (2. * intersection) / im_sum
    return dice


def cal_IoU(pred, gt):
    """Calculates the IoU for the masks"""
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    intersection = (pred * gt).sum()
    union = ((pred + gt) > 0).sum()
    iou = intersection / union
    return iou


def cal_metrics(pred, gt):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    intersection = (pred * gt).sum()
    im_sum = pred.sum() + gt.sum()
    union = ((pred + gt) > 0).sum()
    dice = (2. * intersection) / im_sum
    iou = intersection / union
    return dice, iou
