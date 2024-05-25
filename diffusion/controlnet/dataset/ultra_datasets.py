import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
from torchvision import transforms as T
import cv2
import json


class UltraDataset(Dataset):
    def __init__(self, image_paths, ann_paths, resolution, **kwargs):
        super(UltraDataset, self).__init__()
        self.image_paths = image_paths
        self.ann_paths = ann_paths
        self.image_transforms = T.Compose([
            T.ToTensor(),
            T.Resize((resolution, resolution), antialias=False),
            T.Normalize(mean=[.5], std=[.5]),
        ])
        self.condition_transform = T.Compose([
            T.ToTensor(),
            T.Resize((resolution, resolution), antialias=False),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = sio.loadmat(image_path)['img'].astype(np.uint8)
        ann_path = self.ann_paths[item]
        with open(ann_path, 'r') as f:
            ann = json.load(f)
            ann = de_ann(ann)
        condition = generate_condition(image, ann)
        condition = self.condition_transform(condition)
        image = self.image_transforms(image)

        return image, condition


def generate_condition(image, ann):
    def find_region_value(image, pt1, pt2):
        h = 5
        x1 = int(min(pt1[0], pt2[0]))
        x2 = int(max(pt1[0], pt2[0]))
        y = int(pt1[1]/2+pt2[1]/2)
        region = image[y-h:y+h, x1:x2]
        return region.mean()

    if len(ann) > 0:
        T = float('inf')
        if 'L0' in ann and 'L1' in ann:
            pt1 = ann['L0']
            pt2 = ann['L1']
            v = find_region_value(image, pt1, pt2)
            if v < T:
                T = v
        if 'L2' in ann and 'L3' in ann:
            pt1 = ann['L2']
            pt2 = ann['L3']
            v = find_region_value(image, pt1, pt2)
            if v < T:
                T = v
        T = T
    else:
        T = image.max() * 0.5

    _, thresh = cv2.threshold(image, T, 1, cv2.THRESH_BINARY)
    condition = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return condition


def de_ann(ann):
    new = {}
    shapes = ann['shapes']
    for shape in shapes:
        label = shape['label']
        pt = shape['points'][0]
        new[label] = pt
    return new