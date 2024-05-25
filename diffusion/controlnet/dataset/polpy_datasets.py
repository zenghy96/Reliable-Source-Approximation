import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
from torchvision import transforms as T
import cv2
import json
import matplotlib.pyplot as plt
import random
from utils import _list_files


def load_dataset(
    data_dir,
    ann_dir,
    batch_size,
    image_size,
    split_ratio=0.8,
):
    """
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param split_ratio: the proportion of training data
    """
    if not data_dir:
        raise ValueError("unspecified datasets directory")
    image_paths = _list_files(data_dir)
    ann_paths = _list_files(ann_dir)
    assert len(image_paths) == len(ann_paths)

    train_num = int(len(image_paths)*split_ratio)
    train_dataset = PolypDataset(
        image_paths[:train_num],
        ann_paths[:train_num],
        image_size,
        "train"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )

    test_dataset = PolypDataset(
            image_paths[train_num:],
            ann_paths[train_num:],
            image_size,
            "test"
        )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )

    return train_loader, test_loader


class PolypDataset(Dataset):
    def __init__(self, image_paths, ann_paths, resolution, split):
        super(PolypDataset, self).__init__()
        self.image_paths = image_paths
        self.ann_paths = ann_paths
        self.split = split
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
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_id = image_path.split('/')[-1].split('.')[0]

        ann_path = self.ann_paths[item]
        mask = cv2.imread(ann_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
        condition = generate_condition(image, mask)
        condition = HWC3(condition)
        assert condition.shape == image.shape
        condition = self.condition_transform(condition)
        image = self.image_transforms(image)

        if self.split == "train":
            return image, condition
        else:
            return image, condition, img_id


def generate_condition(image, mask):
    def canny(image, thresh):
        image = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=1)
        edge = cv2.Canny(image, thresh, thresh)
        return edge
    
    def match_mask_edge(mask, edge):
        # count mask edge pixel number
        mask_edge = cv2.Canny(mask, 1, 1)
        _, mask_edge = cv2.threshold(mask_edge, 0, 1, cv2.THRESH_BINARY)
        num = mask_edge.sum()
        # count inter edge pixel number
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask1 = cv2.dilate(mask, kernel, iterations=1)
        mask2 = cv2.erode(mask, kernel, iterations=1)
        mask_ = mask1 - mask2
        _, edge = cv2.threshold(edge, 0, 1, cv2.THRESH_BINARY)
        inter = mask_ * edge
        # count edge pixel number
        inter_num = inter.sum()
        if inter_num < num * 0.25:
            v = float('inf')
        else:
            v = edge.sum()
        return v
    
    threshs = [40, 60, 80, 100]
    if mask.any():
        best_edge = np.zeros_like(mask)
        best_v = float('inf')
        for T in threshs:
            edge = canny(image, T)
            v = match_mask_edge(mask, edge)
            if v < best_v:
                best_edge = edge
                best_v = v
    else:
        T = random.choice(threshs)
        best_edge = canny(image, T)
    condition = best_edge
    return condition


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


if __name__ == "__main__":
    data_dir = "data/PolypDataSet/Kvasir/images"
    ann_dir = "data/PolypDataSet/Kvasir/masks"
    image_paths = _list_files()(data_dir)
    ann_paths = _list_files()(ann_dir)
    train_dataset = PolypDataset(
        image_paths,
        ann_paths,
        256,
        "train"
    )
    for i in range(len(train_dataset)):
        i = 60
        image, condition = train_dataset[i]
        plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(condition)
        plt.savefig(f"outputs/rets/{i}.png")
        plt.close()
    