
import os
import numpy as np
import torch
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
import SimpleITK as sitk
from segmentation.dataset.utils import list_files, normalize


class TargetDataset(Dataset):
    def __init__(self, data_dir, split, resolution, r_steps, run_num):
        super(TargetDataset, self).__init__()
        self.images = []
        self.masks = []
        self.imgs_id = []
        resolution = (resolution, resolution)
        image_dir = os.path.join(data_dir, split, 'images')
        mask_dir = os.path.join(data_dir, split, 'labels')
        image_files = list_files(image_dir)
        mask_files = list_files(mask_dir)
        pbar = tqdm(total=len(mask_files), desc=f'Loading {split} data', unit='file')
        for img_path, mask_path in zip(image_files, mask_files):
            img_id = img_path.split('/')[-1].split('.')[0]
            assert img_id == mask_path.split('/')[-1].split('-msk')[0]
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
            img = normalize(img)
            img = cv2.resize(img, resolution, interpolation=cv2.INTER_AREA)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
            mask = cv2.resize(mask, resolution, interpolation=cv2.INTER_AREA)
            self.images.append(img)
            self.masks.append(mask)
            self.imgs_id.append(img_id)
            pbar.update(1)
        pbar.close()
        self.r_steps = r_steps
        self.run_num = run_num
        self.resolution = (resolution, resolution)
        self.condition_transform = T.ToTensor()
        self.image_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[.5], std=[.5])])        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        img_id = self.imgs_id[index]
        mask_true = self.masks[index]
        conditions = []
        img = (image * 255).astype(np.uint8)
        blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
        for r in self.r_steps:
            edge = cv2.Canny(blur, 20, r)
            condition = edge
            condition = self.condition_transform(condition)
            for _ in range(self.run_num):
                conditions.append(condition)
        conditions = torch.stack(conditions, dim=0)
        image = self.image_transform(image)
        return image, conditions, img_id, mask_true
