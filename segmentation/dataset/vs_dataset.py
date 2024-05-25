import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import SimpleITK as sitk
import os
from .utils import *
from tqdm import tqdm
import albumentations as A
import torch


class VSDataset(Dataset):
    def __init__(self, data_dir, split, resolution=320):
        super(VSDataset, self).__init__()
        self.images = []
        self.masks = []
        self.ids = []
        self.split = split
        image_dir = os.path.join(data_dir, split, 'images')
        mask_dir = os.path.join(data_dir, split, 'labels')
        image_files = list_files(image_dir)
        mask_files = list_files(mask_dir)
        pbar = tqdm(total=len(mask_files), desc=f'Load {split} data', unit='file')
        for img_path, mask_path in zip(image_files, mask_files):
            img_id = img_path.split('/')[-1].split('.')[0]
            assert img_id == mask_path.split('/')[-1].split('-msk')[0]
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
            img = normalize(img)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
            self.images.append(img)
            self.masks.append(mask)
            self.ids.append(img_id)
            pbar.update(1)
        pbar.close()

        if split == "training":
            self.augs = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=0.99, p=0.3),
                A.Affine(translate_percent=(0.01,0.01), p=0.3),
                A.Affine(rotate=(-5,5), p=0.3),
                A.Resize(height=resolution, width=resolution)
                ])
        else:
            self.augs = A.Resize(height=resolution, width=resolution)
            
        self.transforms = T.Compose([T.ToTensor(), T.Normalize(mean=[.5], std=[.5])])        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]
        img_id = self.ids[index]
        auged = self.augs(image=img, mask=mask)            
        img = auged['image']
        mask = auged['mask']
        img = self.transforms(img)
        mask = torch.from_numpy(mask).unsqueeze(dim=0)
        return img, mask, img_id
