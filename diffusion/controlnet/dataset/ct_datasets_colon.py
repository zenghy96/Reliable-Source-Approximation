from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import SimpleITK as sitk
import sys
sys.path.append('.')
from tqdm import tqdm
from diffusion.ddpm.dataset.utils import normalize
import nibabel as nib
import nibabel.orientations as nio
import os
import nibabel as nib
import SimpleITK as sitk
import cv2
import random


class CTDataset(Dataset):
    def __init__(self, data_dir, ann_dir, resolution, split='train'):
        super(CTDataset).__init__()  
        self.images = []
        self.masks = []
        self.imgs_id = []
        self.resolution = (resolution, resolution)
        self.split = split
        n_images = 20000 if split == 'train' else 100
        filenames = os.listdir(ann_dir)
        filenames = sorted(filenames)
        try:
            for filename in filenames:
                name = filename.split('_seg')[0]
                masks = sitk.GetArrayFromImage(sitk.ReadImage(f"{ann_dir}/{filename}"))
                num = masks.shape[0]
                ct_dir = os.path.join(data_dir, name)
                for dcm_root, dirs, files in os.walk(ct_dir):
                    if len(dirs) == 0 and len(files) == num:           
                        ct_volume = getDicomSeriesVolumeImage(dcm_root)
                        for i in range(ct_volume.shape[-1]):
                            s = ct_volume[:, :, i]
                            mask = masks[:, :, i]
                            # if mask.max() > 0:
                            self.images.append(s)
                            self.masks.append(mask)
                            self.imgs_id.append(f"{name}-{i}")
                            if len(self.images) >= n_images:
                                raise ValueError
        except ValueError:
            pass
        self.image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[.5], std=[.5])
        ])
        self.condition_transforms = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]
        img_id = self.imgs_id[index]

        # img = normalize(img)
        # img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_AREA)
        condition = self.generate_condition(img.copy())
        condition = self.condition_transforms(condition)
        img = normalize(img)
        img = cv2.resize(img, self.resolution)
        img = self.image_transforms(img)
        return img, condition, img_id

    def generate_condition(self, image):
        T = 100
        image[image < 0] = 0
        image = normalize(image)
        image = np.uint8(image * 255)
        image = cv2.resize(image, self.resolution)
        # blur = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0)
        edge = cv2.Canny(image, T, T)
        condition = edge
        return condition

def getDicomSeriesVolumeImage(dcm_dir):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dcm_dir)
    dicom_files = sorted(dicom_files, key=lambda x: int(x.split('.dcm')[0].split('-')[-1]))
    reader.SetFileNames(dicom_files)
    data = reader.Execute()
    volume = sitk.GetArrayFromImage(data)
    return volume
   
    
if __name__ == "__main__":
    import shutil
    data_dir = "/raid/zeng_hongye/data/CT_COLONOGRAPHY/CT"
    ann_dir = "/raid/zeng_hongye/data/CT_COLONOGRAPHY/ann/COLONOGRAPHY"
    image_size = 320
    loader = CTDataset(data_dir, ann_dir, image_size)
    save_dir = 'ct'
    os.makedirs(save_dir, exist_ok=True)
    i = 250
    img, condition, img_id = loader[i]
    mask = loader.masks[i]
    plt.figure(figsize=(9, 4))
    plt.subplot(131)
    plt.imshow(img[0])
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    plt.imshow(condition[0])
    plt.savefig(f'{save_dir}/{i}.png')
    plt.close()
