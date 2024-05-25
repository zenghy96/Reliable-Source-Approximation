import sys
sys.path.append('.')
import SimpleITK as sitk
import cv2
from matplotlib import pyplot as plt
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import albumentations as A
from torchvision import transforms as T
from segmentation.dataset.utils import list_files, normalize



class GoBackDataset(Dataset):
    def __init__(self, data_dir, split, sample_dir, resolution):
        super(GoBackDataset, self).__init__()
        self.images, self.imgs_id, self.labels = [], [], []
        self.target_idx, self.translated_idx = [], []
        sample_files = list_files(sample_dir)
        pbar = tqdm(total=2*len(sample_files), desc=f'Load {split} data', unit='file')
        for sample_file in sample_files:
            img_id = sample_file.split('/')[-1].split('.')[0]
           # load translated image
            mat = sio.loadmat(sample_file)
            sample = mat['sample']
            # sample = np.uint8(sample*255)
            pseudo_label = mat['pseudo'].astype(np.float32)
            self.images.append(sample)
            self.imgs_id.append(img_id+'-a')
            self.labels.append(pseudo_label)
            self.translated_idx.append(len(self.images)-1)
           
            # load target image
            img_path = f'{data_dir}/{split}/images/{img_id}.nii.gz'
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
            img = normalize(img)
            img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
            self.images.append(img)
            self.imgs_id.append(img_id+'-t')
            self.labels.append(pseudo_label)
            self.target_idx.append(len(self.images)-1)
            
            pbar.update(2)
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
        return len(self.samples)

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.labels[index]
        img_id = self.imgs_id[index]
        auged = self.augs(image=img, mask=mask)            
        img = auged['image']
        mask = auged['mask']
        img = self.transforms(img)
        mask = torch.from_numpy(mask).unsqueeze(dim=0)       
        return img, mask, img_id


if __name__ == '__main__':
    data_dir = 'data/VS/T2'
    sample_dir = 'outputs/final_0/translated'
    ds = GoBackDataset(data_dir, 'training', sample_dir, 320)
    img1, mask1, img_id1 = ds[0]
    img2, mask2, img_id2 = ds[1]
    print(img_id1, img_id2)
    plt.figure()
    plt.subplot(221)
    plt.imshow(img1[0])
    plt.subplot(222)
    plt.imshow(img2[0])
    plt.subplot(223)
    plt.imshow(mask1[0])
    plt.subplot(224)
    plt.imshow(mask2[0])
    plt.savefig('example.png')
    plt.close()
