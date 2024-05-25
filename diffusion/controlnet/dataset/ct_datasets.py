import cv2
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
import blobfile as bf


class CTDataset(Dataset):
    def __init__(self, data_dir, ann_dir, resolution, split) -> None:
        super(CTDataset).__init__()  
        slices_0, slices_1, msk_0, msk_1, _0_ids, _1_ids = [], [], [], [], [], []
        if split == 'train':
            n_images = 8000
            splits = ["dataset-01training", "dataset-02validation"]
        else:
            n_images = 200
            splits = ["dataset-03test"]
        try:
            for split in splits:
                data_root = bf.join(data_dir, split)
                for name in bf.listdir(data_root):
                    if 'sub-verse602' in name:
                        continue
                    img_nib = nib.load(bf.join(data_root, name, 'ct.nii.gz'))
                    img_np = img_nib.get_fdata()
                    msk_nib = nib.load(bf.join(data_root, name, 'mask.nii.gz'))
                    msk_np = msk_nib.get_fdata()
                    msk_np[msk_np<0] = 0
                    assert img_np.shape == msk_np.shape
                    for i in range(img_np.shape[-1]):
                        s = img_np[:,:,i].astype(np.float32)
                        msk = msk_np[:, :, i].astype(np.uint8)
                        if msk.max() > 0:
                            slices_1.append(s)
                            msk_1.append(msk)
                            _1_ids.append(f"{name}-{i}")
                        else:
                            slices_0.append(s)
                            msk_0.append(msk)   
                            _0_ids.append(f"{name}-{i}")
                        if len(msk_1) > n_images * 0.8 and len(msk_0) > n_images * 0.2:
                            raise ValueError
        except ValueError:
            pass       
        n_0 = int(np.median([0, len(slices_0), n_images - len(slices_1)]))
        self.slices = slices_1 + slices_0[:n_0]
        self.masks = msk_1 + msk_0[:n_0]
        self.ids = _1_ids + _0_ids[:n_0]
        self.resolution = (resolution, resolution)

        self.image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[.5], std=[.5]),
        ])
        self.condition_transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = self.slices[index]
        origin = normalize(img)
        origin = cv2.resize(origin, self.resolution, interpolation=cv2.INTER_AREA)
        origin = self.image_transforms(origin)

        img[img<300] = 0
        img = normalize(img)
        img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_AREA)
        condition = generate_condition(img)
        condition = self.condition_transform(condition)

        return origin, condition, img_id
    

def generate_condition(image):
    image = (image * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0)
    T = 70
    edge = cv2.Canny(blur, T, T)
    condition = edge
    return condition


if __name__ == "__main__":
    import shutil
    data_dir = 'data/CTSpine/verse20'
    dataset = CTDataset(data_dir, '', 320, 'test')
    print(len(dataset))
    img, condition, img_id = dataset[122]

    plt.figure(figsize=(9,9))
    plt.subplot(121)
    plt.imshow(img[0])
    plt.subplot(122)
    plt.imshow(condition[0])
    plt.savefig(f'verse_edge.png')
    plt.close()
    
    # image_size = 256
    # loader = CTDataset(data_dir, image_size)
    # save_dir = 'ct'
    # os.makedirs(save_dir, exist_ok=True)
    # for i in tqdm(range(len(loader))):
    #     img = loader[i]
    #     if i == 243:
    #         print()
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.savefig(f'{save_dir}/{i}.png')
    #     plt.close()