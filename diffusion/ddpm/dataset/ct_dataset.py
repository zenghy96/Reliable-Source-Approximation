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
    def __init__(self, data_dir, resolution) -> None:
        super(CTDataset).__init__()  
        self.slices = []
        n_images = 30000
        splits = ["dataset-01training", "dataset-02validation", "dataset-03test"]
        try:
            for split in splits:
                data_root = bf.join(data_dir, split)
                for name in bf.listdir(data_root):
                    img_nib = nib.load(bf.join(data_root, name, 'ct.nii.gz'))
                    img_np = img_nib.get_fdata()
                    # msk_nib = nib.load(bf.join(data_root, name, 'mask.nii.gz'))
                    # msk_np = msk_nib.get_fdata()
                    # assert img_np.shape == msk_np.shape
                    for i in range(img_np.shape[-1]):
                        s = img_np[:,:,i].astype(np.float32)                            
                        self.slices.append(s)
                        if len(self.slices) >= n_images:
                            raise ValueError            
        except ValueError:
            pass
            
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((resolution, resolution), antialias=False),
            T.Normalize(mean=[.5], std=[.5])
        ])

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        img = self.slices[index]
        img = normalize(img)
        img = self.transforms(img)
        return img
    

if __name__ == "__main__":
    import shutil
    data_dir = 'data/CTSpine/verse20'
    dataset = CTDataset(data_dir, 320)
    print(len(dataset))
    img = dataset[200]

    plt.figure(figsize=(9,9))
    plt.imshow(img[0])
    plt.savefig(f'ct.png')
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