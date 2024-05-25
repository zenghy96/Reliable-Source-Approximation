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


class CTDataset(Dataset):
    def __init__(self, data_dir, resolution) -> None:
        super(CTDataset).__init__()  
        self.slices = []
        n_images = 30000
        data_dir = '/raid/zeng_hongye/data/CT_COLONOGRAPHY/CT'
        ann_dir = '/raid/zeng_hongye/data/CT_COLONOGRAPHY/ann/COLONOGRAPHY'
        anns_path = os.listdir(ann_dir)
        anns_path = sorted(anns_path)[:400]
        try:
            for entry in tqdm(anns_path):
                name = entry.split('_seg')[0]
                masks = sitk.GetArrayFromImage(sitk.ReadImage(f"{ann_dir}/{entry}"))
                num = masks.shape[0]
                ct_dir = os.path.join(data_dir, name)
                for dcm_root, dirs, files in os.walk(ct_dir):
                    if len(dirs) == 0 and len(files) == num:           
                        ct_volume = getDicomSeriesVolumeImage(dcm_root)
                        for i in range(ct_volume.shape[-1]):
                            s = ct_volume[:, :, i].astype(np.float32)
                            self.slices.append(s)
                            if len(self.slices) >= n_images:
                                raise ValueError
                        break           
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
    data_dir = '/rashongye/data/CT_COLONOGRAPHY/ann/COLONOGRAPHY'
    data_dir = '/raid/zeng_hongye/data/CT_COLONOGRAPHY/CT'
    ann_dir = '/raid/zeng_hongye/data/CT_COLONOGRAPHY/ann/COLONOGRAPHY'
    anns_path = os.listdir(ann_dir)
    anns_path = sorted(anns_path)[:400]
    for entry in anns_path:
        name = entry.split('_seg')[0]
        masks = sitk.GetArrayFromImage(sitk.ReadImage(f"{ann_dir}/{entry}"))
        num = masks.shape[0]
        ct_dir = os.path.join(data_dir, name)
        for dcm_root, dirs, files in os.walk(ct_dir):
            if len(dirs) == 0 and len(files) == num:           
                ct_volume = getDicomSeriesVolumeImage(dcm_root)
                break
        break
    img = ct_volume[:,:,222]
    # plt.figure()
    # plt.hist(img)
    # plt.savefig(f'hist.png')
    plt.figure(figsize=(9,9))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')

    plt.subplot(132)
    # img[img<0]=0
    a = normalize(img)
    a = np.uint8(a * 255)
    blur = cv2.GaussianBlur(a, ksize=(3, 3), sigmaX=0)
    edge = cv2.Canny(blur, 100, 100)
    plt.imshow(edge, cmap='gray')

    plt.subplot(133)
    img[img<0]=0
    plt.imshow(img, cmap='gray')

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