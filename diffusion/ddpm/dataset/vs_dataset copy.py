import blobfile as bf
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import SimpleITK as sitk
from .utils import normalize


class MRIDataset(Dataset):
    def __init__(self, data_dir, resolution) -> None:
        super(MRIDataset).__init__()
        self.images = []
        images_0 = []
        images_1 = []
        n_images = 10000    # total training images
        for entry in sorted(bf.listdir(data_dir)):
            mri_path = bf.join(data_dir, entry, "vs_gk_t1_refT1.nii.gz")
            mask_path = bf.join(data_dir, entry, "vs_gk_seg_refT1.nii.gz")
            data = sitk.GetArrayFromImage(sitk.ReadImage(mri_path)).astype(np.float32)
            ann = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
            for i in range(data.shape[0]):
                img = data[i]
                mask = ann[i]
                if mask.sum() > 0:
                    images_1.append(img)
                else:
                    images_0.append(img)
        # We add all images with VS, which are minority in dataset
        n_0 = min(len(images_0), n_images - len(images_1))
        self.images = images_1 + images_0[:n_0]
        print(f'load {len(images_1)} images with VS, {n_0} without VS')

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((resolution, resolution), antialias=False),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[.5], std=[.5])
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = self.images[index]
        img = normalize(img, img.min(), img.max())
        img = self.transforms(img)
        return img
