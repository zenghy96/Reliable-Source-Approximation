from tkinter.font import names
import blobfile as bf
import numpy as np
from sklearn.neighbors import VALID_METRICS
from torch.utils.data import Dataset
from torchvision import transforms as T
import SimpleITK as sitk
from .utils import normalize


class MRIDataset(Dataset):
    def __init__(self, data_dir, resolution) -> None:
        super(MRIDataset).__init__()
        self.images = []
        n_images = 10000    # total training images
        entrys = bf.listdir(data_dir)
        entrys = sorted(entrys, key=lambda x: int(x.split('_')[-1]))
        subjects = []
        try:
            for entry in entrys[:100]:
                subjects.append(entry)
                mri_path = bf.join(data_dir, entry, "vs_gk_t1_refT1.nii.gz")
                data = sitk.GetArrayFromImage(sitk.ReadImage(mri_path)).astype(np.float32)
                for i in range(data.shape[0]):
                    img = data[i]
                    self.images.append(img)
                    if len(self.images) >= n_images:
                        raise ValueError
        except ValueError:
            print(f'load from {subjects}')
    
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
        img = normalize(img)
        img = self.transforms(img)
        return img
