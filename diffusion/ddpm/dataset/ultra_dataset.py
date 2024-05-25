import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
from torchvision import transforms as T
import blobfile as bf


class UltraDataset(Dataset):
    def __init__(self, data_dir, resolution):
        super(UltraDataset, self).__init__()
        self.images_path = _list_files(data_dir)
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((resolution, resolution), antialias=False),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[.5], std=[.5])
        ])

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image_path = self.images_path[item]
        img = sio.loadmat(image_path)['img']
        img = self.transforms(img.astype(np.uint8))
        return img


def _list_files(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        results.append(full_path)
    return results
