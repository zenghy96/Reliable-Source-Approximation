import scipy.io as sio
import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
import SimpleITK as sitk
from segmentation.dataset.utils import list_files, normalize


class SampleDataset(Dataset):
    def __init__(self, sample_dir, data_dir, split):
        super(SampleDataset, self).__init__()
        self.samples, self.sample_ids = [], []
        self.images, self.labels = [], []
        sample_files = list_files(sample_dir)
        pbar = tqdm(total=len(sample_files), unit='file')
        for sample_file in sample_files:
            sample_id = sample_file.split('/')[-1].split('.')[0]
            # if not sample_id == 'vs_gk_108-18':
            #     continue
            samples = sio.loadmat(sample_file)['samples']
            img_path = f'{data_dir}/{split}/images/{sample_id}.nii.gz'
            msk_path = f'{data_dir}/{split}/labels/{sample_id}-msk.nii.gz'
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
            img = normalize(img)
            img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(msk_path)).astype(np.float32)
            mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_AREA)
            self.samples.append(samples)
            self.sample_ids.append(sample_id)
            self.images.append(img)
            self.labels.append(mask)
            pbar.update(1)
        pbar.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        samples = self.samples[index]
        image = self.images[index]
        label = self.labels[index]
        sample_id = self.sample_ids[index]
        return samples, image, label, sample_id
