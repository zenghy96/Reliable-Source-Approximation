from matplotlib import pyplot as plt
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


def getDicomSeriesVolumeImage(dcm_dir):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dcm_dir)
    dicom_files = sorted(dicom_files, key=lambda x: int(x.split('.dcm')[0].split('-')[-1]))
    reader.SetFileNames(dicom_files)
    data = reader.Execute()
    volume = sitk.GetArrayFromImage(data)
    return volume


if __name__ == "__main__":
    data_dir = '/raid/zeng_hongye/data/CT_COLONOGRAPHY/CT'
    ann_dir = '/raid/zeng_hongye/data/CT_COLONOGRAPHY/ann/COLONOGRAPHY'
    data_dst = 'data/CTSpine1K/COLONOGRAPHY'
    ann_dst = 'data/CTSpine1K/COLONOGRAPHY_seg'
    for file in tqdm(os.listdir(ann_dir)):
        name = file.split('_seg')[0]
        masks = sitk.GetArrayFromImage(sitk.ReadImage(f"{ann_dir}/{file}"))
        num = masks.shape[0]
        ct_dir = os.path.join(data_dir, name)
        for dcm_root, dirs, files in os.walk(ct_dir):
            if len(dirs) == 0 and len(files) == num:           
                ct_volume = getDicomSeriesVolumeImage(dcm_root)
                try:
                    for i in range(ct_volume.shape[-1]):
                        s = ct_volume[:, :, i]
                        mask = masks[:, :, i]
                        out = sitk.GetImageFromArray(s)
                        sitk.WriteImage(out, f"{data_dst}/{name}-{i:04d}.nii.gz")
                        out = sitk.GetImageFromArray(mask)
                        sitk.WriteImage(out, f"{ann_dst}/{name}-{i:04d}_seg.nii.gz")
                except IndexError:
                    print(name, ct_volume.shape, masks.shape)
                    continue
                break
        # break
 



   
    
