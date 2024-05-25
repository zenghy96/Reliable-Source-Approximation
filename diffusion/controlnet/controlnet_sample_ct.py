from cv2 import norm
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import blobfile as bf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from diffusers import DDPMScheduler
import numpy as np
import torch
from models.UNet2DModel import UNet2DModel
from models.controlnet import ControlNetModel
from models.pipeline_controlnet import DDPMControlNetPipeline
from tqdm import tqdm
import argparse
from dataset.utils import _list_files
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
import SimpleITK as sitk
from dataset.vs_datasets import normalize
import albumentations as A
import nibabel as nib


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='data/CTSpine/verse20')
parser.add_argument("--output_dir", type=str, default='results/ct_control_')
parser.add_argument("--sample_save_dir", type=str, default='results/VS/samples')
parser.add_argument("--unet_ckpt_dir", type=str, default='ckpts/ct_ddpm_official')
parser.add_argument("--controlnet_ckpt_dir", type=str, default='ckpts/ct_controlnet/checkpoint-12000')
parser.add_argument("--r_steps", type=str, default='60, 100, 3')
parser.add_argument("--run_num", type=int, default=3, help="Number of runs for an image")
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)


class ConditionGenerator(Dataset):
    def __init__(self, data_dir, ann_dir, resolution, r_steps, run_num, n_images):
        super(ConditionGenerator, self).__init__()
        slices, masks, ids = [], [], []
        split = "dataset-03test"
        try:
            data_root = bf.join(data_dir, split)
            for name in bf.listdir(data_root):
                img_nib = nib.load(bf.join(data_root, name, 'ct.nii.gz'))
                img_np = img_nib.get_fdata()
                msk_nib = nib.load(bf.join(data_root, name, 'mask.nii.gz'))
                msk_np = msk_nib.get_fdata()
                msk_np[msk_np<0] = 0
                assert img_np.shape == msk_np.shape
                for i in range(img_np.shape[-1]):
                    s = img_np[:,:,i].astype(np.float32)
                    msk = msk_np[:, :, i].astype(np.uint8)
                    if msk.max() > 7:
                        slices.append(s)
                        masks.append(msk)
                        ids.append(f"{name}-{i}")
                    if len(masks) > n_images:
                        raise ValueError
        except ValueError:
            pass   

        self.images = slices
        self.masks = masks
        self.imgs_id = ids
        self.r_steps = r_steps
        self.run_num = run_num
        self.resolution = (resolution, resolution)
        self.condition_transform = T.Compose([
            T.ToTensor(),
            # T.Resize((resolution, resolution), antialias=False),
            # T.Normalize(mean=[.5], std=[.5]),
        ])

    def __getitem__(self, index):
        image = self.images[index]
        origin = image.copy()
        image[image<300] = 0
        image = normalize(image)
        image = cv2.resize(image, self.resolution, interpolation=cv2.INTER_AREA)
        img_id = self.imgs_id[index]
        mask = self.masks[index]

        conditions = []
        img = (image * 255).astype(np.uint8)
        blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
        for r in self.r_steps:
            edge = cv2.Canny(blur, r, r)
            condition = self.condition_transform(edge)
            for _ in range(self.run_num):
                conditions.append(condition)
        conditions = torch.stack(conditions, dim=0)
        return origin, conditions, img_id, mask


if __name__ == "__main__":
    torch.set_num_threads(1)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sample_save_dir, exist_ok=True)
    device = torch.device("cuda:0")

    # model
    noise_scheduler = DDPMScheduler.from_pretrained(args.unet_ckpt_dir, subfolder='scheduler')
    unet = UNet2DModel.from_pretrained(args.unet_ckpt_dir, subfolder='unet')
    controlnet = ControlNetModel.from_pretrained(args.controlnet_ckpt_dir, subfolder='controlnet')
    unet.to(device)
    controlnet.to(device)
    pipeline = DDPMControlNetPipeline(
        contronet=controlnet,
        unet=unet,
        scheduler=noise_scheduler
    )

    # data
    # r0, r1, n = 60, 100, 5
    # r_steps = np.linspace(float(r0), float(r1), int(n))
    r_steps = [40, 60, 80, 100]
    # r_steps = [100, 130]
    condition_generator = ConditionGenerator(
        data_dir=args.data_dir,
        ann_dir=args.data_dir,
        resolution=unet.sample_size,
        r_steps=r_steps,
        run_num=args.run_num,
        n_images=100,
    )

    # sample
    for image, conditions, img_id, mask in condition_generator:
        # image = cv2.resize(image, (unet.sample_size, unet.sample_size))
        resolution = (image.shape[1], image.shape[0])
        conditions = conditions.to(device)
        samples = pipeline(
            conditions,
            num_inference_steps=args.num_inference_steps,
            generator=torch.manual_seed(args.seed),
            output_type='numpy',
        )[0].squeeze()

        # save samples
        # sio.savemat(f"{args.sample_save_dir}/{img_id}.mat", {f'sample': samples})

        # plot image, conditions and samples
        n_steps = len(r_steps)
        samples = np.array_split(samples, n_steps, axis=0)
        conditions = conditions.cpu().numpy().squeeze()
        conditions = np.array_split(conditions, n_steps, axis=0)          
        for r, (condition, sample) in enumerate(zip(conditions, samples)):
            plt.figure(figsize=(9, 9))
            plt.subplot(231)
            plt.imshow(image, cmap='gray')
            plt.subplot(232)
            plt.imshow(mask)
            plt.subplot(233)
            plt.imshow(condition[0])
            for i in range(sample.shape[0]):
                plt.subplot(2, 3, i+4)
                s = sample[i]
                s = cv2.resize(s, resolution, interpolation=cv2.INTER_AREA)
                plt.imshow(s, cmap='gray')
            plt.savefig(f"{args.output_dir}/{img_id}_{r+1}.png")
            plt.close()
