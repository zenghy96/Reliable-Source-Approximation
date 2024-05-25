import matplotlib.pyplot as plt
from regex import I
import scipy.io as sio
import os
from diffusers import DDPMScheduler
import numpy as np
import torch
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
import sys
import argparse
import SimpleITK as sitk
sys.path.append('.')
# ## diffusion
from diffusion.controlnet.models.UNet2DModel import UNet2DModel
from diffusion.controlnet.models.controlnet import ControlNetModel
from diffusion.controlnet.models.pipeline_controlnet import DDPMControlNetPipeline
from diffusion.controlnet.dataset.utils import list_files
from diffusion.controlnet.dataset.vs_datasets import normalize
# ## unet
from segmentation.models.unet import EvidentialUNet
from segmentation.models.model_tools import load_model
from dataclasses import dataclass

from rsa.utils import *


@dataclass
class Config:
    data_dir = 'data/VS/T2'
    split = 'training'
    output_dir = 'outputs/mri_1'
    unet_ckpt_dir = 'checkpoints/vs_ddpm'
    controlnet_ckpt_dir = 'checkpoints/vs_controlnet'
    seg_ckpt_dir = f'checkpoints/vs_unet_seed_3/0599.pth.tar'

    r_steps = [30, 40, 50]
    # r0, r1, n = 20, 60, 6
    # r_steps = np.linspace(float(r0), float(r1), int(n))
    run_num = 2
    num_inference_steps = 50
    seed = 3


class ConditionGenerator(Dataset):
    def __init__(self, data_dir, split, resolution, r_steps, run_num):
        super(ConditionGenerator, self).__init__()
        self.images = []
        self.masks = []
        self.imgs_id = []
        resolution = (resolution, resolution)
        image_dir = os.path.join(data_dir, split, 'images')
        mask_dir = os.path.join(data_dir, split, 'labels')
        image_files = list_files(image_dir)
        mask_files = list_files(mask_dir)
        pbar = tqdm(total=len(mask_files), desc=f'Loading {split} data', unit='file')
        for img_path, mask_path in zip(image_files, mask_files):
            img_id = img_path.split('/')[-1].split('.')[0]
            assert img_id == mask_path.split('/')[-1].split('-msk')[0]
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
            img = normalize(img)
            img = cv2.resize(img, resolution, interpolation=cv2.INTER_AREA)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
            mask = cv2.resize(mask, resolution, interpolation=cv2.INTER_AREA)
            self.images.append(img)
            self.masks.append(mask)
            self.imgs_id.append(img_id)
            pbar.update(1)
        pbar.close()
        self.r_steps = r_steps
        self.run_num = run_num
        self.resolution = (resolution, resolution)
        self.condition_transform = T.ToTensor()
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        img_id = self.imgs_id[index]
        mask_true = self.masks[index]
        conditions = []
        img = (image * 255).astype(np.uint8)
        blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
        for r in self.r_steps:
            edge = cv2.Canny(blur, 10, r)
            condition = edge
            condition = self.condition_transform(condition)
            for _ in range(self.run_num):
                conditions.append(condition)
        conditions = torch.stack(conditions, dim=0)
        return image, conditions, img_id, mask_true
    



if __name__ == "__main__":
    # torch.set_num_threads(1)
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda:1")
    set_seed(config.seed)

    # diffusion model
    noise_scheduler = DDPMScheduler.from_pretrained(config.unet_ckpt_dir, subfolder='scheduler')
    unet = UNet2DModel.from_pretrained(config.unet_ckpt_dir, subfolder='unet')
    controlnet = ControlNetModel.from_pretrained(config.controlnet_ckpt_dir, subfolder='controlnet')
    unet.to(device)
    controlnet.to(device)
    pipeline = DDPMControlNetPipeline(
        contronet=controlnet,
        unet=unet,
        scheduler=noise_scheduler
    )

    # unet model
    evidentialUNet = EvidentialUNet(
        n_channels=1, 
        n_classes=1, 
    )
    evidentialUNet = load_model(evidentialUNet, config.seg_ckpt_dir)
    evidentialUNet.to(device=device)
    evidentialUNet.eval()

    # data
    r_steps = config.r_steps
    print(f'steps: {r_steps}')
    n_steps = len(r_steps)
    condition_generator = ConditionGenerator(
        data_dir=config.data_dir,
        split=config.split,
        resolution=unet.sample_size,
        r_steps=r_steps,
        run_num=config.run_num,
    )
    image_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[.5], std=[.5]),
    ])

    # sample
    for image, conditions, img_id, mask_true in condition_generator:
        conditions = conditions.to(device)
        samples = pipeline(
            conditions,
            num_inference_steps=config.num_inference_steps,
            generator=torch.manual_seed(0),
            output_type='numpy',
        )[0].squeeze()

        # segmentation
        inp = [image_transforms(sample) for sample in samples]
        inp = torch.stack(inp).to(device)
        masks_pred, v, alpha, beta = evidentialUNet(inp)
        masks_pred = (torch.nn.functional.sigmoid(masks_pred) > 0.5).float() 
        masks_pred = masks_pred.cpu().detach().numpy().squeeze()
        un_maps = beta / (v * (alpha - 1))
        un_maps = un_maps.cpu().detach().numpy().squeeze()

        # sio.savemat(
        #     f"{config.output_dir}/{img_id}.mat",
        #     {
        #         'samples': samples,
        #         'conditions': conditions,
        #         'preds': masks_pred,
        #         'un_maps': un_maps,
        #     }
        # )

        plt.figure(figsize=(16, 16))
        for i, (sample, mask_pred, un_map) in enumerate(zip(samples, masks_pred, un_maps)):
            un_map = un_map > 0.01
            mask_pred = get_new_pred(mask_pred, un_map > 0.01)

            plt.subplot(4, 4, i*2+1)
            plot_img(sample)

            plt.subplot(4, 4, i*2+2)
            plot_img(mask_pred)
            dice = cal_dice(im1=mask_pred, im2=mask_true)
            plt.title(f'dice={dice:.04f}')

        plt.subplot(4, 4, 15)
        plot_img(image)
        plt.subplot(4, 4, 16)
        plot_img(mask_true)

        plt.savefig(f"{config.output_dir}/{img_id}.png")
        plt.close()

