import matplotlib.pyplot as plt
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
    output_dir = 'outputs/mri_demo'
    unet_ckpt_dir = 'checkpoints/vs_ddpm'
    controlnet_ckpt_dir = 'checkpoints/vs_controlnet'
    seg_ckpt_dir = f'checkpoints/vs_unet_seed_3/0599.pth.tar'

    r_steps = [30, 40, 50, 60]
    # r_steps = [30, 40, 50, 60]
    run_num = 3
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
            if not img_id == 'vs_gk_134-22':
                continue
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
            edge = cv2.Canny(blur, 20, r)
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
    device = torch.device("cuda:0")
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
    # r0, r1, n = 60, 110, 6
    # r_steps = np.linspace(float(r0), float(r1), int(n))
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

        samples = np.array_split(samples, n_steps, axis=0)
        conditions = conditions.cpu().numpy().squeeze()
        conditions = np.array_split(conditions, n_steps, axis=0)
        masks_pred = np.array_split(masks_pred, n_steps, axis=0)
        un_maps = np.array_split(un_maps, n_steps, axis=0)

        # plot results
        uncertainty = np.ones((n_steps, ))
        variation = np.ones((n_steps, )) * 10000
        
        # sio.savemat(
        #     f"{config.output_dir}/{img_id}.mat",
        #     {
        #         'samples': samples,
        #         'conditions': conditions,
        #         'preds': masks_pred,
        #         'un_maps': un_maps,
        #     }
        # )

        for r, (condition, sample, mask_pred, un_map) in enumerate(zip(conditions, samples, masks_pred, un_maps)):
            plt.figure(figsize=(12, 9))
            n = config.run_num + 1
            plt.subplot(n, 4, 1)
            plot_img(image)
            plt.subplot(n, 4, 2)
            plot_img(mask_true)
            plt.subplot(n, 4, 3)
            plot_img(condition[0])

            uns = []
            var = cal_var(mask_pred)
            for i in range(sample.shape[0]):
                aligned = sample[i]
                pred = mask_pred[i]
                un = un_map[i]
                un = (un - un.min()) / (un.max() - un.min())
                # un = un > 0.01

                aligned = np.uint8(aligned * 255)
                blur = cv2.GaussianBlur(aligned, ksize=(5, 5), sigmaX=0)    
                thresh = r_steps[r]
                edge = cv2.Canny(blur, thresh, thresh+8)

                plt.subplot(n, 4, i*4+5)
                plot_img(aligned)

                plt.subplot(n, 4, i*4+6)
                plot_img(pred)
                dice = cal_dice(im1=pred, im2=mask_true)
                plt.title(f'dice={dice:.04f}')

                new_pred = get_new_pred(pred, un>0.01)
                dice = cal_dice(im1=new_pred, im2=mask_true)
                plt.subplot(n, 4, i*4+7)
                plot_img(new_pred)
                plt.title(f'dice={dice:.04f}')

                plt.subplot(n, 4, i*4+8)
                plot_img(un>0.0001)
                un = mask_mean(un, un>0.0001)
                plt.title(f'un={un:.06f}')

            plt.savefig(f"{config.output_dir}/{img_id}_{r}.png")
            plt.close()
        del condition, samples, masks_pred
    
