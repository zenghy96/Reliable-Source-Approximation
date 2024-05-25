from cv2 import norm
import matplotlib.pyplot as plt
import scipy.io as sio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='data/Vestibular-Schwannoma-SEG')
parser.add_argument("--output_dir", type=str, default='results/VS——002')
parser.add_argument("--sample_save_dir", type=str, default='results/VS/samples')
parser.add_argument("--unet_ckpt_dir", type=str, default='ckpts/vs_ddpm')
parser.add_argument("--controlnet_ckpt_dir", type=str, default='ckpts/vs_controlnet_final')
parser.add_argument("--r_steps", type=str, default='60, 100, 3')
parser.add_argument("--run_num", type=int, default=3, help="Number of runs for an image")
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)


class ConditionGenerator(Dataset):
    def __init__(self, data_dir, ann_dir, resolution, r_steps, run_num, n_images):
        super(ConditionGenerator, self).__init__()
        images, masks, ids = [], [], []
        domain = '2'
        subjects_name = os.listdir(data_dir)
        anns_name = os.listdir(ann_dir)
        subjects_name = sorted(subjects_name, key=lambda x: int(x.split('_')[-1]))
        anns_name = sorted(anns_name, key=lambda x: int(x.split('_')[-1]))
        subjects_name = subjects_name[197:]
        anns_name = anns_name[197:]
        print(subjects_name)
        try:
            for subject_name, ann_name in zip(subjects_name, anns_name):
                assert subject_name == ann_name
                # if subject_name == 'vs_gk_':
                data = sitk.ReadImage(f"{data_dir}/{subject_name}/vs_gk_t{domain}_refT{domain}.nii.gz")
                data = sitk.GetArrayFromImage(data).astype(np.float32)
                ann = sitk.ReadImage(f"{data_dir}/{ann_name}/vs_gk_seg_refT{domain}.nii.gz")
                ann = sitk.GetArrayFromImage(ann).astype(np.float32)
                for i, (img, mask) in enumerate(zip(data, ann)):
                    if mask.sum() > 150:
                        images.append(img)
                        masks.append(mask)
                        ids.append(f"{subject_name}-{i}")
                        if len(images) >= n_images:
                            raise ValueError
        except ValueError:
            pass
        self.images = images
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
        image = normalize(image)
        image = cv2.resize(image, self.resolution, interpolation=cv2.INTER_AREA)
        img_id = self.imgs_id[index]
        mask = self.masks[index]
        conditions = []
        img = (image * 255).astype(np.uint8)
        blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
        for r in self.r_steps:
            edge = cv2.Canny(blur, r, r+5)
            condition = self.condition_transform(edge)
            for _ in range(self.run_num):
                conditions.append(condition)
        conditions = torch.stack(conditions, dim=0)
        return image, conditions, img_id, mask


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
    # r_steps = [40, 60, 80, 100]
    r_steps = np.linspace(20, 80, 16)
    # r_steps = [100, 130]
    condition_generator = ConditionGenerator(
        data_dir=args.data_dir,
        ann_dir=args.data_dir,
        resolution=unet.sample_size,
        r_steps=r_steps,
        run_num=args.run_num,
        n_images=float('inf'),
    )

    # sample
    for image, conditions, img_id, mask in condition_generator:
        # image = cv2.resize(image, (unet.sample_size, unet.sample_size))
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
            plt.figure(figsize=(9, 6))
            plt.subplot(231)
            plt.imshow(image, cmap='gray')
            plt.subplot(232)
            plt.imshow(mask, cmap='gray')
            plt.subplot(233)
            plt.imshow(condition[0], cmap='gray')
            for i in range(sample.shape[0]):
                plt.subplot(2, 3, i+4)
                plt.imshow(sample[i], cmap='gray')
            plt.savefig(f"{args.output_dir}/{img_id}_{r+1}.png", bbox_inches='tight', pad_inches=0.1)
            plt.close()
