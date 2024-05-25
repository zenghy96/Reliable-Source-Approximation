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
from dataset.ultra_datasets import _list_image_files_recursively, generate_condition
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T


parser = argparse.ArgumentParser()
parser.add_argument(
        "--data_dir",
        type=str,
        default='datasets/ClariusDetData/data',
        help="The directory where the original data are stored.",
)
parser.add_argument(
        "--output_dir",
        type=str,
        default='outputs/sample_pic',
        help="The directory where the generated images will be stored.",
)
parser.add_argument(
        "--sample_save_dir",
        type=str,
        default='/home/data/zenghy/ttda',
        help="The directory where the generated images will be stored.",
)
parser.add_argument(
        "--unet_ckpt_dir",
        type=str,
        default='ckpt/ddpm_0',
        help="The directory where the unet model and scheduler are stored.",
)
parser.add_argument(
        "--controlnet_ckpt_dir",
        type=str,
        default='ckpt/controlnet_0',
        help="The directory where the controlnet model are stored.",
)
parser.add_argument(
        "--r_steps",
        type=str,
        default='0.3, 0.7, 4',
        help="The directory where the controlnet model are stored.",
)
parser.add_argument(
        "--run_num",
        type=int,
        default=4,
        help="Number of runs for an image",
)
parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
)
parser.add_argument("--seed", type=int, default=0)


class ConditionGenerator(Dataset):
    def __init__(self, image_paths, resolution, r_steps, run_num) -> None:
        self.image_paths = image_paths
        self.r_steps = r_steps
        self.run_num = run_num
        self.condition = torch
        self.condition_transform = T.Compose([
                T.ToTensor(),
                T.Resize((resolution, resolution), antialias=False),
        ])

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img_id = image_path.split('/')[-1].split('.')[0]
        image = sio.loadmat(image_path)['img'].astype(np.uint8)
        img_max = image.max()

        conditions = []
        for r in self.r_steps:
            T = img_max * r
            _, thresh = cv2.threshold(image, T, 1, cv2.THRESH_BINARY)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            condition = self.condition_transform(opening)
            for _ in range(self.run_num):
                conditions.append(condition)
        conditions = torch.stack(conditions, dim=0)
        return image, conditions, img_id


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
    r0, r1, n = args.r_steps.split(',')
    r_steps = np.linspace(float(r0), float(r1), int(n))
    image_paths = _list_image_files_recursively(args.data_dir)
    condition_generator = ConditionGenerator(
        image_paths=image_paths,
        resolution=unet.sample_size,
        r_steps=r_steps,
        run_num=args.run_num,
    )

#     ids = [
#         # '2021-01-30-16-35-20-3010-2_190',
#         '2021-01-30-13-30-40-3006_090',
#         '2021-01-30-13-30-40-3006_130',
#         # '2021-01-30-13-30-40-3006_000',
#         # '2021-01-30-13-30-40-3006_230',
#         # '2021-02-27-17-02-21_350',
#         '2021-01-30-14-40-18-3008_190',
#     ]
    
    # sample
    for image, conditions, img_id in condition_generator:
        # if img_id not in ids:
        #     continue 
        image = cv2.resize(image, (unet.sample_size, unet.sample_size))
        conditions = conditions.to(device)
        samples = pipeline(
            conditions,
            num_inference_steps=args.num_inference_steps,
            generator=torch.manual_seed(args.seed),
            output_type='numpy',
        )[0]
        samples = samples.squeeze()

        # save samples
        sio.savemat(f"{args.sample_save_dir}/{img_id}.mat", {f'sample': samples})

        # plot image, conditions and samples
        samples = np.array_split(samples, r_steps.shape[0], axis=0)
        conditions = conditions.cpu().numpy().squeeze()
        conditions = np.array_split(conditions, r_steps.shape[0], axis=0)          
        for r, (condition, sample) in enumerate(zip(conditions, samples)):
            plt.figure()
            plt.subplot(231)
            plt.imshow(image)
            plt.subplot(232)
            plt.imshow(condition[0])
            for i in range(sample.shape[0]):
                plt.subplot(2, 3, i+3)
                plt.imshow(sample[i])
            plt.savefig(f"{args.output_dir}/{img_id}_{r+1}.png")
            plt.close()
