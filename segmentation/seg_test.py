from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import os
import sys
from tqdm import tqdm
sys.path.append('.')
from torch.utils.data import DataLoader
from models.unet import EvidentialUNet
from models.model_tools import load_model

from utils.seed import set_seed
from dataset.vs_dataset import VSDataset
from dataclasses import dataclass
from segmentation.utils.metrics import cal_dice
from rsa.utils import mask_mean
from utils.assd import cal_assd

@dataclass
class TestConfig:
    # path
    data_dir = "data/VS/T2"
    seed = 3
    output_dir = None

    # ckpt_path = 'checkpoints/ablation_consistency_2/thresh07/ckpt/best_val.pth.tar'
    # ckpt_path = 'checkpoints/ablation_uncertainty/thresh002/ckpt/best_val.pth.tar'
    # ckpt_path = 'checkpoints/ablation_N/N8/ckpt/best_val.pth.tar'
    ckpt_path = 'checkpoints/ablation_N/N2_seed125/ckpt/best_val.pth.tar'
    # # SFDA
    # ckpt_path = 'checkpoints/SFDA_seg/seed_3_2/ckpt/best_val.pth.tar'
    # ckpt_path = 'checkpoints/SFDA_seg/seed_3_1/ckpt/best_val.pth.tar'
    # ckpt_path = f'checkpoints/SFDA_seg/seed_{seed}/ckpt/best_val.pth.tar'

    # # Source only
    # ckpt_path = f'checkpoints/source_seg/seed_{seed}/ckpt/best_val.pth.tar'

    # # Target supervised
    # ckpt_path = f'checkpoints/target_supervised/seed_{seed}/ckpt/best_val.pth.tar'

    # model
    n_class = 1
    inp_size = 320
    inp_channel = 1
    amp = True
    bilinear = False
    batch_size = 8

    device = 1


if __name__ == '__main__':
    # 1. Initialize
    config = TestConfig()
    device = torch.device(f'cuda:{config.device}')
    spacing = [320/384*0.5] * 2
    set_seed(config.seed)
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    # 2. Create dataset
    test_ds = VSDataset(data_dir=config.data_dir, split='testing', resolution=config.inp_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=True, num_workers=1, drop_last=False)
    
    # 3. Prepare model
    model = EvidentialUNet(n_channels=config.inp_channel, n_classes=config.n_class, bilinear=config.bilinear)
    model = load_model(model, config.ckpt_path)
    model.to(device=device)
    model.eval()
    
    # 4. Iterate 
    num_val_batches = len(test_loader)
    n = 0
    eval_rets = []
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=config.amp):
        pbar = tqdm(test_loader, total=num_val_batches, desc='Test round', unit='batch')
        for index, (images, masks_true, imgs_id) in enumerate(pbar):
            images = images.to(device=device)
            masks_true = masks_true.to(device=device)
            # predict the mask
            masks_pred, v, alpha, beta = model(images)
            un_maps = beta / (v * (alpha - 1))
            masks_pred = (F.sigmoid(masks_pred) > 0.5).float() 
            
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().detach().numpy().squeeze()
            masks_true = masks_true.cpu().detach().numpy().squeeze()
            masks_pred = masks_pred.cpu().detach().numpy().squeeze()
            un_maps = un_maps.cpu().detach().numpy().squeeze()
            for i in range(images.shape[0]):
                img = images[i]
                gt = masks_true[i]
                img_id = imgs_id[i]
                pred = masks_pred[i]
                un_map = un_maps[i]

                dice = cal_dice(pred, gt)
                assd = cal_assd(pred, gt, spacing)
                eval_rets.append([dice, assd])

                un_map = (un_map - un_map.min()) / (un_map.max() - un_map.min())
                un = mask_mean(un_map, un_map>0.0001)

                if config.output_dir is not None:
                    plt.figure()
                    plt.subplot(2, 2, 1)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.subplot(2, 2, 2)
                    plt.imshow(gt)
                    plt.axis('off')
                    plt.subplot(2, 2, 3)
                    plt.imshow(pred)
                    plt.title(f'{dice:.2f}')
                    plt.axis('off')
                    plt.subplot(2, 2, 4)
                    plt.imshow(un_map)
                    plt.title(f"{un:.4f}")
                    plt.axis('off')
                    plt.savefig(f"{config.output_dir}/{img_id}.png")
                    plt.close()

    eval_rets = np.array(eval_rets)
    dice, assd = np.mean(eval_rets, axis=0)
    print(f"dice={dice:.04}, assd={assd:.04f}mm")    
    
    