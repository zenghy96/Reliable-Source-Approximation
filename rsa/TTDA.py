import json
import scipy.io as sio
import os
from diffusers import DDPMScheduler
import numpy as np
import torch
from torchvision import transforms as T
import sys
import argparse
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm
sys.path.append('.')
# ## diffusion
from diffusion.controlnet.models.UNet2DModel import UNet2DModel
from diffusion.controlnet.models.controlnet import ControlNetModel
from diffusion.controlnet.models.pipeline_controlnet import DDPMControlNetPipeline
from rsa.dataset.ttda_dataset import TargetDataset
from rsa.utils import cal_var, set_seed, find_best
from segmentation.utils.metrics import cal_dice
from segmentation.utils.assd import cal_assd
from segmentation.models.unet import EvidentialUNet
from segmentation.models.model_tools import load_model, save_model
from segmentation.models.evidence_loss import calculate_evidential_loss
from segmentation.models.dice_score import dice_loss
import argparse

# @dataclass
# class args:
#     data_dir = 'data/VS/T2'
#     split = 'testing'
#     output_dir = 'outputs/final_ttda/0'
#     ddpm_ckpt_dir = 'checkpoints/vs_ddpm'
#     controlnet_ckpt_dir = 'checkpoints/vs_controlnet'
#     seg_ckpt_dir ='checkpoints/source_seg/seed_3/ckpt/best_val.pth.tar'

#     var_thresh = 0.3
#     epoch_n = 1
#     batch_size = 16

#     # r_steps = [30, 40, 50, 60]
#     # r_steps = [30, 40, 50, 60]
#     r0, r1, step_n = 30, 80, 2
#     r_steps = np.linspace(float(r0), float(r1), int(step_n))
#     run_num = 2
#     num_inference_steps = 50
#     seed = 0
#     device = 1
#     amp = True

def get_args():
    parser = argparse.ArgumentParser(description="命令行参数示例")
    # 添加参数
    parser.add_argument('--data_dir', type=str, default='data/VS/T2')
    parser.add_argument('--split', type=str, default='testing')
    parser.add_argument('--output_dir', type=str, default='outputs/final_ttda/seed_3')
    parser.add_argument('--ddpm_ckpt_dir', type=str, default='checkpoints/vs_ddpm')
    parser.add_argument('--controlnet_ckpt_dir', type=str, default='checkpoints/vs_controlnet')
    parser.add_argument('--seg_ckpt_dir', type=str, default='checkpoints/source_seg/seed_125/ckpt/best_val.pth.tar')
    parser.add_argument('--var_thresh', type=float, default=0.3)
    parser.add_argument('--epoch_n', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--r0', type=float, default=30)
    parser.add_argument('--r1', type=float, default=80)
    parser.add_argument('--step_n', type=int, default=2)
    parser.add_argument('--run_num', type=int, default=2)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--amp', action='store_true')  # 默认为False，如果指定了该参数则为True
    # 解析参数
    args = parser.parse_args()
    # 构造日志文件的完整路径
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "log.txt")
    # 将参数记录到日志文件
    with open(log_file_path, "w") as log_file:
        for arg, value in vars(args).items():
            log_file.write(f"{arg}: {value}\n")

    return args


if __name__ == "__main__":
    args = get_args()
    sample_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}")
    set_seed(args.seed)
    spacing = [320/384*0.5] * 2

    # diffusion model
    # noise_scheduler = DDPMScheduler.from_pretrained(args.ddpm_ckpt_dir, subfolder='scheduler')
    # unet = UNet2DModel.from_pretrained(args.ddpm_ckpt_dir, subfolder='unet')
    # controlnet = ControlNetModel.from_pretrained(args.controlnet_ckpt_dir, subfolder='controlnet')
    # unet.to(device)
    # controlnet.to(device)
    # pipeline = DDPMControlNetPipeline(
    #     contronet=controlnet,
    #     unet=unet,
    #     scheduler=noise_scheduler,
    #     use_bar=False
    # )
    # unet model
    LockUNet = EvidentialUNet(n_channels=1, n_classes=1)
    LockUNet = load_model(LockUNet, args.seg_ckpt_dir)
    LockUNet.to(device=device)
    LockUNet.eval()

    FreeUNet = EvidentialUNet(n_channels=1, n_classes=1)
    FreeUNet = load_model(FreeUNet, args.seg_ckpt_dir)
    FreeUNet.to(device=device)

    # data
    r_steps = np.linspace(args.r0, args.r1, args.step_n)
    print(f'steps: {r_steps}')
    target_ds = TargetDataset(
        data_dir=args.data_dir,
        split=args.split,
        resolution=320,
        r_steps=r_steps,
        run_num=args.run_num,
    )
    target_loader = DataLoader(
        target_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1, 
        drop_last=False
    )
    transforms = T.Compose([T.ToTensor(), T.Normalize(mean=[.5], std=[.5])])

    # optimizer
    optimizer = optim.Adam(FreeUNet.parameters(), lr=1e-4)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = torch.nn.BCEWithLogitsLoss()

    # sample
    record = {}
    for epoch in range(1, args.epoch_n+1):
        json_path = f'{args.output_dir}/record_{epoch}.json'
        for batch_id, (images, conditions, imgs_id, masks_true) in enumerate(tqdm(target_loader)):
            # Prediction using current model
            batch_id = str(batch_id)
            inp = images.to(device)
            FreeUNet.eval()
            xt_preds, v, alpha, beta = FreeUNet(inp)
            xt_preds = (F.sigmoid(xt_preds.cpu()) > 0.5).float().cpu().detach().numpy().squeeze()
            test_dice = []
            test_assd = []
            masks_true = masks_true.numpy().squeeze()
            for pred, gt in zip(xt_preds, masks_true):
                test_dice.append(cal_dice(pred, gt))
                test_assd.append(cal_assd(pred, gt, spacing))
            record[batch_id] = {'test_dice': test_dice}
            # print(f'batch_id ={batch_id}, dice={sum(test_dice)/len(test_dice):.04f}')
            
            # process each image in batch for pseudo labels
            pseudo_label_stack = []
            target_data_stack = []
            source_like_stack = []
            quality = []
            for i, conds in enumerate(conditions):
                # Imgae translation
                samples = sio.loadmat(f'outputs/final_ttda/0/samples/{imgs_id[i]}')['samples']
                # conds = conds.to(device)
                # samples = pipeline(
                #     conds,
                #     num_inference_steps=args.num_inference_steps,
                #     generator=torch.manual_seed(0),
                #     output_type='numpy',
                # )[0].squeeze()
                # sio.savemat(f'{sample_dir}/{imgs_id[i]}.mat', {'samples': samples})

                # predict
                inp = [transforms(sample) for sample in samples]
                inp = torch.stack(inp).to(device)
                sample_preds, v, alpha, beta = LockUNet(inp)
                sample_preds = (F.sigmoid(sample_preds) > 0.5).float() 
                sample_preds = sample_preds.cpu().detach().numpy().squeeze()
                sample_preds = np.array_split(sample_preds, args.step_n, axis=0)
                samples = np.array_split(samples, args.step_n, axis=0)

                # select
                gt = masks_true[i].squeeze()
                all_var = np.ones((args.step_n,))
                for j, preds in enumerate(sample_preds):
                    all_var[j] = cal_var(preds)
                best_var, best_pred, best_step, best_run = \
                        find_best(all_var, args.var_thresh, sample_preds)

                if best_pred is not None:
                    dice = cal_dice(best_pred, gt)
                    quality.append(dice)
                    best_sample = samples[best_step][best_run]
                    pseudo_label_stack.append(best_pred)
                    target_data_stack.append(images[i])
                    source_like_stack.append(best_sample)
                    # print(f'i={i}: {dice:.04f}')
                # dice_cur.append(cal_dice(pred, gt).item())
                # dice_cur = [f"{dice:.4f}" for dice in dice_cur]
                # print(f'i={i}: {dice_cur}')
            record[batch_id] = {'test_dice': test_dice,'test_assd': test_assd, 'quality': quality}

            # Update free model:
            if len(pseudo_label_stack) > 0:
                # data
                source_like_stack = [transforms(data) for data in source_like_stack]
                pseudo_label_stack = pseudo_label_stack * 2
                inp = target_data_stack + source_like_stack
                inp = torch.stack(inp).to(device)
                pseudo_labels = [torch.from_numpy(label).unsqueeze(dim=0) for label in pseudo_label_stack]
                pseudo_labels = torch.stack(pseudo_labels).to(device)
                # train
                FreeUNet.train()
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                    preds, v, alpha, beta = FreeUNet(inp)
                    loss1 = criterion(preds, pseudo_labels)
                    preds = F.sigmoid(preds)
                    loss2 = dice_loss(preds, pseudo_labels)
                    loss3 = calculate_evidential_loss(pseudo_labels, preds, v, alpha, beta)
                    # loss = loss1 + loss2 + loss3*args.lamda
                    loss = loss1 + loss2 + loss3
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(FreeUNet.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

            with open(json_path, 'w') as f:
                json.dump(record, f, indent=4)
        
        dice_avg = []
        assd_avg = []
        for k, v in record.items():
            dice = v['test_dice']
            assd = v['test_assd']
            dice_avg.extend(dice)
            assd_avg.extend(assd)
        dice_avg = np.array(dice_avg)
        print(f'dice_avg = {dice_avg.mean():.04f}')
        assd_avg = np.array(assd_avg)
        print(f'assd_avg = {assd_avg.mean():.04f}')

    # save_model(
    #     path=f"{args.output_dir}/last.pth.tar", 
    #     epoch=epoch, 
    #     model=FreeUNet,
    #     optimizer=optimizer,
    # )
