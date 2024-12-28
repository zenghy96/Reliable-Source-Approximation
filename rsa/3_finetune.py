from re import S
import torch.nn.functional as F
import torch
import os
import tensorboardX
from torch import optim
import sys
import torch.nn as nn
from tqdm import tqdm
sys.path.append('.')
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import argparse

from segmentation.models.unet import EvidentialUNet
from segmentation.models.model_tools import load_model
from segmentation.models.dice_score import dice_loss
from segmentation.models.evidence_loss import calculate_evidential_loss
from segmentation.models.model_tools import save_model
from segmentation.utils.validation import validate
from segmentation.utils.seed import set_seed
from segmentation.utils.logger import Logger
from segmentation.dataset.vs_dataset import VSDataset
from dataset.train_dataset import GoBackDataset
from dataset.sampler import TwoStreamBatchSampler


def get_args():
    parser = argparse.ArgumentParser(description="Parser for training settings")
   # Directories and file paths
    parser.add_argument('--data_dir', type=str, default="data/VS/T2", help='Path to the data domain directory')
    parser.add_argument('--sample_dir', type=str, default="outputs/final_0/translated", help='Path to the translated data directory')
    parser.add_argument('--save_dir', type=str, default="checkpoints/SFDA_seg/seed_3", help='Path to save checkpoints')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/source_seg/seed_3/ckpt/best_val.pth.tar", help='Path to saved checkpoints')
    # parser.add_argument('--checkpoint', type=str, default="", help='Path to saved checkpoints')

    # Model parameters
    parser.add_argument('--n_class', type=int, default=1, help='Number of classes')
    parser.add_argument('--inp_channel', type=int, default=1, help='Input channels')
    parser.add_argument('--lamda', type=float, default=0.5, help='Evidential loss')
    parser.add_argument('--inp_size', type=float, default=320, help='Evidential loss')   

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_epoch', type=int, default=150, help='Number of epochs')
    parser.add_argument('--val_fre', type=int, default=10, help='Frequency of validation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--bilinear', type=bool, default=False, help='Use bilinear upsampling')
    parser.add_argument('--amp', type=bool, default=True, help='Use automatic mixed precision')

    # Hardware settings
    parser.add_argument('--device', type=int, default=0, help='Device ID')
    parser.add_argument('--seed', type=int, default=3, help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    # 1. Initialize logging
    args = get_args()
    logger = Logger(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}')
    ckpt_dir = f"{args.save_dir}/ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. Create dataset
    # training data includes translated data, use pseudo labels
    train_ds = GoBackDataset(data_dir=args.data_dir, split='training', sample_dir=args.sample_dir, resolution=args.inp_size)
    batch_sampler = TwoStreamBatchSampler(train_ds.target_idx, train_ds.translated_idx, args.batch_size, 1)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=2)
    # validation only include T2, use annotations
    val_ds = VSDataset(data_dir=args.data_dir, split='validation', resolution=args.inp_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False)

    # 3. Set up the model, the optimizer, the loss, the learning 
    # rate scheduler and the loss scaling for AMP
    model = EvidentialUNet(n_channels=args.inp_channel, n_classes=args.n_class, bilinear=args.bilinear)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.checkpoint:
        model, optimizer, start_epoch = load_model(model, args.checkpoint, optimizer=optimizer)
    model.to(device=device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    # 4. Begin training
    best_val_score = 0
    for epoch in range(1, args.n_epoch + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.n_epoch}', unit='batch')
        for images, true_masks, _ in train_loader:
            images = images.to(device=device)
            true_masks = true_masks.to(device=device)
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                pred, v, alpha, beta = model(images)
                loss1 = criterion(pred, true_masks)
                pred = F.softmax(pred, dim=1) if model.n_classes > 1 else F.sigmoid(pred)
                loss2 = dice_loss(pred, true_masks)
                loss3 = calculate_evidential_loss(true_masks, pred, v, alpha, beta)
                loss = loss1 + loss2 + loss3*args.lamda
                logging_dict = {'loss': loss.item(), 'loss1': loss1.item(), 'loss2': loss2.item(), 'loss3': loss3.item()}
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            logging_dict['lr'] = optimizer.param_groups[0]['lr']
            pbar.set_postfix(**logging_dict)
            pbar.update(1)

        pbar.close()
        logging_dict = f'Epoch[{epoch:03d}]: ' + " | ".join([f"{key}: {value:.6f}" for key, value in logging_dict.items()]) + '\n'
        logger.write(logging_dict)
        logger.scalar_summary('train_loss', epoch_loss/len(train_loader), epoch)

        # Evaluation round
        if epoch % args.val_fre == 0 or epoch == args.n_epoch:
            val_score = validate(model, val_loader, device, args.amp)
            txt = f'validation dice = {val_score}\n'
            print(txt, end='')
            scheduler.step(val_score)
            logger.write(txt)
            logger.scalar_summary("val_score", val_score, epoch)
            if val_score > best_val_score:
                save_model(
                    path=f"{ckpt_dir}/best_val.pth.tar", 
                    epoch=epoch, model=model
                )
                best_val_score = val_score

        if epoch == args.n_epoch:
            save_model(
                path=f"{ckpt_dir}/last.pth.tar", 
                epoch=epoch, model=model,
                optimizer=optimizer,
            )
    logger.close()



if __name__ == '__main__':
    main()