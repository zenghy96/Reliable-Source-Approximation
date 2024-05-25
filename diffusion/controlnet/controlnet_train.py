import matplotlib.pyplot as plt
import os

import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from diffusers import DDPMScheduler
import torch
from models.UNet2DModel import UNet2DModel
from models.controlnet import ControlNetModel
from models.pipeline_controlnet import DDPMControlNetPipeline
from dataset import load_dataset
from config import TrainingConfig
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm import tqdm
import torch.nn.functional as F
import math
import sys
sys.path.append('.')
import logging


logger = get_logger(__name__)


def main():
    # torch.set_num_threads(1)
    config = TrainingConfig()
    logging_dir = os.path.join(config.output_dir, "logs")

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_project_config,
        log_with="tensorboard",
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # data
    train_dataloader, test_dataloader = load_dataset(
        data_dir=config.data_dir,
        ann_dir=config.ann_dir,
        data_type=config.data_type,
        batch_size=config.train_batch_size,
        image_size=config.image_size,
    )
    logger.info(f"train on {len(train_dataloader)*config.train_batch_size} images, \
                validation on {len(test_dataloader)*config.train_batch_size} images.")

    # model saving function
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            i = len(weights) - 1
            while len(weights) > 0:
                weights.pop()
                model = models[i]
                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                i -= 1

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()
            # load diffusers style into model
            load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # preapre model
    noise_scheduler = DDPMScheduler.from_pretrained(config.ckpt_ddpm, subfolder='scheduler')
    unet = UNet2DModel.from_pretrained(config.ckpt_ddpm, subfolder='unet')
    if config.ckpt_controlnet is None:
        controlnet = ControlNetModel.from_unet(unet)
        logger.info("Initializing controlnet weights from unet")
    else:
        controlnet = ControlNetModel.from_pretrained(config.ckpt_controlnet)
        logger.info("Loading existing controlnet weights")
    unet.requires_grad_(False)
    controlnet.train()

    # optimizer
    params_to_optimize = controlnet.parameters()
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    # lr scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        # num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        # num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    # prepare
    if accelerator.is_main_process:
        accelerator.init_trackers('train')
    controlnet, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    global_step = 0
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)

    # Now you train the model
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)*config.train_batch_size*accelerator.num_processes}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # resume
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    
    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, config.num_train_epochs):
        for step, (images, conditions, imgs_id) in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Sample noise to add to the images
                noise = torch.randn(images.shape).to(images.device)
                bs = images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=images.device
                ).long()
                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

                # ControNet output
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_images,
                    timesteps,
                    controlnet_cond=conditions,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_images,
                    timesteps,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # calulate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, 1)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # After each epoch you sample all validation images with evaluate() and save the model
            if accelerator.is_main_process:
                if global_step % config.save_ckpt_steps == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                
                if global_step % config.validation_steps == 0:
                    validation_loop(config, global_step, unet, controlnet, noise_scheduler, accelerator, test_dataloader)

                if global_step % config.save_model_steps == 0 or global_step == config.max_train_steps:
                    controlnet = accelerator.unwrap_model(controlnet)
                    controlnet.save_pretrained(os.path.join(config.output_dir, "controlnet"))
    
    accelerator.end_training()


def validation_loop(config, step, unet, controlnet, noise_scheduler, accelerator, test_dataloader):
    pipeline = DDPMControlNetPipeline(
        contronet=accelerator.unwrap_model(controlnet),
        unet=accelerator.unwrap_model(unet), 
        scheduler=noise_scheduler,
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
    
    all_images, all_conditions, all_samples = [], [], []
    for images, conditions, img_ids in test_dataloader:
        samples = pipeline(
            conditions,
            generator=generator,
            output_type='numpy',
            num_inference_steps=50,
        )[0]
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        conditions = (conditions / 2 + 0.5).clamp(0, 1)
        conditions = conditions.cpu().permute(0, 2, 3, 1).numpy()

        all_images.append(images)
        all_conditions.append(conditions)
        all_samples.append(samples)

    all_images = np.concatenate(all_images, axis=0)
    all_conditions = np.concatenate(all_conditions, axis=0)
    all_samples = np.concatenate(all_samples, axis=0)
    for tracker in accelerator.trackers:
        tracker.writer.add_images("origin", all_images, step, dataformats="NHWC")
        tracker.writer.add_images("condition", all_conditions, step, dataformats="NHWC")
        tracker.writer.add_images("generated", all_samples, step, dataformats="NHWC")

    

if __name__ == "__main__":
    main()