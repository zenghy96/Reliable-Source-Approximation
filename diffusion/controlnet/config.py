from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # path
    data_dir = "data/Vestibular-Schwannoma-SEG"
    ann_dir = "data/Vestibular-Schwannoma-SEG"
    data_type = 'mri'
    ckpt_ddpm = 'ckpts/vs_ddpm'
    ckpt_controlnet = None
    output_dir = 'ckpts/vs_controlnet_final_z0'
    resume_from_checkpoint = ""
    
    # input
    image_size = 320  # the generated image resolution
    in_channels = 1
    out_channels = 1

    # optimizer
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08

    # lr
    lr_scheduler = "constant"
    learning_rate = 1e-5
    lr_warmup_steps = 500
    lr_num_cycles = 1
    lr_power = 1.0
    gradient_accumulation_steps = 1
    
    # train
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_train_epochs = 100
    validation_steps = 3000
    save_model_steps = 3000
    save_ckpt_steps = 3000
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision

    seed = 0


    # unet
    layers_per_block = 2
    block_out_channels = (128, 128, 256, 256, 512, 512)
    down_block_types = (
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types = (
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )

    