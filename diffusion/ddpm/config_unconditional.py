from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_dir = 'data/CTSpine/verse20'
    data_type = 'ct'
    output_dir = 'ckpts/ct_ddpm_official'
    logger = "tensorboard"
    logging_dir = "logs"

    # checkpoint
    resume_from_checkpoint = ""
    checkpointing_steps = 20000
    checkpoints_total_limit = None
    
    # input and output
    image_size = 320  # the generated image resolution
    in_channels = 1
    out_channels = 1

    # ddpm
    ddpm_num_steps = 1000
    ddpm_beta_schedule = "linear"
    ddpm_num_inference_steps = 50

    # train args
    num_epochs = 400
    save_images_epochs = 20
    save_model_epochs = 20
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation

    # optimizer
    adam_beta1 = 0.95
    adam_beta2 = 0.999
    adam_weight_decay = 1e-6
    adam_epsilon = 1e-08
    lr_scheduler = "cosine"
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision

     # # UNet parameter
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

    # others
    use_ema = False
    prediction_type = "epsilon"
    model_config_name_or_path = None
