from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_dir = '/raid/zeng_hongye/data/CT_COLONOGRAPHY/CT'
    data_type = 'ct'
    output_dir = 'ckpts/ct_ddpm_'

    image_size = 320  # the generated image resolution
    in_channels = 1
    out_channels = 1

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

    # train args
    train_batch_size = 20
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 50
    save_model_epochs = 50
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    seed = 0
