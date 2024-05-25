import matplotlib.pyplot as plt
import scipy.io as sio
import os
from PIL import Image
from diffusers import DDPMPipeline
import matplotlib as mpl
import numpy as np
import sys


save_dir = 'outputs/zzzzz'
os.makedirs(save_dir, exist_ok=True)

pipeline = DDPMPipeline.from_pretrained("ckpts/vs_ddpm")
pipeline.to('cuda:0')
for i in range(20):
    batch_size = 16
    images = pipeline(num_inference_steps=50, batch_size=batch_size).images
    for j in range(len(images)):
        image = images[j]
        image.save(f"{save_dir}/{i}-{j}.png")

