import torch
import random
import numpy as np
from diffusers.utils import load_image
from diffusers import StableDiffusion3Img2ImgPipeline

# Function to set seed for all random operations
def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return generator

# Local path to Stable Diffusion 3
model_directory = "/root/autodl-tmp/stable-diffusion-3.5-large-diffusers"

# Stable Diffusion 3 pipeline
pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_directory, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

# Basic parameters
seed = 1
generator = seed_everywhere(seed)
total_step = 50
guidance_scale = 5.0  # 这个默认值是5.0，用来控制生成的图像与提示词的相关性，越大相关性越强，但是图像质量也会下降。后面需要自己微调找到一个还不错的值
height = 1024
width = 1024
ref_image2 = load_image("./CacheDiff/test2.png").resize((height, width))
strength = 0.8  # [0.0, 1.0]

# prompts
ref_prompts = [
    'A graceful cat sitting in a warm and story-rich environment, highlighting its silky fur.',
    ]

gen_prompts = [
    'A fluffy white cat with blue eyes sitting gracefully on a windowsill, bathed in golden sunlight, with a serene garden visible through the window.',
]

# Demo
image = pipe(prompt=gen_prompts[0], image=ref_image2, num_inference_steps=total_step, strength=strength, guidance_scale=guidance_scale, generator=generator, height=1024, width=1024).images[0]
image.save("./CacheDiff/single_gen_test.png")
