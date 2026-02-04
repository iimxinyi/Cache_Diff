import torch
import random
import numpy as np
from diffusers import StableDiffusion3Pipeline

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
pipe = StableDiffusion3Pipeline.from_pretrained(model_directory, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

# Basic parameters
seed = 1
generator = seed_everywhere(seed)
total_step = 50
guidance_scale = 4.5  # 这个默认值是4.5-5.0，用来控制生成的图像与提示词的相关性，越大相关性越强，但是图像质量也会下降。后面需要自己微调找到一个还不错的值，个人建议生成缓存图像的guidance_scale偏大，生成用户请求的新图像的guidance_scale偏大（一般不超过7.0）。

# prompts
prompts = [
    'A graceful cat sitting in a warm and story-rich environment, highlighting its silky fur.',
    ]


# Demo for fixed generation
image = pipe(prompt=prompts[0], num_inference_steps=total_step, guidance_scale=guidance_scale, generator=generator, height=1024, width=1024).images[0]
image.save("./CacheDiff/test.png")

# Demo for random generation
image1 = pipe(prompt=prompts[0], num_inference_steps=total_step, guidance_scale=guidance_scale, height=1024, width=1024).images[0]
image2 = pipe(prompt=prompts[0], num_inference_steps=total_step, guidance_scale=guidance_scale, height=1024, width=1024).images[0]
image3 = pipe(prompt=prompts[0], num_inference_steps=total_step, guidance_scale=guidance_scale, height=1024, width=1024).images[0]
image1.save("./CacheDiff/test1.png")
image2.save("./CacheDiff/test2.png")
image3.save("./CacheDiff/test3.png")
