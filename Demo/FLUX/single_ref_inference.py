import torch
import random
import numpy as np
from diffusers.utils import load_image
from diffusers import FluxImg2ImgPipeline

# Function to set seed for all random operations
def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return generator



# Local path to FLUX.1[dev]
model_directory = "/root/autodl-tmp/FLUX.1-dev"

# FLUX.1[dev] pipeline
pipe = FluxImg2ImgPipeline.from_pretrained(model_directory, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# Basic parameters
seed = 1
generator = seed_everywhere(seed)
total_step = 50
guidance_scale = 7.0  # 这个默认值是3.5，用来控制生成的图像与提示词的相关性，越大相关性越强，但是图像质量也会下降。后面需要自己微调找到一个还不错的值，个人建议生成缓存图像的guidance_scale偏小，生成用户请求的新图像的guidance_scale偏大（一般不超过7.0）。
height = 1024
width = 1024
ref_image1 = load_image("./CacheDiff/test1.png").resize((height, width))
strength = 0.8  # [0.0, 1.0]

# prompts
ref_prompts = [
    'A graceful cat sitting in a warm and story-rich environment, highlighting its silky fur.',
    ]

gen_prompts = [
    'A fluffy white cat with blue eyes sitting gracefully on a windowsill, bathed in golden sunlight, with a serene garden visible through the window.',
]

# Demo
image = pipe(prompt=gen_prompts[0], image=ref_image1, num_inference_steps=total_step, strength=strength, guidance_scale=guidance_scale, generator=generator, height=height, width=width).images[0]
image.save("./CacheDiff/single_gen_test.png")