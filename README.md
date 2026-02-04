# Cache_Diff


### 1.1 Environment Setup for Stable Diffusion 3.5 Large
```
conda create --name SD3.5 python=3.10
```

```
conda activate SD3.5
```

```
pip install torch==2.9.0
pip install diffusers==0.36.0
pip install transformers==5.0.0
pip install accelerate==1.12.0
pip install protobuf==6.33.5
pip install sentencepiece==0.2.1
pip install torchvision==0.24.0
pip install openpyxl==3.1.5
```

### 1.2 Environment Setup for FLUX.1 [dev]
```
conda create --name FLUX.1 python=3.10
```

```
conda activate FLUX.1
```

```
pip install torch==2.8.0
pip install torchvision==0.23.0
pip install einops==0.8.1
pip install transformers==4.56.1
pip install safetensors==0.4.5
pip install fire==0.7.1
pip install openai==2.8.1
pip install accelerate==1.12.0
pip install ruff==0.6.8
pip install sentencepiece==0.2.1
pip install protobuf==6.33.5
pip install invisible-watermark==0.2.0
pip install ruff==0.15.0
```
