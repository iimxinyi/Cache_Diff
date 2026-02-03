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

### 1.2 Environment Setup for FLUX.2 [klein]
```
conda create --name FLUX.2 python=3.10
```

```
conda activate FLUX.2
```

```
pip install
```

Due to the current unavailability of the `Flux2KleinPipeline` in the stable PyPI release of the diffusers library, please install it directly from the source repository using: `pip install git+https://github.com/huggingface/diffusers`.
```

```
