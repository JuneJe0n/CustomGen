<div align="center">
<h2> CustomGen: Unified Multi-Condition Control for Diffusion Models </h2>
	
<h4> If you like our project, please give us a star ⭐ on GitHub. </h4>
</div>

This repository contains the official code implementation of "CustomGen: Towards Training-Free Custom Generation". We provide the inference code reported in the paper.
<p align="center">
  <img src="https://raw.githubusercontent.com/JuneJe0n/CustomGen/master/assets/comparison.png" width="1100">
</p>


<h2>Abstract</h2>

Despite recent advances in diffusion-based image generation, existing approaches lack a unified framework that can simultaneously handle identity preservation, style transfer, pose control, and text guidance. To address this limitation, we propose CustomGen, a lightweight, training-free diffusion framework that unifies image-based control of identity, style, and pose with text guidance in a single generation process. 
CustomGen is composed of three modules: a style extractor for image-based style injection, a prompt generator for effective textual guidance, and a face–pose aligner that fuses multiple ControlNet conditionings.
Through extensive ablation studies, we systematically analyze the characteristics of different ControlNet conditions and derive meaningful insights into how each condition influences the generation process. 
CustomGen adopts a modular Multi-ControlNet integration, remaining lightweight and plug-and-play across diverse diffusion-based models. 
<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/JuneJe0n/CustomGen/master/assets/architecture.png" width="1100">
</p>




<h2>Getting Started</h2>
<h3>1) Clone the repository</h3>

```bash  
git clone https://github.com/JuneJe0n/CustomGen.git
cd CustomGen
```


<h3>2) Environment Setup</h3>

```bash
conda create --name customgen python=3.8.10
conda activate customgen

# Install requirements
pip install -r requirements.txt
```


<h3>3) Download Models</h3>

Follow [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter?tab=readme-ov-file#download-models) to download pre-trained checkpoints from [here](https://huggingface.co/h94/IP-Adapter).

```bash
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/sdxl_models models/IP-Adapter
```

Then, run the following command to download necessary ControlNet and antelopev2 models:
```bash
python utils/download_models.py
```

Once you have prepared all models, the folder tree should be like:
```
  .
  └── models
	  ├── IP-Adapter
	  ├── antelopev2
	  ├── controlnet-union-sdxl-1.0	  
	  └── controlnet-openpose-sdxl-1.0
```




<h2>Usage</h2>
Ensure that the workspace is the root directory of the project. Then run:

```bash
python infer.py
```

<h2>Citation</h2>
If you find CustomGen useful for your research and applications, please cite us using this BibTeX:
