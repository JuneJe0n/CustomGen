from huggingface_hub import hf_hub_download


repo = "diffusers/controlnet-depth-sdxl-1.0-small"
local_dir = "/data2/jiyoon/customgen/models/controlnet-depth-sdxl-1.0-small"

hf_hub_download(repo, "config.json", local_dir=local_dir)
hf_hub_download(repo, "diffusion_pytorch_model.bin", local_dir=local_dir)