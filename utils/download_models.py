from huggingface_hub import hf_hub_download
import gdown
import os


hf_hub_download(
    repo_id="xinsir/controlnet-union-sdxl-1.0",
    filename="config.json",
    local_dir="./models/controlnet-union-sdxl-1.0",
)

hf_hub_download(
    repo_id="xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model.safetensors",
    local_dir="./models/controlnet-union-sdxl-1.0",
)

hf_hub_download(
    repo_id="thibaud/controlnet-openpose-sdxl-1.0",
    filename="config.json",
    local_dir="./models/controlnet-openpose-sdxl-1.0",
)

hf_hub_download(
    repo_id="thibaud/controlnet-openpose-sdxl-1.0",
    filename="diffusion_pytorch_model.bin",
    local_dir="./models/controlnet-openpose-sdxl-1.0",
)


# download antelopev2
gdown.download(url="https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing", output="./models/", quiet=False, fuzzy=True)
# unzip antelopev2.zip
os.system("unzip ./models/antelopev2.zip -d ./models/")