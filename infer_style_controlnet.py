# canny_face.py
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

import cv2
from PIL import Image

from ip_adapter import IPAdapterXL

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
ip_ckpt = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)

# load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

# style image
image = "/data2/jiyoon/StyleID/test/style/s5.jpg"
image = Image.open(image)
image.resize((512, 512))

# control image
input_image = cv2.imread("/data2/jeesoo/FFHQ/00000/00000.png")
detected_map = cv2.Canny(input_image, 50, 200)
canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)) # convert to PILs

# generate image
images = ip_model.generate(pil_image=image,
                           prompt="a baby eating a sandwich",
                           negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=30, 
                           seed=42,
                           image=canny_map,
                           controlnet_conditioning_scale=0.6,
                          )

images[0].save("/data2/jiyoon/instantstyle/test/results/s5_sandwich.jpg")