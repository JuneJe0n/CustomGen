from pathlib import Path
from utils import PromptGenerator

# img paths
FACE_IMG  = Path("/data2/jiyoon/custom/data/ablation/face/baby/00000.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/ablation/pose/baby/b_0.jpeg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/ablation/style/wikiart_035.jpg")
OUTDIR    = Path("/data2/jiyoon/custom/results/method5_ablation/ablation_5/00000_b_0_wikiart_035")
OUTDIR.mkdir(parents=True, exist_ok=True)

# prompts
generator = PromptGenerator()
PROMPT = generator.generate_combined_prompt(FACE_IMG, POSE_IMG)
NEG = "(lowres, bad quality, watermark,strange limbs)"

# model paths
CN_HED     = "/data2/jiyoon/custom/ckpts/controlnet-union-sdxl-1.0"
CN_POSE    = "/data2/jiyoon/custom/ckpts/controlnet-openpose-sdxl-1.0"
BASE_SDXL  = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE_ENC  = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP   = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

# params
COND_HED     = 0.8
COND_POSE    = 0.85
STYLE_SCALE  = 0.8
CFG, STEPS   = 7.0, 50
SEED         = 42
