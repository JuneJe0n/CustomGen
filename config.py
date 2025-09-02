from pathlib import Path
from utils import PromptGenerator

# img paths
FACE_IMG  = Path("/data2/jeesoo/FFHQ/00000/00062.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/pose/adult_upper/au_16.jpg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/style/s4.png")
OUTDIR    = Path("/data2/jiyoon/custom/results/method5_ablation/ablation_4")
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