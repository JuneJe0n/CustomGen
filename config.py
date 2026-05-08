from pathlib import Path

# img paths
FACE_IMG  = Path("/data2/CustomGen/CustomGen/data/face/adult/00020.png")
POSE_IMG  = Path("/data2/CustomGen/CustomGen/data/pose/adult/af_0.png")
STYLE_IMG = Path("/data2/CustomGen/CustomGen/data/style/wikiart_021.jpg")
OUTPUT_DIR="/data2/CustomGen/CustomGen/data"
#OUTDIR.mkdir(parents=True, exist_ok=True)

# prompts
NEG = "(lowres, bad quality, watermark,strange limbs)"

# model paths
CN_HED     = "/data2/CustomGen/CustomGen/models/controlnet-union-sdxl-1.0"
CN_POSE    = "/data2/CustomGen/CustomGen/models/controlnet-openpose-sdxl-1.0"
BASE_SDXL  = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE_ENC  = "/data2/CustomGen/CustomGen/models/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP   = "/data2/CustomGen/CustomGen/models/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

# params
COND_HED     = 0.8
COND_POSE    = 0.85
STYLE_SCALE  = 0.8
CFG, STEPS   = 7.0, 50
SEED         = 42
