from pathlib import Path

# prompts
PROMPT = "Woman, clear facial features, smooth colors"
NEG = "(lowres, bad quality, watermark,strange limbs)"

# img paths
FACE_IMG  = Path("/data2/jeesoo/FFHQ/00000/00020.png")
POSE_IMG  = Path("/data2/jeesoo/custom_gen_js/pose_data/adult2.jpg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/style/s4.png")
OUTDIR    = Path("/home/jiyoon/adult2")
OUTDIR.mkdir(parents=True, exist_ok=True)

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