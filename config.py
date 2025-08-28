from pathlib import Path

# --- Config ---
PROMPT = "a boy, sitting, clear facial features, detailed, realistic, smooth colors"
NEG = "(lowres, bad quality, watermark, disjointed, strange limbs, cut off, bad anatomymissing limbs, fused fingers)"
FACE_IMG  = Path("/data2/jeesoo/FFHQ/00000/00086.png")
POSE_IMG  = Path("//data2/jiyoon/custom/data/pose/p1.jpg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/style/s4.png")
OUTDIR       = Path("/data2/jiyoon/custom/results/survey/m6/00086_p1_s4")
OUTDIR.mkdir(parents=True, exist_ok=True)

CN_HED     = "/data2/jiyoon/custom/ckpts/controlnet-union-sdxl-1.0"
CN_POSE    = "/data2/jiyoon/custom/ckpts/controlnet-openpose-sdxl-1.0"
BASE_SDXL  = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE_ENC  = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP   = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

COND_HED     = 1.0
COND_POSE    = 0.6
STYLE_SCALE  = 0.8
CFG, STEPS   = 7.0, 50
SEED         = 42