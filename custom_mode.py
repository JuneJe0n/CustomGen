import argparse, torch, cv2, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
)
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import MidasDetector, OpenposeDetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL

# ────────────── 고정 경로 ─────────────────────────────────────────
BASE_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
CN_KPS    = "/data2/jiyoon/custom/ckpts/controlnet-openpose-sdxl-1.0"
CN_DEPTH  = "diffusers/controlnet-depth-sdxl-1.0-small"

STYLE_ENC = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP  = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

FACE_IMG  = "/data2/jeesoo/FFHQ/00000/00000.png"
POSE_IMG  = "/data2/jiyoon/InstantID/test/pose/p2.jpeg"
STYLE_IMG = "/data2/jiyoon/StyleID/test/style/s1.png"

OUTDIR    = Path("/data2/jiyoon/custom/results/mode/2/kps"); OUTDIR.mkdir(parents=True, exist_ok=True)
# ────────────────────────────────────────────────────────────────
PROMPT, NEG = "a baby", "(lowres, bad quality, watermark)"
SEED, STEPS = 42, 30
DEV, DTYPE  = "cuda:2", torch.float16
torch.manual_seed(SEED)

# ───── 헬퍼 ──────────────────────────────────────────────────────
def load_rgb(p): return Image.open(p).convert("RGB")
def to_res(im, base=64, short=1024, long=1280):
    w,h = im.size
    r = short/min(w,h);  w,h = int(w*r), int(h*r)
    r =  long/max(w,h);  w,h = int(w*r), int(h*r)
    return im.resize(((w//base)*base, (h//base)*base), Image.BILINEAR)

def depth_to_rgb(arr:np.ndarray) -> Image.Image:
    arr = (arr-arr.min())/(arr.max()-arr.min()+1e-8)
    g   = (arr*255).astype("uint8")
    return Image.fromarray(np.repeat(g[...,None], 3, -1))

# ───── 메인 ─────────────────────────────────────────────────────
def run(ctrl:str, use_style:bool):
    """
    ctrl  : none | kps | depth | both
    style : True  →  IP-Adapter 사용
    """
    # --- detectors -----------------------------------------------------
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(DEV)
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(DEV)

    # --- input images --------------------------------------------------
    pose_im  = to_res(load_rgb(POSE_IMG))
    style_im = load_rgb(STYLE_IMG)

    # --- ControlNet maps ----------------------------------------------
    kps_pil   = openpose(pose_im)
    depth_raw = midas(pose_im)
    depth_pil = depth_raw if isinstance(depth_raw, Image.Image) else depth_to_rgb(depth_raw)
    # 디버그 저장
    kps_pil.save(OUTDIR/"dbg_kps.png"); depth_pil.save(OUTDIR/"dbg_depth.png")

    # --- ControlNet 목록 & 입력 ----------------------------------------
    nets, imgs, scales = [], [], []
    if ctrl in ("kps", "both"):
        nets.append(ControlNetModel.from_pretrained(CN_KPS, torch_dtype=DTYPE).to(DEV))
        imgs.append(kps_pil)
        scales.append(1.2)
    if ctrl in ("depth", "both"):
        nets.append(ControlNetModel.from_pretrained(CN_DEPTH, torch_dtype=DTYPE).to(DEV))
        imgs.append(depth_pil)
        scales.append(0.7)

    # --- 파이프라인 ----------------------------------------------------
    if ctrl == "none":
        pipe = StableDiffusionXLPipeline.from_pretrained(
                    BASE_SDXL, torch_dtype=DTYPE, add_watermarker=False
               ).to(DEV)
        extra = {}
    else:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    BASE_SDXL,
                    controlnet = nets[0] if len(nets)==1 else MultiControlNetModel(nets),
                    torch_dtype=DTYPE, add_watermarker=False
               ).to(DEV)
        extra = {
            "image": imgs,
            "controlnet_conditioning_scale": scales[0] if len(scales)==1 else scales
        }

    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    # --- IP-Adapter (옵션) --------------------------------------------
    if use_style:
        ip = IPAdapterXL(
                pipe,
                image_encoder_path = STYLE_ENC,
                ip_ckpt            = STYLE_IP,
                device             = DEV,
                target_blocks      = ["up_blocks.0.attentions.1"]
            )
        img = ip.generate(
                pil_image      = style_im,
                prompt         = PROMPT,
                negative_prompt= NEG,
                guidance_scale = 6,
                num_inference_steps = STEPS,
                seed           = SEED,
                **extra
              )[0]
    else:
        img = pipe(
                prompt              = PROMPT,
                negative_prompt     = NEG,
                num_inference_steps = STEPS,
                **extra
              ).images[0]

    # --- save ----------------------------------------------------------
    mode_tag = f"{ctrl}{'_style' if use_style else ''}"
    out = OUTDIR/f"{mode_tag}.png"
    img.save(out); print("✔ saved →", out)

# ───── cli ──────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ctrl", choices=["none","kps","depth","both"], required=True,
                   help="which ControlNet(s) to use")
    p.add_argument("--style", action="store_true",
                   help="enable IP-Adapter-XL style injection")
    args = p.parse_args()
    run(args.ctrl, args.style)
