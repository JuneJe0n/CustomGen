#!/usr/bin/env python
# custom_hed_facepose_softblend.py  (2025-07-17 rev-G: multi-style image support)

import argparse, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import HEDdetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL

# ─────────────────── 설정 ───────────────────
PROMPT = "a baby sitting, clear facial features, detailed, realistic, smooth colors"
NEG = "(lowres, bad quality, watermark, disjointed, strange limbs, cut off, bad anatomymissing limbs, fused fingers)"
FACE_IMG  = Path("/data2/jiyoon/custom/data/face/00000.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/pose/p2.jpeg")
STYLE_DIR = Path("/data2/jiyoon/wikiart")  # path to your folder

CN_HED     = "/data2/jiyoon/custom/ckpts/controlnet-union-sdxl-1.0"
BASE_SDXL  = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE_ENC  = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP   = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

COND_HED     = 0.8
STYLE_SCALE  = 0.8
CFG, STEPS   = 7.0, 50
SEED         = 42
OUTDIR       = Path("/data2/jiyoon/custom/results/eval")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────── 유틸 ───────────────────
def load_rgb(p): return Image.open(p).convert("RGB")

def to_sdxl_res(img, base=64, short=1024, long=1024):
    w, h = img.size
    r = short / min(w, h); w, h = int(w*r), int(h*r)
    r = long  / max(w, h); w, h = int(w*r), int(h*r)
    return img.resize(((w//base)*base, (h//base)*base), Image.LANCZOS)

# ─────────────────── 메인 ───────────────────
def main(gpu_idx):
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)

    # ───── 얼굴 감지 & HED detector
    face_det = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=gpu_idx, det_size=(640, 640))
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)

    # ───── 이미지 불러오기
    face_im = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im = to_sdxl_res(load_rgb(POSE_IMG))
    w_pose, h_pose = pose_im.size

    # ───── pose에서 얼굴 bbox 추출
    pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
    p_info  = max(face_det.get(pose_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    x1, y1, x2, y2 = map(int, p_info['bbox'])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_pose, x2), min(h_pose, y2)
    pw, ph = x2 - x1, y2 - y1

    # ───── face 이미지 → face HED
    face_cv = cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)
    f_info  = max(face_det.get(face_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    fx1, fy1, fx2, fy2 = map(int, f_info['bbox'])
    face_crop_pil = face_im.crop((fx1, fy1, fx2, fy2))
    face_hed_pil = hed(face_crop_pil, safe=False, scribble=False)
    face_hed_resized = face_hed_pil.resize((pw, ph), Image.LANCZOS)
    face_hed_np = np.array(face_hed_resized).astype(np.float32)

    # ───── pose 전체 HED
    pose_hed_pil = hed(pose_im, safe=False, scribble=False).resize(pose_im.size, Image.LANCZOS)
    pose_hed_np = np.array(pose_hed_pil).astype(np.float32)

    # ───── 소프트 마스킹
    mask = np.zeros((h_pose, w_pose), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 31), sigmaX=10, sigmaY=10)[..., None]

    face_canvas_np = np.zeros_like(pose_hed_np).astype(np.float32)
    face_canvas_np[y1:y2, x1:x2] = face_hed_np

    pose_np = pose_hed_np.astype(np.float32)
    blended_np = mask * face_canvas_np + (1 - mask) * pose_np
    blended_np = blended_np.clip(0, 255).astype(np.uint8)
    merged_hed_pil = Image.fromarray(blended_np).convert("RGB")
    merged_hed_pil.save(OUTDIR / "merged_hed_soft.png")

    # ───── ControlNet 구성
    controlnets, images, scales, masks = [], [], [], []
    controlnets.append(ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE))
    images.append(merged_hed_pil)
    scales.append(COND_HED)
    masks.append(None)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_SDXL,
        controlnet=MultiControlNetModel(controlnets),
        torch_dtype=DTYPE,
        add_watermarker=False
    ).to(DEVICE)

    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()

    # ───── 스타일 이미지 반복 처리
    for style_path in sorted(STYLE_DIR.glob("*")):
        if not style_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            continue
        style_pil = load_rgb(style_path)

        gen_args = dict(
            prompt=PROMPT,
            negative_prompt=NEG,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            image=images,
            controlnet_conditioning_scale=scales,
            control_mask=masks,
        )

        ip = IPAdapterXL(
            pipe, STYLE_ENC, STYLE_IP, DEVICE,
            target_blocks=["up_blocks.0.attentions.1"]
        )
        out = ip.generate(pil_image=style_pil, scale=STYLE_SCALE,
                          seed=SEED, **gen_args)[0]

        fname = OUTDIR / f"{style_path.stem}.png"
        out.save(fname)
        print("✅ saved →", fname)

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES 안에서 논리 GPU 번호")
    args = ap.parse_args()
    main(args.gpu)
