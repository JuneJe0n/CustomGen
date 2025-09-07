"""
1. m1 (face bbox HED + pose HED)
"""

import argparse, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import OpenposeDetector, HEDdetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL
import mediapipe as mp

from config import *
from utils import *


# --- Main ---
def main(gpu_idx):
    # Set GPU
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)

    # Load input imgs
    face_im   = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im   = to_sdxl_res(load_rgb(POSE_IMG))
    style_pil = load_rgb(STYLE_IMG)
    face_im.save(OUTDIR/"0_face_input.png")
    pose_im.save(OUTDIR/"1_pose_input.png")
    style_pil.save(OUTDIR/"2_style_input.png")
    w_pose, h_pose = pose_im.size


    # --- bbox ---
    # Face detector
    face_det = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=gpu_idx, det_size=(640, 640))
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
   

    # Face bbox from face img
    face_cv = cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)
    f_info  = max(face_det.get(face_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    fx1, fy1, fx2, fy2 = map(int, f_info['bbox'])
    face_crop_pil = face_im.crop((fx1, fy1, fx2, fy2))


    # Face bbox from pose img
    pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
    p_info  = max(face_det.get(pose_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    x1, y1, x2, y2 = map(int, p_info['bbox'])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_pose, x2), min(h_pose, y2)
    pw, ph = x2 - x1, y2 - y1


    # --- HED ---
    # pose HED
    pose_hed_pil = hed(pose_im, safe=False, scribble=False).resize(pose_im.size, Image.LANCZOS)
    pose_hed_np = np.array(pose_hed_pil).astype(np.float32)

    # face HED
    face_hed_pil = hed(face_crop_pil, safe=False, scribble=False)
    face_hed_resized = face_hed_pil.resize((pw, ph), Image.LANCZOS)
    face_hed_np = np.array(face_hed_resized).astype(np.float32)


    # --- Composite ---
    # Soft mask 
    mask = np.zeros((h_pose, w_pose), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 31), sigmaX=10, sigmaY=10)[..., None]  # shape (H, W, 1)

    # Integrate face hed onto canvas
    face_canvas_np = np.zeros_like(pose_hed_np).astype(np.float32)
    face_canvas_np[y1:y2, x1:x2] = face_hed_np

    # blending
    pose_np = pose_hed_np.astype(np.float32)
    blended_np = mask * face_canvas_np + (1 - mask) * pose_np
    blended_np = blended_np.clip(0, 255).astype(np.uint8)
    merged_hed_pil = Image.fromarray(blended_np).convert("RGB")
    merged_hed_pil.save(OUTDIR/"3_merged.png")


    # --- Infer ---
    # ControlNet
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

    if not use_style:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(DEVICE)

    gen_args = dict(
        prompt=PROMPT,
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        image=images,
        controlnet_conditioning_scale=scales,
        control_mask=masks,
    )


    # IP-Adapter
    ip = IPAdapterXL(
        pipe, STYLE_ENC, STYLE_IP, DEVICE,
        target_blocks=["up_blocks.0.attentions.1"]
    )
    out = ip.generate(pil_image=style_pil, scale=STYLE_SCALE,
                        seed=SEED, **gen_args)[0]


    fname = OUTDIR/"4_final_result.png"
    print(f"✅ Saved all intermediates in {OUTDIR}")

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",   type=int, default=0, help="CUDA_VISIBLE_DEVICES 안에서 논리 GPU 번호")
    args = ap.parse_args()
    main(args.gpu)