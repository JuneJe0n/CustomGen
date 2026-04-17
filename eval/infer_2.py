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

# ───────── 옵션 기본값(미정의 시 대체) ─────────
try:
    FACE_DET_ROOT
except NameError:
    FACE_DET_ROOT = "/data2/jiyoon/InstantID"
try:
    SAVE_INTERMEDIATES
except NameError:
    SAVE_INTERMEDIATES = True

use_style = True

_REQUIRED_CFG = [
    "BASE_SDXL", "CN_HED",
    "COND_HED",
    "NEG", "CFG", "STEPS", "SEED",
    "OUTDIR", "STYLE_ENC", "STYLE_IP", "STYLE_SCALE",
    "use_style"
]
_missing = [k for k in _REQUIRED_CFG if k not in globals()]
if _missing:
    raise RuntimeError(f"[config.py] 다음 키가 필요합니다: {_missing}")

# --- Main ---
def main(face_img_path: str, pose_img_path: str, style_img_path: str, output_path: str, gpu_idx: int = 0):
    # Set GPU - use cuda:0 when CUDA_VISIBLE_DEVICES is set, otherwise use specified gpu_idx
    import os
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        DEVICE = "cuda:0"
    else:
        DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)
    
    # Set output path
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input imgs
    face_im   = to_sdxl_res(load_rgb(face_img_path))
    pose_im   = to_sdxl_res(load_rgb(pose_img_path))
    style_pil = load_rgb(style_img_path)
    
    # Generate prompt based on input images
    from utils import PromptGenerator
    generator = PromptGenerator()
    prompt = generator.generate_combined_prompt(face_img_path, pose_img_path)
    
    if SAVE_INTERMEDIATES:
        face_im.save(OUTDIR/"0_face_input.png")
        pose_im.save(OUTDIR/"1_pose_input.png")
        style_pil.save(OUTDIR/"2_style_input.png")
    w_pose, h_pose = pose_im.size


    # --- bbox ---
    # Face detector
    # Use device 0 when CUDA_VISIBLE_DEVICES is set, otherwise use gpu_idx
    device_id = 0 if 'CUDA_VISIBLE_DEVICES' in os.environ else gpu_idx
    face_det = FaceAnalysis(
        name="antelopev2",
        root=str(FACE_DET_ROOT),
        providers=[('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=device_id, det_size=(640, 640))
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
        prompt=prompt,
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        image=images,
        controlnet_conditioning_scale=scales,
        control_mask=masks,
    )

    if use_style:
        # IP-Adapter
        ip = IPAdapterXL(
            pipe, STYLE_ENC, STYLE_IP, DEVICE,
            target_blocks=["up_blocks.0.attentions.1"]
        )
        out = ip.generate(pil_image=style_pil, scale=STYLE_SCALE,
                            seed=SEED, **gen_args)[0]
    else:
        out = pipe(**gen_args).images[0]

    out.save(final_path)
    print(f"✅ Saved final result to {final_path}")
    
    # Clear GPU memory
    if 'ip' in locals():
        del ip
    del pipe
    torch.cuda.empty_cache()
    
    return True

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--face_img", type=str, required=True, help="Path to face image")
    ap.add_argument("--pose_img", type=str, required=True, help="Path to pose image")
    ap.add_argument("--style_img", type=str, required=True, help="Path to style image")
    ap.add_argument("--output_path", type=str, required=True, help="Output path for final result")
    ap.add_argument("--gpu", type=int, default=2, help="GPU index to use")
    args = ap.parse_args()
    
    try:
        success = main(args.face_img, args.pose_img, args.style_img, args.output_path, args.gpu)
        if success:
            exit(0)
        else:
            exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)