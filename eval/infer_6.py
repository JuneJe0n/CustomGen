"""
6. m5 - ablation 4 (face HED wo mask multiplication + face mask + pose kps)
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
def main(gpu_idx: int):
    # Set GPU
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)

    # Load input imgs
    face_im  = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im  = to_sdxl_res(load_rgb(POSE_IMG))
    style_pil = load_rgb(STYLE_IMG)
    face_im.save(OUTDIR/"0_face_input.png")
    pose_im.save(OUTDIR/"1_pose_input.png")
    style_pil.save(OUTDIR/"2_style_input.png")
    W, H = pose_im.size

    # Face detector
    face_det = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=gpu_idx, det_size=(640, 640))

    # Face bbox from face img
    face_cv_full = cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)
    f_faces = face_det.get(face_cv_full)
    if not f_faces:
        raise RuntimeError("Couldn't detect face from face img")
    f_info = max(f_faces, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    fx1, fy1, fx2, fy2 = map(int, f_info['bbox'])
    fw, fh = fx2 - fx1, fy2 - fy1
    face_crop_pil = face_im.crop((fx1, fy1, fx2, fy2))

    # Face bbox from pose img
    pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
    p_faces = face_det.get(pose_cv)
    if not p_faces:
        raise RuntimeError("Couldn't detect face from pose img")
    p_info = max(p_faces, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    x1, y1, x2, y2 = map(int, p_info['bbox'])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    pw, ph = x2 - x1, y2 - y1


    # Compute scale factor
    scale_w, scale_h = pw / fw, ph / fh
    scale = min(scale_w, scale_h)
    new_w, new_h = int(fw * scale), int(fh * scale)

    # Resize HED 
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    face_hed_crop_pil = hed(face_crop_pil, safe=False, scribble=False)
    face_hed_resized  = face_hed_crop_pil.resize((new_w, new_h), Image.LANCZOS)
    face_hed_np       = np.array(face_hed_resized).astype(np.float32)


    # Create face mask using FaceMesh polygon
    poly_pts_scaled, poly_mask, poly_mask_3c = create_face_mask(face_crop_pil, fw, fh, scale, new_h, new_w)

    # Apply face mask on HED
    # face_hed_np_masked = (face_hed_np * poly_mask_3c).astype(np.float32)
    face_hed_np_masked = face_hed_np.astype(np.float32)
   
    # Openpose
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    pose_openpose_pil = openpose(pose_im, hand_and_face=True).resize((W, H), Image.LANCZOS)
    pose_openpose_pil.save(OUTDIR / "3_pose_kps.png")
    pose_openpose_np  = np.array(pose_openpose_pil).astype(np.float32) # openpose skeleton img

    
    # Insert face HED on pose img size empty canvas
    face_hed_canvas_np = np.zeros_like(pose_openpose_np, dtype=np.float32) # empty canvas of size pose
    cx, cy = (x1+x2)//2, (y1+y2)//2  # center of face bbox of pose img
    start_x, start_y = cx - new_w//2, cy - new_h//2
    face_hed_canvas_np[start_y:start_y+new_h, start_x:start_x+new_w] = face_hed_np_masked
    face_hed_canvas_pil = Image.fromarray(face_hed_canvas_np.clip(0,255).astype(np.uint8)).convert("RGB")
    face_hed_canvas_pil.save(OUTDIR / "4_hed_aligned.png")
 

    # Body mask (Inverse of face mask)
    face_mask_full = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(face_mask_full, [poly_pts_scaled + [start_x, start_y]], 1.0)
    face_mask_full = cv2.GaussianBlur(face_mask_full, (31,31), sigmaX=10, sigmaY=10)
    to_mask_image(face_mask_full).save(OUTDIR/"5_face_mask_full.png")
    body_mask = (1.0 - face_mask_full).astype(np.float32)
    to_mask_image(body_mask).save(OUTDIR/"6_body_mask.png")


    # ControlNet setup
    controlnets = [
        ControlNetModel.from_pretrained(CN_POSE, torch_dtype=DTYPE, use_safetensors=False),
        ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE)
    ]

    images = [pose_openpose_pil,face_hed_canvas_pil]
    scales = [COND_POSE, COND_HED]
    masks = [to_mask_image(body_mask),to_mask_image(face_mask_full)]


    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_SDXL,
        controlnet=MultiControlNetModel(controlnets),
        torch_dtype=DTYPE,
        add_watermarker=False
    ).to(DEVICE)

    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()


    gen_args = dict(
        prompt=PROMPT,
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        image=images,
        controlnet_conditioning_scale=scales,
        control_mask=masks,
        generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    )

    # IP-Adapter
    ip = IPAdapterXL(
        pipe, STYLE_ENC, STYLE_IP, DEVICE,
        target_blocks=["up_blocks.0.attentions.1"]
    )
    pipe_args = {k: v for k, v in gen_args.items() if k != "generator"}
    out = ip.generate(
        pil_image=style_pil,
        scale=STYLE_SCALE,
        seed=SEED,                 
        **pipe_args
    )[0]
    del ip
    
    out.save(OUTDIR/"7_final_result.png")
    print(f"✅ Saved all intermediates in {OUTDIR}")


# --- CLI ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    main(args.gpu)
