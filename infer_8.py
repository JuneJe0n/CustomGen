"""
8. ours (face mesh HED + face kps + pose kps)
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

from config import NEG, CN_HED, CN_POSE, BASE_SDXL, STYLE_ENC, STYLE_IP, COND_HED, COND_POSE, STYLE_SCALE, CFG, STEPS, SEED
from utils import *

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

    # Load input imgs
    face_im  = to_sdxl_res(load_rgb(face_img_path))
    pose_im  = to_sdxl_res(load_rgb(pose_img_path))
    style_pil = load_rgb(style_img_path)
    # Set output path
    final_output_path = Path(output_path)
    # Create directory for intermediate files
    output_dir = final_output_path.parent / final_output_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate prompt based on input images
    from utils import PromptGenerator
    generator = PromptGenerator()
    prompt = generator.generate_combined_prompt(face_img_path, pose_img_path)
    W, H = pose_im.size


    # --- bbox ---
    # Face detector
    # Use device 0 when CUDA_VISIBLE_DEVICES is set, otherwise use gpu_idx
    device_id = 0 if 'CUDA_VISIBLE_DEVICES' in os.environ else gpu_idx
    face_det = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=device_id, det_size=(640, 640))

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



    # --- Resize ---
    # Compute scale factor
    scale_w, scale_h = pw / fw, ph / fh
    scale = min(scale_w, scale_h)
    new_w, new_h = int(fw * scale), int(fh * scale)

    # Resize HED 
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    face_hed_crop_pil = hed(face_crop_pil, safe=False, scribble=False)
    face_hed_resized  = face_hed_crop_pil.resize((new_w, new_h), Image.LANCZOS)
    face_hed_np       = np.array(face_hed_resized).astype(np.float32)

    # Resize face crop
    face_crop_pil_resized = face_crop_pil.resize((new_w, new_h), Image.LANCZOS)
    face_crop_np = np.array(face_crop_pil_resized).astype(np.float32)



    # --- Mask ---
    # Create face mask using FaceMesh polygon
    poly_pts_scaled, poly_mask, poly_mask_3c = create_face_mask(face_crop_pil, fw, fh, scale, new_h, new_w)

    # Apply face mask on HED
    face_hed_np_masked = (face_hed_np * poly_mask_3c).astype(np.float32)

    # Apply face mask on face crop
    face_crop_np_masked = (face_crop_np * poly_mask_3c).astype(np.float32)

    # Calculate position for face placement
    cx, cy = (x1+x2)//2, (y1+y2)//2  # center of face bbox of pose img
    start_x, start_y = cx - new_w//2, cy - new_h//2
    
    # Body mask (Inverse of face mask)
    face_mask_full = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(face_mask_full, [poly_pts_scaled + [start_x, start_y]], 1.0)
    face_mask_full = cv2.GaussianBlur(face_mask_full, (31,31), sigmaX=10, sigmaY=10)
    body_mask = (1.0 - face_mask_full).astype(np.float32)
    to_mask_image(face_mask_full).save(output_dir/"3_face_mask_full.png")

    # Apply body mask on pose img
    pose_np = np.array(pose_im).astype(np.float32)
    pose_np_masked = (pose_np * body_mask[:,:,np.newaxis]).astype(np.float32)




    # --- Composite ---

    # Create composite: face crop in face region + pose img in body region
    pose_np = np.array(pose_im).astype(np.float32)
    
    # Start with pose image masked to body only
    integrated_canvas_np = pose_np * body_mask[:,:,np.newaxis]
    
    # Overlay masked face crop at the correct position
    end_x, end_y = start_x + new_w, start_y + new_h
    start_x_clip, start_y_clip = max(0, start_x), max(0, start_y)
    end_x_clip, end_y_clip = min(W, end_x), min(H, end_y)
    
    face_start_x = max(0, -start_x + cx - new_w//2)
    face_start_y = max(0, -start_y + cy - new_h//2)
    face_end_x = face_start_x + (end_x_clip - start_x_clip)
    face_end_y = face_start_y + (end_y_clip - start_y_clip)
    
    integrated_canvas_np[start_y_clip:end_y_clip, start_x_clip:end_x_clip] += face_crop_np_masked[face_start_y:face_end_y, face_start_x:face_end_x]
    integrated_canvas_pil = Image.fromarray(integrated_canvas_np.clip(0,255).astype(np.uint8)).convert("RGB")

    integrated_canvas_pil.save(output_dir/ "5_integrated_canvas.png")
    


    # --- kps ---
    # Openpose
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    pose_openpose_pil = openpose(integrated_canvas_pil, hand_and_face=True).resize((W, H), Image.LANCZOS)
    pose_openpose_pil.save(output_dir/ "7_pose_kps.png")
    pose_openpose_np  = np.array(pose_openpose_pil).astype(np.float32) # openpose skeleton img

    # Insert face HED on pose img size empty canvas
    face_hed_canvas_np = np.zeros_like(pose_openpose_np, dtype=np.float32) # empty canvas of size pose
    face_hed_canvas_np[start_y:start_y+new_h, start_x:start_x+new_w] = face_hed_np_masked
    face_hed_canvas_pil = Image.fromarray(face_hed_canvas_np.clip(0,255).astype(np.uint8)).convert("RGB")
    face_hed_canvas_pil.save(output_dir/ "6_hed_aligned.png")
    
    
    # --- Infer ---
    # ControlNet 
    controlnets = [
        ControlNetModel.from_pretrained(CN_POSE, torch_dtype=DTYPE, use_safetensors=False),
        ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE)
    ]

    images = [pose_openpose_pil,face_hed_canvas_pil]
    scales = [COND_POSE, COND_HED]
    masks = [None,None]


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
        prompt=prompt,
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
    
    # Clear GPU memory
    del pipe
    torch.cuda.empty_cache()
    out.save(output_dir/"8_final_result.png")
    # Also save to the final output path
    out.save(final_output_path)
    print(f"✅ Saved final result to {final_output_path}")
    
    return True


# --- CLI ---
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
