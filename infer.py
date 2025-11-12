import cv2, torch, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import OpenposeDetector, HEDdetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL
import mediapipe as mp
from utils import *

# --- Config ---
NEG = "(lowres, bad quality, watermark,strange limbs)"

BASE_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
CN_HED = "./models/controlnet-union-sdxl-1.0"
CN_POSE = "./models/controlnet-openpose-sdxl-1.0"
STYLE_ENC = "./models/IP-Adapter/image_encoder"
STYLE_IP = "./models/IP-Adapter/ip-adapter_sdxl.bin"

FACE_IMG = Path("/home/jiyoon/CustomGen/assets/face.png")
POSE_IMG = Path("/home/jiyoon/CustomGen/assets/pose.jpeg")
STYLE_IMG = Path("/home/jiyoon/CustomGen/assets/style.jpg")
OUTDIR = Path("./results")
OUTDIR.mkdir(parents=True, exist_ok=True)


COND_HED = 0.8
COND_POSE = 0.85
STYLE_SCALE = 0.8
CFG, STEPS = 7.0, 50
SEED = 4

# --- Main ---
def main():
    # Set GPU
    DEVICE = "cuda:0"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)

    # Set path
    face_im  = to_sdxl_res(load_rgb(str(FACE_IMG)))
    pose_im  = to_sdxl_res(load_rgb(str(POSE_IMG)))
    style_pil = load_rgb(str(STYLE_IMG))
    output_dir = OUTDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate prompt
    generator = PromptGenerator()
    prompt = generator.generate_combined_prompt(str(FACE_IMG), str(POSE_IMG))
    W, H = pose_im.size


    # --- bbox ---
    # Face detector
    device_id = 0
    face_det = FaceAnalysis(
        name="antelopev2",
        root="./models",
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
    # Create face mask
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
    to_mask_image(face_mask_full).save(output_dir/"0_aligned_face_mask.png")
    pose_np = np.array(pose_im).astype(np.float32)


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

    integrated_canvas_pil.save(output_dir/ "1_aligned_img.png")
    

    # --- kps ---
    # Openpose
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    pose_openpose_pil = openpose(integrated_canvas_pil, hand_and_face=True).resize((W, H), Image.LANCZOS)
    pose_openpose_pil.save(output_dir/ "2_aligned_kps.png")
    pose_openpose_np  = np.array(pose_openpose_pil).astype(np.float32) # openpose skeleton img

    # Insert face HED on pose img size empty canvas
    face_hed_canvas_np = np.zeros_like(pose_openpose_np, dtype=np.float32) # empty canvas of size pose
    face_hed_canvas_np[start_y:start_y+new_h, start_x:start_x+new_w] = face_hed_np_masked
    face_hed_canvas_pil = Image.fromarray(face_hed_canvas_np.clip(0,255).astype(np.uint8)).convert("RGB")
    face_hed_canvas_pil.save(output_dir/ "3_aligned_hed.png")
    
    
    # --- Infer ---
    # ControlNet 
    controlnets = [
        ControlNetModel.from_pretrained(CN_POSE, torch_dtype=DTYPE, use_safetensors=False),
        ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE)
    ]

    images = [pose_openpose_pil,face_hed_canvas_pil]
    scales = [COND_POSE, COND_HED]


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
    out.save(output_dir/"4_final_result.png")
    print(f"🎉 Saved results to {output_dir}")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            exit(0)
        else:
            exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
