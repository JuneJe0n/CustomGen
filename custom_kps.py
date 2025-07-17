#!/usr/bin/env python
# custom_facial_kps_enhanced.py (2025-07-17 rev-A : KPS+Canny+Depth ControlNets)

"""
예시
CUDA_VISIBLE_DEVICES=0 python custom_facial_kps_enhanced.py --ctrl edge_kps --gpu 0
CUDA_VISIBLE_DEVICES=0 python custom_facial_kps_enhanced.py --ctrl edge_kps_depth --style --gpu 0
"""
import argparse, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import MidasDetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL

# ─────────────── 공통 파라미터 ───────────────
PROMPT = "a baby with sharp, ultra-detailed facial features"
NEG    = "(blurry face, smooth skin, watermark, lowres)"
FACE_IMG  = Path("/data2/jiyoon/custom/data/face/00000.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/pose/p2.jpeg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/style/s3.png")

CN_EDGE  = "diffusers/controlnet-canny-sdxl-1.0"
CN_DEPTH = "diffusers/controlnet-depth-sdxl-1.0-small"
BASE_SDXL= "stabilityai/stable-diffusion-xl-base-1.0"

STYLE_ENC = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP  = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

STEPS, SEED = 50, 42
OUTDIR = Path("/data2/jiyoon/custom/results/mode/8/kps")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────── 유틸 ───────────────
def load_rgb(p): return Image.open(p).convert("RGB")

def to_sdxl_res(img, base=64, short=1024, long=1024):
    w, h = img.size
    r = short/min(w,h); w, h = int(w*r), int(h*r)
    r = long /max(w,h);  w, h = int(w*r), int(h*r)
    return img.resize(((w//base)*base,(h//base)*base), Image.BILINEAR)

def depth_to_rgb(arr):
    if isinstance(arr, Image.Image): return arr.convert("RGB")
    if torch.is_tensor(arr): arr = arr.squeeze().cpu().numpy()
    arr = (arr-arr.min())/(arr.max()-arr.min()+1e-8)
    g   = (arr*255).astype("uint8")
    return Image.fromarray(np.repeat(g[...,None],3,-1))

# ─────────────── 메인 ───────────────
def main(ctrl_type, use_style, gpu_idx):
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)

    # detector & depth
    face_det = FaceAnalysis(name="antelopev2", root="/data2/jiyoon/InstantID",
                            providers=[('CUDAExecutionProvider', {'device_id':gpu_idx}),
                                       'CPUExecutionProvider'])
    face_det.prepare(ctx_id=gpu_idx, det_size=(640,640))
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)

    # load images
    face_im = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im = to_sdxl_res(load_rgb(POSE_IMG))
    style_pil = load_rgb(STYLE_IMG)
    w_pose, h_pose = pose_im.size

    # pose bbox & mask for face region in pose_im
    pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
    p_faces = face_det.get(pose_cv)
    if not p_faces:
        print("Error: No face detected in pose image.")
        return
    p_info  = max(p_faces,
                  key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    px1,py1,px2,py2 = map(int, p_info['bbox'])
    px1,py1 = max(0,px1), max(0,py1);  px2,py2 = min(w_pose,px2), min(h_pose,py2)

    # Create a feathered mask for the pose face region
    mask_canvas = np.zeros((h_pose,w_pose,3), np.uint8)
    # Define a larger area for the mask to cover potential face expansion
    # You might need to adjust this margin for optimal results
    mask_margin = 0.2
    mw, mh = int((px2-px1)*mask_margin), int((py2-py1)*mask_margin)
    mask_x1 = max(0, px1 - mw)
    mask_y1 = max(0, py1 - mh)
    mask_x2 = min(w_pose, px2 + mw)
    mask_y2 = min(h_pose, py2 + mh)
    cv2.rectangle(mask_canvas, (mask_x1, mask_y1), (mask_x2, mask_y2), (255,255,255), -1)
    mask_canvas = cv2.GaussianBlur(mask_canvas, (51,51), 0) # Feathering
    mask_pil = Image.fromarray(mask_canvas)
    mask_pil.save(OUTDIR/f"pose_face_mask.png") # Debug output

    # face bbox & facial keypoints/canny extraction from face_im
    face_cv = cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)
    f_faces = face_det.get(face_cv)
    if not f_faces:
        print("Error: No face detected in face image.")
        return
    f_info  = max(f_faces,
                  key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    fx1,fy1,fx2,fy2 = map(int, f_info['bbox'])
    facial_kps = f_info['kps'] # This contains a list of (x,y) keypoint coordinates

    # ─────────────── Generate KPS image ───────────────
    face_crop_w, face_crop_h = fx2 - fx1, fy2 - fy1
    kps_canvas_orig = Image.new("RGB", (face_crop_w, face_crop_h), (0, 0, 0)) # Black background
    draw = ImageDraw.Draw(kps_canvas_orig)
    for kp in facial_kps:
        x_kp, y_kp = int(kp[0] - fx1), int(kp[1] - fy1)
        draw.ellipse((x_kp - 2, y_kp - 2, x_kp + 2, y_kp + 2), fill=(255, 255, 255), outline=(255, 255, 255))
    kps_pil_orig = kps_canvas_orig

    # ─────────────── Generate Canny image from face_im ───────────────
    # Apply processing similar to original method B or A for better edges
    face_crop_for_canny = face_cv[fy1:fy2, fx1:fx2]
    face_crop_for_canny = cv2.GaussianBlur(face_crop_for_canny, (5,5), 0)
    canny_edge = cv2.Canny(face_crop_for_canny, 80, 160) # Optimized Canny parameters
    canny_edge = cv2.dilate(canny_edge, np.ones((3,3), np.uint8), 1)
    canny_pil_orig = Image.fromarray(cv2.cvtColor(canny_edge, cv2.COLOR_GRAY2RGB))

    # ─────────────── Resize and Inject KPS & Canny into pose face bbox ───────────────
    # Calculate target size for the injected face content
    target_face_w, target_face_h = px2 - px1, py2 - py1

    # KPS Injection
    kps_resized = kps_pil_orig.resize((target_face_w, target_face_h), Image.BILINEAR)
    kps_canvas_full = Image.new("RGB", (w_pose, h_pose), (0,0,0))
    kps_canvas_full.paste(kps_resized, (px1, py1))
    kps_pil_final = kps_canvas_full
    kps_pil_final.save(OUTDIR/f"injected_kps_control.png") # Debug output

    # Canny Injection
    canny_resized = canny_pil_orig.resize((target_face_w, target_face_h), Image.BILINEAR)
    canny_canvas_full = Image.new("RGB", (w_pose, h_pose), (0,0,0))
    canny_canvas_full.paste(canny_resized, (px1, py1))
    canny_pil_final = canny_canvas_full
    canny_pil_final.save(OUTDIR/f"injected_canny_control.png") # Debug output


    # ─────────────── Depth extraction from pose_im (for full image control) ───────────────
    depth_pil = depth_to_rgb(midas(pose_im)).resize(pose_im.size, Image.BILINEAR)
    depth_pil.save(OUTDIR/f"pose_depth_control.png") # Debug output

    # ─────────────── ControlNet Configuration ───────────────
    # Adjust these weights for desired influence. KPS and Canny should be strong for face.
    KPS_CN_SCALE = 1.0 # Strong influence for facial structure from KPS
    CANNY_CN_SCALE = 0.8 # Moderate influence for facial edges from Canny
    DEPTH_CN_SCALE = 0.5 # General influence for pose and background depth
    CFG = 6.5 # Guidance scale, can be adjusted

    controlnets = []
    images_for_cn = []
    scales_for_cn = []
    masks_for_cn = [] # Masks are applied to individual control images if needed

    # Always add KPS ControlNet if 'edge_kps' or 'edge_kps_depth' is chosen
    if "edge_kps" in ctrl_type:
        controlnets.append(ControlNetModel.from_pretrained(CN_EDGE, torch_dtype=DTYPE))
        images_for_cn.append(kps_pil_final)
        scales_for_cn.append(KPS_CN_SCALE)
        masks_for_cn.append(mask_pil) # Apply mask to KPS control

    # Always add Canny ControlNet if 'edge_kps' or 'edge_kps_depth' is chosen
    if "edge_kps" in ctrl_type:
        controlnets.append(ControlNetModel.from_pretrained(CN_EDGE, torch_dtype=DTYPE))
        images_for_cn.append(canny_pil_final)
        scales_for_cn.append(CANNY_CN_SCALE)
        masks_for_cn.append(mask_pil) # Apply mask to Canny control

    if "depth" in ctrl_type:
        controlnets.append(ControlNetModel.from_pretrained(CN_DEPTH, torch_dtype=DTYPE))
        images_for_cn.append(depth_pil)
        scales_for_cn.append(DEPTH_CN_SCALE)
        masks_for_cn.append(None) # No mask for full-image depth control

    if not controlnets:
        print("Error: No controlnets selected. Please choose 'edge_kps' or 'edge_kps_depth'.")
        return

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_SDXL, controlnet=MultiControlNetModel(controlnets),
        torch_dtype=DTYPE, add_watermarker=False
    ).to(DEVICE)
    pipe.enable_vae_tiling(); pipe.enable_xformers_memory_efficient_attention()
    if not use_style: pipe.enable_sequential_cpu_offload()
    else:             pipe.to(DEVICE)

    gen = dict(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=STEPS,
               guidance_scale=CFG, image=images_for_cn,
               controlnet_conditioning_scale=scales_for_cn, control_mask=masks_for_cn)

    if use_style:
        ip = IPAdapterXL(pipe, STYLE_ENC, STYLE_IP, DEVICE,
                         target_blocks=["up_blocks.0.attentions.1"])
        result = ip.generate(pil_image=style_pil, scale=0.8, seed=SEED, **gen)[0]
    else:
        result = pipe(**gen).images[0]

    out_path = OUTDIR/f"result_kps_canny_depth_style_{use_style}_{ctrl_type}.png"
    result.save(out_path); print("✅ saved:", out_path)

# ─────────────── CLI ───────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", choices=["edge_kps","edge_kps_depth"], required=True,
                    help="Control types: 'edge_kps' for KPS+Canny, 'edge_kps_depth' for KPS+Canny+Depth.")
    ap.add_argument("--style", action="store_true", help="IP-Adapter 스타일 주입 여부")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES 안 논리 GPU")
    args = ap.parse_args()
    main(args.ctrl, args.style, args.gpu)