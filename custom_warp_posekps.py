"""
Complete working version - Face alignment with pose keypoints
Based on original code structure with improvements
"""
import argparse, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import HEDdetector, OpenposeDetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL
from skimage.transform import SimilarityTransform, warp
import mediapipe as mp

# ─────────────────── Settings ───────────────────
PROMPT = "a baby sitting, clear facial features, detailed, realistic, smooth colors"
NEG = "(lowres, bad quality, watermark, disjointed, strange limbs, cut off, bad anatomymissing limbs, fused fingers)"
FACE_IMG  = Path("/data2/jeesoo/FFHQ/00000/00000.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/pose/p2.jpeg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/style/s3.png")

CN_HED     = "/data2/jiyoon/custom/ckpts/controlnet-union-sdxl-1.0"
CN_POSE    = "/data2/jiyoon/custom/ckpts/controlnet-openpose-sdxl-1.0"
BASE_SDXL  = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE_ENC  = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP   = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

COND_HED     = 0.8
COND_POSE    = 0.6
STYLE_SCALE  = 0.8
CFG, STEPS   = 7.0, 50
SEED         = 42
OUTDIR       = Path("/data2/jiyoon/custom/results/pose_kps/00000")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────── Utils ───────────────────
def load_rgb(p): 
    return Image.open(p).convert("RGB")

def to_sdxl_res(img, base=64, short=1024, long=1024):
    w, h = img.size
    r = short / min(w, h); w, h = int(w*r), int(h*r)
    r = long  / max(w, h); w, h = int(w*r), int(h*r)
    return img.resize(((w//base)*base, (h//base)*base), Image.LANCZOS)

def extract_pose_keypoints(pose_img, pose_detector):
    """Extract pose keypoints from pose image"""
    print("🤸 Extracting pose keypoints...")
    pose_kps = pose_detector(pose_img)
    pose_kps.save(OUTDIR / "pose_keypoints.png")
    print(f"💾 Pose keypoints saved: {OUTDIR / 'pose_keypoints.png'}")
    return pose_kps

def align_face_with_landmarks(face_img, pose_img, face_det):
    """InsightFace landmark-based face alignment - matches original code"""
    
    face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    pose_cv = cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR)
    
    # Extract face info
    face_infos = face_det.get(face_cv)
    pose_infos = face_det.get(pose_cv)
    
    if not face_infos or not pose_infos:
        print("⚠️  Face detection failed, using fallback method")
        return face_img, None
    
    # Select largest face
    face_info = max(face_infos, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    pose_info = max(pose_infos, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    
    # Extract 5 key landmarks
    face_kps = face_info['kps']  # (5, 2)
    pose_kps = pose_info['kps']
    
    try:
        # Calculate similarity transform (rotation, scaling, translation)
        tform = SimilarityTransform()
        tform.estimate(face_kps, pose_kps)
        
        # Transform face image to match pose
        h, w = pose_img.size[::-1]
        aligned_face = warp(
            np.array(face_img), 
            tform.inverse, 
            output_shape=(h, w),
            preserve_range=True
        )
        
        aligned_face_pil = Image.fromarray(aligned_face.astype(np.uint8))
        
        # Calculate new bbox for transformed face
        transformed_kps = tform(face_kps)
        x_coords = transformed_kps[:, 0]
        y_coords = transformed_kps[:, 1]
        
        margin = 30  # bbox expansion
        new_bbox = [
            max(0, int(x_coords.min() - margin)),
            max(0, int(y_coords.min() - margin)),
            min(w, int(x_coords.max() + margin)),
            min(h, int(y_coords.max() + margin))
        ]
        
        # Save aligned face
        aligned_face_pil.save(OUTDIR / "aligned_face_landmark.png")
        print(f"💾 Landmark-aligned face saved: {OUTDIR / 'aligned_face_landmark.png'}")
        print(f"📏 Transformed face bbox: {new_bbox}")
        
        return aligned_face_pil, new_bbox
        
    except Exception as e:
        print(f"⚠️  Landmark alignment failed: {e}, using fallback method")
        return face_img, None

def align_face_for_hed(face_img, target_size=(512, 512)):
    """Original function signature - for compatibility"""
    print("👤 Face alignment and HED extraction...")
    
    # Face detection
    face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    face_infos = face_det.get(face_cv)
    
    if not face_infos:
        print("⚠️  Face detection failed")
        return None, None
    
    # Select largest face
    face_info = max(face_infos, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    bbox = face_info['bbox']
    kps = face_info['kps']
    
    # Crop face region
    x1, y1, x2, y2 = map(int, bbox)
    face_crop = face_img.crop((x1, y1, x2, y2))
    
    # Standard face landmarks (frontal face)
    standard_kps = np.array([
        [0.31556875000000000, 0.4615741071428571],  # left eye
        [0.68262291666666670, 0.4615741071428571],  # right eye  
        [0.50026249999999990, 0.6405053571428571],  # nose
        [0.34947187500000004, 0.8246919642857142],  # left mouth
        [0.65343645833333330, 0.8246919642857142]   # right mouth
    ]) * np.array(target_size)
    
    try:
        # Relative keypoints in cropped face
        crop_kps = kps - np.array([x1, y1])
        
        # Calculate similarity transform
        tform = SimilarityTransform()
        tform.estimate(crop_kps, standard_kps)
        
        # Align face
        aligned_face = warp(
            np.array(face_crop),
            tform.inverse,
            output_shape=target_size,
            preserve_range=True
        )
        
        aligned_face_pil = Image.fromarray(aligned_face.astype(np.uint8))
        
        # Extract HED
        face_hed = hed(aligned_face_pil, safe=False, scribble=False)
        
        # Save results
        aligned_face_pil.save(OUTDIR / "aligned_face_standard.png")
        face_hed.save(OUTDIR / "face_hed.png")
        print(f"💾 Standard aligned face saved: {OUTDIR / 'aligned_face_standard.png'}")
        print(f"💾 Face HED saved: {OUTDIR / 'face_hed.png'}")
        
        return aligned_face_pil, face_hed
        
    except Exception as e:
        print(f"⚠️  Face alignment failed: {e}")
        return None, None

def create_mediapipe_face_mask(img, save_path=None):
    """MediaPipe Face Mesh for precise face mask"""
    
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = face_mesh.process(img_cv)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Face oval indices
            face_oval = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            
            # Create mask
            mask = np.zeros(img_cv.shape[:2], dtype=np.float32)
            points = []
            
            for idx in face_oval:
                x = int(landmarks.landmark[idx].x * img_cv.shape[1])
                y = int(landmarks.landmark[idx].y * img_cv.shape[0])
                points.append([x, y])
            
            cv2.fillPoly(mask, [np.array(points)], 1.0)
            
            # Gaussian blur for smooth edges
            mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=7, sigmaY=7)
            
            if save_path:
                mask_vis = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_vis, mode='L').save(save_path)
                print(f"💾 MediaPipe mask saved: {save_path}")
            
            return mask[..., None]  # (H, W, 1)
    
    return None

def create_enhanced_soft_mask(pose_img, bbox, save_dir=None):
    """Enhanced soft mask generation - matches original code"""
    
    h, w = pose_img.size[::-1]
    x1, y1, x2, y2 = bbox
    
    # Try MediaPipe precise face mask
    mp_mask_path = save_dir / "mediapipe_mask.png" if save_dir else None
    mp_mask = create_mediapipe_face_mask(pose_img, mp_mask_path)
    
    if mp_mask is not None:
        # Extract face region only
        roi_mask = np.zeros_like(mp_mask)
        roi_mask[y1:y2, x1:x2] = mp_mask[y1:y2, x1:x2]
        return roi_mask
    
    # Fallback to bbox method
    print("⚠️  MediaPipe mask failed, using bbox method")
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    
    # Soft blur
    blur_size = max(31, int((x2-x1) * 0.1))
    if blur_size % 2 == 0:
        blur_size += 1
    
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), sigmaX=blur_size//3, sigmaY=blur_size//3)
    
    return mask[..., None]

def blend_face_hed_with_pose(face_hed, pose_img, face_mask):
    """Original function signature - simple blending with mask"""
    
    # Get pose image face bbox for resizing
    pose_cv = cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR)
    pose_infos = face_det.get(pose_cv)
    
    if not pose_infos:
        print("⚠️  No face detected in pose image")
        return pose_img
    
    # Select largest face
    face_info = max(pose_infos, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    x1, y1, x2, y2 = map(int, face_info['bbox'])
    
    print(f"📏 Pose face region: ({x1}, {y1}) - ({x2}, {y2}), size: {x2-x1}x{y2-y1}")
    
    # Resize face HED to match pose face size
    target_width = x2 - x1
    target_height = y2 - y1
    face_hed_resized = face_hed.resize((target_width, target_height), Image.LANCZOS)
    
    # Create canvas with pose image size
    pose_np = np.array(pose_img).astype(np.float32)
    face_canvas_np = pose_np.copy()
    
    # Place resized face HED at correct position
    face_hed_np = np.array(face_hed_resized).astype(np.float32)
    face_canvas_np[y1:y2, x1:x2] = face_hed_np
    
    # Apply mask for blending
    if face_mask is not None:
        if len(face_mask.shape) == 3 and face_mask.shape[2] == 1:
            face_mask = np.repeat(face_mask, 3, axis=2)
        blended_np = face_mask * face_canvas_np + (1 - face_mask) * pose_np
    else:
        print("⚠️  No face mask, using direct replacement")
        blended_np = face_canvas_np
    
    blended_np = blended_np.clip(0, 255).astype(np.uint8)
    blended_pil = Image.fromarray(blended_np)
    
    # Save results
    face_hed_resized.save(OUTDIR / "face_hed_resized.png")
    blended_pil.save(OUTDIR / "blended_face_pose.png")
    print(f"💾 Resized face HED saved: {OUTDIR / 'face_hed_resized.png'}")
    print(f"💾 Blended result saved: {OUTDIR / 'blended_face_pose.png'}")
    
    return blended_pil

# ─────────────────── Main (Enhanced) ───────────────────
def main(use_style, gpu_idx):
    global face_det, hed
    
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)

    # Load models
    face_det = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=gpu_idx, det_size=(640, 640), det_thresh=0.3)
    
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)

    # Load images
    face_im = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im = to_sdxl_res(load_rgb(POSE_IMG))
    style_pil = load_rgb(STYLE_IMG)
    w_pose, h_pose = pose_im.size

    print("🔄 Enhanced pipeline: Face alignment + Pose keypoints")
    print(f"Face image size: {face_im.size}")
    print(f"Pose image size: {pose_im.size}")
    
    # ───── 1. Extract pose keypoints
    pose_keypoints = extract_pose_keypoints(pose_im, openpose)
    
    # ───── 2. Try landmark-based face alignment first
    aligned_face, aligned_bbox = align_face_with_landmarks(face_im, pose_im, face_det)
    
    if aligned_bbox is None:
        # Fallback: original method
        print("📦 Using original bbox method")
        pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
        p_info = max(face_det.get(pose_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
        x1, y1, x2, y2 = map(int, p_info['bbox'])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_pose, x2), min(h_pose, y2)
        
        # Extract face HED
        aligned_face, face_hed = align_face_for_hed(face_im, target_size=(512, 512))
        
        if aligned_face is None or face_hed is None:
            print("❌ Face alignment completely failed")
            return
        
        # Create enhanced soft mask
        mask = create_enhanced_soft_mask(pose_im, (x1, y1, x2, y2), save_dir=OUTDIR)
        
        # Blend using original function signature
        merged_hed_pil = blend_face_hed_with_pose(face_hed, pose_im, mask)
        
    else:
        print("✨ Landmark alignment successful")
        x1, y1, x2, y2 = aligned_bbox
        
        # Extract HED from aligned face
        face_crop = aligned_face.crop((x1, y1, x2, y2))
        face_hed = hed(face_crop, safe=False, scribble=False)
        
        # Create enhanced soft mask
        mask = create_enhanced_soft_mask(pose_im, (x1, y1, x2, y2), save_dir=OUTDIR)
        
        # Blend using original function signature
        merged_hed_pil = blend_face_hed_with_pose(face_hed, pose_im, mask)

    # Save final HED
    merged_hed_pil.save(OUTDIR/"merged_hed_enhanced.png")
    print(f"💾 Enhanced HED saved: {OUTDIR/'merged_hed_enhanced.png'}")

    # ───── ControlNet setup
    controlnets = [
        ControlNetModel.from_pretrained(CN_POSE, torch_dtype=DTYPE),
        ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE)
    ]
    
    control_images = [pose_keypoints, merged_hed_pil]
    control_scales = [COND_POSE, COND_HED]

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

    # ───── Image generation
    gen_args = dict(
        prompt=PROMPT,
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        image=control_images,
        controlnet_conditioning_scale=control_scales,
    )

    if use_style:
        ip = IPAdapterXL(
            pipe, STYLE_ENC, STYLE_IP, DEVICE,
            target_blocks=["up_blocks.0.attentions.1"]
        )
        out = ip.generate(pil_image=style_pil, scale=STYLE_SCALE,
                          seed=SEED, **gen_args)[0]
    else:
        out = pipe(**gen_args).images[0]

    fname = OUTDIR/"enhanced_result.png"
    out.save(fname)
    print(f"✅ Final result saved: {fname}")

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", action="store_true", help="IP-Adapter style injection")
    ap.add_argument("--gpu", type=int, default=0, help="GPU number")
    args = ap.parse_args()
    main(args.style, args.gpu)

