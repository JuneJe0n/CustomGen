"""
face : FaceID embedding (masked inpaint — impacts face region only)
pose : openpose kps
"""
import argparse, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
)
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import OpenposeDetector
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline  # ★ InstantID Pipeline
from skimage.transform import SimilarityTransform, warp
import mediapipe as mp

# ─────────────────── Settings ───────────────────
PROMPT = "a baby girl sitting, clear facial features, detailed, realistic, smooth colors"
NEG = "(lowres, bad quality, watermark, disjointed, strange limbs, cut off, bad anatomymissing limbs, fused fingers)"
FACE_IMG  = Path("/data2/jeesoo/FFHQ/00000/00003.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/pose/p2.jpeg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/style/s3.png")  # optional (not used here)

CN_POSE    = "/data2/jiyoon/custom/ckpts/controlnet-openpose-sdxl-1.0"
BASE_SDXL  = "stabilityai/stable-diffusion-xl-base-1.0"

# ★ FaceID adapter & InsightFace model paths (from https://huggingface.co/h94/IP-Adapter-FaceID)
FACEID_ADAPTER_CKPT = "/data2/jiyoon/custom/ckpts/ip-adapter-faceid_sdxl/ip-adapter-faceid_sdxl.bin"
INSIGHTFACE_ROOT    = "/data2/jiyoon/InstantID"  

COND_POSE    = 0.65
FACEID_SCALE = 0.85           
CFG, STEPS   = 7.0, 40
SEED         = 42

# inpaint strength for SDXL: 0 (no change) → 1 (full regenerate region)
INPAINT_STRENGTH = 0.60

OUTDIR = Path("/data2/jiyoon/custom/results/face_emb")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────── Globals ───────────────────
face_det = None

# ─────────────────── Utils ───────────────────
def load_rgb(p): return Image.open(p).convert("RGB")

def to_sdxl_res(img, base=64, short=1024, long=1024, low_mem=False):
    if low_mem: short, long = 768, 768
    w, h = img.size
    r = short / min(w, h); w, h = int(w*r), int(h*r)
    r = long  / max(w, h); w, h = int(w*r), int(h*r)
    return img.resize(((w//base)*base, (h//base)*base), Image.LANCZOS)

def expand_bbox(bbox, scale, W, H):
    x1, y1, x2, y2 = map(int, bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    nx1, ny1 = int(max(0, cx - w/2)), int(max(0, cy - h/2))
    nx2, ny2 = int(min(W, cx + w/2)), int(min(H, cy + h/2))
    return [nx1, ny1, nx2, ny2]

def extract_pose_keypoints(img, pose_detector, include_body=True, include_hand=True, include_face=True, save_name="01_pose_kps.png"):
    kps = pose_detector(img, include_body=include_body, include_hand=include_hand, include_face=include_face)
    kps = kps.resize(img.size, Image.LANCZOS)
    kps.save(OUTDIR / save_name)
    print(f"💾 Saved: {save_name}")
    return kps

def align_face_with_landmarks(face_img, pose_img, face_det):
    face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    pose_cv = cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR)
    face_infos = face_det.get(face_cv)
    pose_infos = face_det.get(pose_cv)
    if not face_infos or not pose_infos:
        return face_img, None, None
    face_info = max(face_infos, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    pose_info = max(pose_infos, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    face_kps, pose_kps = np.array(face_info['kps']), np.array(pose_info['kps'])

    # similarity transform face->pose
    tform = SimilarityTransform()
    ok = tform.estimate(face_kps, pose_kps)
    if not ok:
        return face_img, None, face_info

    h, w = pose_img.size[::-1]
    aligned_face = warp(np.array(face_img), tform.inverse, output_shape=(h, w), preserve_range=True)
    aligned_face_pil = Image.fromarray(aligned_face.astype(np.uint8))
    aligned_face_pil.save(OUTDIR/"02_aligned_face.png")
    pose_bbox_expanded = expand_bbox(pose_info['bbox'], scale=1.35, W=w, H=h)
    return aligned_face_pil, pose_bbox_expanded, face_info

def create_mediapipe_face_mask(img):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = fm.process(img_cv)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            face_oval = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            mask = np.zeros(img_cv.shape[:2], dtype=np.float32)
            pts = [(int(lm.landmark[i].x*img_cv.shape[1]), int(lm.landmark[i].y*img_cv.shape[0])) for i in face_oval]
            cv2.fillPoly(mask, [np.array(pts)], 1.0)
            mask = cv2.GaussianBlur(mask, (21,21), 7)
            Image.fromarray((mask*255).astype(np.uint8)).save(OUTDIR/"03_mediapipe_mask.png")
            return mask[...,None]
    return None

def create_enhanced_soft_mask(pose_img, bbox):
    h, w = pose_img.size[::-1]
    x1,y1,x2,y2 = bbox
    mp_mask = create_mediapipe_face_mask(pose_img)
    if mp_mask is not None:
        roi = np.zeros_like(mp_mask)
        roi[y1:y2,x1:x2] = mp_mask[y1:y2,x1:x2]
        return roi
    mask = np.zeros((h,w),dtype=np.float32); mask[y1:y2,x1:x2]=1.0
    mask = cv2.GaussianBlur(mask,(51,51),15)
    Image.fromarray((mask*255).astype(np.uint8)).save(OUTDIR/"03b_bbox_mask.png")
    return mask[...,None]

def npmask_to_pil(mask01):
    """mask01: HxWx1 float32 [0..1] -> PIL L (white=paint region)"""
    if mask01.ndim == 3: mask01 = mask01[...,0]
    m = (np.clip(mask01,0,1)*255).astype(np.uint8)
    return Image.fromarray(m, mode="L")

# ─────────────────── Main ───────────────────
def main(gpu_idx, low_memory=False):
    global face_det
    DEVICE=f"cuda:{gpu_idx}"
    DTYPE=torch.float16
    torch.manual_seed(SEED)

    # Face detector / embedder (InsightFace via antelopev2)
    face_det = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_ROOT,
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider'])
    face_det.prepare(ctx_id=gpu_idx, det_size=(640,640), det_thresh=0.3)

    # Inputs
    face_im = to_sdxl_res(load_rgb(FACE_IMG), low_mem=low_memory)
    pose_im = to_sdxl_res(load_rgb(POSE_IMG), low_mem=low_memory)
    face_im.save(OUTDIR/"00_face_input.png"); pose_im.save(OUTDIR/"00_pose_input.png")

    # Align + bbox + soft mask
    aligned_face, bbox, face_info = align_face_with_landmarks(face_im, pose_im, face_det)
    if bbox is None:
        pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
        p_info = max(face_det.get(pose_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
        bbox = expand_bbox(p_info['bbox'], 1.35, pose_im.width, pose_im.height)
        aligned_face = face_im  # fallback
    mask01 = create_enhanced_soft_mask(pose_im, bbox)
    mask_pil = npmask_to_pil(mask01)
    mask_pil.save(OUTDIR/"03c_inpaint_mask.png")

    # Pose kps for full body + hands + face 
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    pose_kps = extract_pose_keypoints(
        pose_im, openpose,
        include_body=True, include_hand=True, include_face=True,
        save_name="01_pose_kps_full.png"
    )
    del openpose; torch.cuda.empty_cache()

    # ── ControlNet (pose only)
    controlnet_pose = ControlNetModel.from_pretrained(CN_POSE, torch_dtype=DTYPE)

    # ── Pass 1: base synthesis with pose only
    pipe_base = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_SDXL, controlnet=controlnet_pose, torch_dtype=DTYPE, add_watermarker=False
    ).to(DEVICE)
    pipe_base.enable_vae_tiling()
    pipe_base.enable_xformers_memory_efficient_attention()
    pipe_base.enable_attention_slicing(1)
    pipe_base.enable_model_cpu_offload()

    base = pipe_base(
        prompt=PROMPT,
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        image=pose_kps,
        controlnet_conditioning_scale=COND_POSE,
        generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    ).images[0]
    base.save(OUTDIR/"05_base_pose_only.png")

    # Clean up after base generation
    del pipe_base
    torch.cuda.empty_cache()

    # ── Pass 2: masked inpaint refinement with InstantID + pose ControlNet
    pipe_inp = StableDiffusionXLInstantIDPipeline.from_pretrained(
        BASE_SDXL, controlnet=controlnet_pose, torch_dtype=DTYPE, add_watermarker=False
    ).to(DEVICE)
    pipe_inp.enable_vae_tiling()
    pipe_inp.enable_xformers_memory_efficient_attention()
    pipe_inp.enable_attention_slicing(1)
    pipe_inp.enable_model_cpu_offload()

    # Load InstantID adapter
    pipe_inp.load_ip_adapter_instantid(
        model_ckpt=FACEID_ADAPTER_CKPT,
        image_emb_dim=512,
        num_tokens=16,
        scale=FACEID_SCALE,
    )

    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    out = ip_face.generate(
        pil_image=face_im,              # Use the face image directly
        scale=FACEID_SCALE,             # identity strength
        prompt=PROMPT,
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        image=pose_kps,                                 # ControlNet condition (pose)
        controlnet_conditioning_scale=COND_POSE,
        init_image=base,                                 # inpaint from base image
        mask_image=mask_pil,                             # ← restrict FaceID edits to face region
        strength=INPAINT_STRENGTH,
        generator=gen,
    )[0]

    out.save(OUTDIR/"06_final_faceid_masked.png")
    print(f"✅ Saved intermediates and result in {OUTDIR}")

# ─────────────────── CLI ───────────────────
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--low-memory", action="store_true")
    args = ap.parse_args()
    main(args.gpu, args.low_memory)
