"""
face : HED + mediapipe
pose : kps
"""

import argparse, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel

from controlnet_aux import OpenposeDetector
from insightface.app import FaceAnalysis
from skimage.transform import SimilarityTransform, warp

import mediapipe as mp

# ───────────────────── Settings ─────────────────────
PROMPT = "a baby sitting, clear facial features, detailed, realistic, smooth colors"
NEG = "(lowres, bad quality, watermark, disjointed, strange limbs, cut off, bad anatomy, missing limbs, fused fingers)"

FACE_IMG  = Path("/data2/jeesoo/FFHQ/00000/00000.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/pose/p2.jpeg")

# ControlNet models (SD 2.1)
CN_FACE   = "CrucibleAI/ControlNetMediaPipeFace"                  # MediaPipe Face (face only) — SD2.1/SD1.5
CN_OPEN   = "thibaud/controlnet-sd21-openpose-diffusers"          # OpenPose for SD2.1
BASE_SD21 = "stabilityai/stable-diffusion-2-1-base"               # 512px base (recommended for this face model) ➊

COND_FACE  = 0.8   # weight for face control
COND_POSE  = 0.6   # weight for body+hands control
CFG, STEPS = 7.0, 30
SEED       = 42

OUTDIR = Path("/data2/jiyoon/custom/results/mediapipeface_openpose_sd21")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ───────────────────── Utils ─────────────────────
def load_rgb(p): return Image.open(p).convert("RGB")

def to_sd21_res(img, base=64, target=512):
    w, h = img.size
    r = target / min(w, h); w, h = int(w*r), int(h*r)
    # keep ~square-ish without exceeding 768
    w = min(w, 768); h = min(h, 768)
    return img.resize(((w//base)*base, (h//base)*base), Image.LANCZOS)

def expand_bbox(bbox, scale, W, H):
    x1, y1, x2, y2 = map(int, bbox)
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    w, h = (x2-x1)*scale, (y2-y1)*scale
    nx1, ny1 = int(max(0, cx - w/2)), int(max(0, cy - h/2))
    nx2, ny2 = int(min(W, cx + w/2)), int(min(H, cy + h/2))
    return [nx1, ny1, nx2, ny2]

# ---------- Face alignment (InsightFace 5-keypoint) ----------
def align_face_with_landmarks(face_img, pose_img, face_det):
    face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    pose_cv = cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR)
    face_infos = face_det.get(face_cv)
    pose_infos = face_det.get(pose_cv)
    if not face_infos or not pose_infos:
        return face_img, None
    face_info = max(face_infos, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    pose_info = max(pose_infos, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    face_kps, pose_kps = face_info['kps'], pose_info['kps']
    try:
        tform = SimilarityTransform()
        tform.estimate(face_kps, pose_kps)  # map face->pose
        h, w = pose_img.size[::-1]
        aligned_face = warp(np.array(face_img), tform.inverse, output_shape=(h, w), preserve_range=True)
        aligned_face_pil = Image.fromarray(aligned_face.astype(np.uint8))
        aligned_face_pil.save(OUTDIR/"02_aligned_face.png")
        pose_bbox_expanded = expand_bbox(pose_info['bbox'], scale=1.35, W=w, H=h)
        return aligned_face_pil, pose_bbox_expanded
    except Exception:
        return face_img, None

def soft_face_mask(pose_img, bbox):
    h, w = pose_img.size[::-1]
    x1,y1,x2,y2 = bbox
    mask = np.zeros((h,w,1), dtype=np.float32)
    mask[y1:y2, x1:x2, 0] = 1.0
    mask = cv2.GaussianBlur(mask, (51,51), 15)
    return mask

def paste_face_into_pose(pose_img: Image.Image, aligned_face: Image.Image, mask: np.ndarray, bbox):
    x1,y1,x2,y2 = bbox
    face_region = aligned_face.crop((x1,y1,x2,y2))
    face_np = np.array(face_region).astype(np.float32)
    pose_np = np.array(pose_img).astype(np.float32)
    m = mask
    if m.shape[-1]==1: m = np.repeat(m, 3, axis=2)
    out = pose_np.copy()
    out[y1:y2, x1:x2] = m[y1:y2,x1:x2]*face_np + (1.0-m[y1:y2,x1:x2])*pose_np[y1:y2,x1:x2]
    comp = Image.fromarray(np.clip(out,0,255).astype(np.uint8))
    comp.save(OUTDIR/"03_composite_pose_plus_face.png")
    return comp

# ---------- MediaPipe Face rendering (matches CrucibleAI training spec) ----------
def render_mediapipe_face_map(image_pil: Image.Image, max_faces=1, min_face_size_px=0) -> Image.Image:
    """Returns an RGB image with only the MediaPipe face contours (the 'source' expected by the face ControlNet)."""
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # DrawingSpec per model card (colors/thickness). ➋
    DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
    f_thick, f_rad = 2, 1
    right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
    right_eye_draw  = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
    right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
    left_iris_draw  = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
    left_eye_draw   = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
    left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
    mouth_draw     = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
    head_draw      = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

    face_connection_spec = {}
    for e in mp_face_mesh.FACEMESH_FACE_OVAL:   face_connection_spec[e] = head_draw
    for e in mp_face_mesh.FACEMESH_LEFT_EYE:    face_connection_spec[e] = left_eye_draw
    for e in mp_face_mesh.FACEMESH_LEFT_EYEBROW:face_connection_spec[e] = left_eyebrow_draw
    for e in mp_face_mesh.FACEMESH_RIGHT_EYE:   face_connection_spec[e] = right_eye_draw
    for e in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:face_connection_spec[e] = right_eyebrow_draw
    for e in mp_face_mesh.FACEMESH_LIPS:        face_connection_spec[e] = mouth_draw
    iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}

    def draw_pupils(bgr_img, landmark_list, drawing_spec, halfwidth=2):
        H,W,_ = bgr_img.shape
        for idx, lm in enumerate(landmark_list.landmark):
            if (lm.x<0 or lm.x>=1) or (lm.y<0 or lm.y>=1): continue
            if idx not in drawing_spec: continue
            x,y = int(W*lm.x), int(H*lm.y)
            color = drawing_spec[idx].color
            bgr_img[y-halfwidth:y+halfwidth, x-halfwidth:x+halfwidth, :] = color

    img_rgb = np.asarray(image_pil)
    empty_bgr = np.zeros_like(img_rgb)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=max_faces,
                               refine_landmarks=True, min_detection_confidence=0.5) as facemesh:
        results = facemesh.process(img_rgb).multi_face_landmarks or []
        # (Optional) filter tiny faces
        filtered = []
        for lm in results:
            if min_face_size_px <= 0: filtered.append(lm); continue
            xs = [p.x for p in lm.landmark]; ys=[p.y for p in lm.landmark]
            w = (max(xs)-min(xs))*image_pil.size[0]; h=(max(ys)-min(ys))*image_pil.size[1]
            if min(w,h) >= min_face_size_px: filtered.append(lm)

        for lm in filtered:
            mp_drawing.draw_landmarks(
                empty_bgr, lm, connections=face_connection_spec.keys(),
                landmark_drawing_spec=None, connection_drawing_spec=face_connection_spec
            )
            draw_pupils(empty_bgr, lm, iris_landmark_spec, 2)

    # Convert BGR canvas back to RGB (MediaPipe draws in BGR here)
    face_map = empty_bgr[:, :, ::-1]
    out = Image.fromarray(face_map)
    out.save(OUTDIR/"04_face_mediapipe_map.png")
    return out

# ───────────────────── Main ─────────────────────
def main(gpu=0):
    device = f"cuda:{gpu}"
    torch.manual_seed(SEED)
    dtype = torch.float16

    # InsightFace for alignment
    face_det = FaceAnalysis(name="antelopev2", root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': gpu}), 'CPUExecutionProvider'])
    face_det.prepare(ctx_id=gpu, det_size=(640,640), det_thresh=0.3)

    # Load + resize
    face_im = to_sd21_res(load_rgb(FACE_IMG), target=512)
    pose_im = to_sd21_res(load_rgb(POSE_IMG), target=512)
    face_im.save(OUTDIR/"00_face_input.png")
    pose_im.save(OUTDIR/"00_pose_input.png")

    # Align face onto pose & composite
    aligned_face, bbox = align_face_with_landmarks(face_im, pose_im, face_det)
    if bbox is None:
        pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
        p = max(face_det.get(pose_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
        bbox = expand_bbox(p['bbox'], 1.35, pose_im.width, pose_im.height)
    mask = soft_face_mask(pose_im, bbox)
    composite = paste_face_into_pose(pose_im, aligned_face, mask, bbox)

    # Control images:
    #   1) MediaPipe face map from COMPOSITE (so the face kps come from your face)
    face_ctrl_img = render_mediapipe_face_map(composite, max_faces=1, min_face_size_px=64)
    #   2) OpenPose kps from POSE image (body + hands only; no face)
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(device)
    pose_kps = openpose(pose_im, include_body=True, include_hand=True, include_face=False)
    pose_kps = pose_kps.resize(pose_im.size, Image.LANCZOS)
    pose_kps.save(OUTDIR/"01_pose_kps_body_hands_only.png")
    del openpose; torch.cuda.empty_cache()

    # Models: MediaPipe-Face + OpenPose (SD 2.1)
    cn_face = ControlNetModel.from_pretrained(CN_FACE, torch_dtype=dtype, variant="fp16")  # ➊
    cn_pose = ControlNetModel.from_pretrained(CN_OPEN, torch_dtype=dtype)
    controlnet = MultiControlNetModel([cn_face, cn_pose])

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_SD21, controlnet=controlnet, safety_checker=None, torch_dtype=dtype
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing(1)
    pipe.enable_model_cpu_offload()

    gen = torch.Generator(device=device).manual_seed(SEED)

    images = [face_ctrl_img, pose_kps]
    scales = [COND_FACE, COND_POSE]

    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEG,
        image=images,
        controlnet_conditioning_scale=scales,
        guidance_scale=CFG,
        num_inference_steps=STEPS,
        generator=gen,
    ).images[0]

    out.save(OUTDIR/"06_final_result.png")
    print(f"✅ Saved intermediates and final image in: {OUTDIR}")

# ───────────────────── CLI ─────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    main(args.gpu)
