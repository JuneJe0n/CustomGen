import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from skimage.transform import SimilarityTransform, warp

from config import OUTDIR

# --- Load ---
def load_rgb(p):
    return Image.open(p).convert("RGB")

def to_sdxl_res(img, base=64, short=1024, long=1024, low_mem=False):
    if low_mem:
        short, long = 768, 768
    w, h = img.size
    r = short / min(w, h); w, h = int(w*r), int(h*r)
    r = long  / max(w, h); w, h = int(w*r), int(h*r)
    return img.resize(((w//base)*base, (h//base)*base), Image.LANCZOS)


# --- Align ---
def expand_bbox(bbox, scale, W, H):
    x1, y1, x2, y2 = map(int, bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    nx1, ny1 = int(max(0, cx - w/2)), int(max(0, cy - h/2))
    nx2, ny2 = int(min(W, cx + w/2)), int(min(H, cy + h/2))
    return [nx1, ny1, nx2, ny2]

def align_face_with_landmarks(face_img, pose_img, face_det):
    face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    pose_cv = cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR)
    face_infos = face_det.get(face_cv)
    pose_infos = face_det.get(pose_cv)
    if not face_infos or not pose_infos:
        return face_img, None
    
    # Extract kps
    face_info = max(face_infos, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    pose_info = max(pose_infos, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    face_kps, pose_kps = face_info['kps'], pose_info['kps']
    try:
        # Align
        tform = SimilarityTransform()
        tform.estimate(face_kps, pose_kps) 

        # Save aligned face
        h, w = pose_img.size[::-1]
        aligned_face = warp(np.array(face_img), tform.inverse, output_shape=(h, w), preserve_range=True)
        aligned_face_pil = Image.fromarray(aligned_face.astype(np.uint8))
        aligned_face_pil.save(OUTDIR/"3_aligned_face.png")

        pose_bbox_expanded = expand_bbox(pose_info['bbox'], scale=1.35, W=w, H=h)
        return aligned_face_pil, pose_bbox_expanded
    except Exception:
        return face_img, None


# --- Mediapipe face mask ---
def create_mediapipe_face_mask(img):
    mp_face_mesh = mp.solutions.face_mesh
    img_rgb = np.array(img.convert("RGB"))

    with mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=False,
            min_detection_confidence=0.3
            ) as fm:
        results = fm.process(img_rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            face_oval = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            h, w = img_rgb.shape[:2]
            mask = np.zeros((h,w), dtype=np.float32)
            pts = [(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)) for i in face_oval]
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)

            # mask
            mask = cv2.GaussianBlur(mask, (21,21), 7)
            Image.fromarray((mask*255).astype(np.uint8)).save(OUTDIR/"5_mediapipe_mask.png")
            return mask[...,None]
    return None

def create_enhanced_soft_mask(pose_img, bbox):
    h, w = pose_img.size[::-1]
    x1,y1,x2,y2 = map(int, bbox)
    mp_mask = create_mediapipe_face_mask(pose_img)

    if mp_mask is not None:
        roi = np.zeros_like(mp_mask)
        roi[y1:y2,x1:x2] = mp_mask[y1:y2,x1:x2]
        return roi
    else:
        # Fallback to bbox
        print("🚨 Facemesh not detected.. Fallback to naive bbox mask")
        mask = np.zeros((h,w),dtype=np.float32)
        mask[y1:y2,x1:x2]=1.0
        mask = cv2.GaussianBlur(mask,(51,51),15)
        Image.fromarray((mask*255).astype(np.uint8)).save(OUTDIR/"bbox_mask.png")
        return mask[...,None]


# --- Make hed condition ---
def blend_face_hed_face_only(face_hed, pose_img, face_mask, bbox):
    # Compute the face region in pose
    x1,y1,x2,y2 = bbox
    tw,th = x2-x1,y2-y1

    # Resize hed accordingly
    face_hed_resized = face_hed.resize((tw,th), Image.LANCZOS)
    hed_np = np.array(face_hed_resized)
    if hed_np.ndim==2: 
        hed_np = np.stack([hed_np]*3,axis=2)
    H,W = pose_img.height, pose_img.width

    # Create canvas and paste the resized hed in the face region
    canvas = np.zeros((H,W,3),dtype=np.float32)
    canvas[y1:y2,x1:x2]=hed_np[:th,:tw]

    # Multiply the canvas w the face mask
    if face_mask is not None:
        if face_mask.ndim==3 and face_mask.shape[2]==1:
            face_mask=np.repeat(face_mask,3,axis=2)
        canvas *= face_mask 

    result = Image.fromarray(np.clip(canvas,0,255).astype(np.uint8)).convert("RGB")
    result.save(OUTDIR/"6_hed_resized.png")
    return result


# --- Build composite canvas ---
def paste_face_into_pose(pose_img: Image.Image, aligned_face: Image.Image, mask: np.ndarray, bbox):
    x1,y1,x2,y2 = bbox
    face_region = aligned_face.crop((x1,y1,x2,y2))
    face_np = np.array(face_region).astype(np.float32)
    pose_np = np.array(pose_img).astype(np.float32)
    m = mask
    if m is None:
        m = np.zeros((pose_img.height, pose_img.width, 1), dtype=np.float32)
        m[y1:y2, x1:x2, 0] = 1.0
        m = cv2.GaussianBlur(m, (51,51), 15)
    if m.shape[-1] == 1:
        m = np.repeat(m, 3, axis=2)
    out = pose_np.copy()
    out[y1:y2, x1:x2] = m[y1:y2, x1:x2]*face_np + (1.0 - m[y1:y2, x1:x2])*pose_np[y1:y2, x1:x2]
    comp = Image.fromarray(np.clip(out,0,255).astype(np.uint8))
    comp.save(OUTDIR/"5_composite_canvas.png")
    return comp


# --- Extract kps ---

def extract_pose_keypoints(img, pose_detector, include_body=True, include_hand=True, include_face=False, save_name="01_pose_kps_body_hands_only.png"):
    """
    Default include_face=False (i.e., remove facial kps).
    """
    kps = pose_detector(
        img,
        include_body=include_body,
        include_hand=include_hand,
        include_face=include_face
    )
    kps = kps.resize(img.size, Image.LANCZOS)
    kps.save(OUTDIR / save_name)
    return kps








