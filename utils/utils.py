import cv2
import numpy as np
import mediapipe as mp
from PIL import Image


# --- Load ---
def load_rgb(p):
    return Image.open(p).convert("RGB")

def to_sdxl_res(img: Image.Image, base=64, short=1024, long=1024, low_mem=False) -> Image.Image:
    if low_mem:
        short, long = 768, 768
    w, h = img.size
    r = short / min(w, h); w, h = int(w * r), int(h * r)
    r = long  / max(w, h);  w, h = int(w * r), int(h * r)
    return img.resize(((w // base) * base, (h // base) * base), Image.LANCZOS)


# --- Face mesh mask ---
def extract_facemesh_polygon(pil_img: Image.Image, idx_list, W: int, H: int):
    mp_face = mp.solutions.face_mesh
    img_np = np.array(pil_img)
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[int(lm[i].x * W), int(lm[i].y * H)] for i in idx_list], dtype=np.int32)
    return pts

def create_face_mask(face_crop_pil, fw, fh, scale, new_h, new_w):
    # Extract FaceMesh polygon 
    contour_idx = [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ]
    poly_pts = extract_facemesh_polygon(face_crop_pil, contour_idx, fw, fh)
    if poly_pts is None:
        raise RuntimeError("Failed extracting FaceMesh polygon")
    poly_pts_scaled = (poly_pts * scale).astype(np.int32)

    poly_mask = np.zeros((new_h, new_w), dtype=np.float32)
    cv2.fillPoly(poly_mask, [poly_pts_scaled], 1.0)
    poly_mask_3c = np.repeat(poly_mask[:, :, None], 3, axis=2)
    
    return poly_pts_scaled, poly_mask, poly_mask_3c

# --- Save img ---
def to_mask_image(mask01: np.ndarray) -> Image.Image:
    m = (np.clip(mask01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(m, mode="L")



