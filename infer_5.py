"""
5. m3 (face HED_ellipse + pose kps)
"""

import argparse, cv2, torch, numpy as np, re
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import OpenposeDetector, HEDdetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL

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
    SAVE_INTERMEDIATES = False  # 기본: 중간 산출물 저장 안 함
    
use_style = True

# 타원 페더링 파라미터(미정의 시 본문 값으로)
try:
    INNER_BW
except NameError:
    INNER_BW = 24.0
try:
    OUTER_BW
except NameError:
    OUTER_BW = 28.0
try:
    OVERLAP_RATIO
except NameError:
    OVERLAP_RATIO = 0.45
try:
    EDGE_ALPHA
except NameError:
    EDGE_ALPHA = 0.65

_REQUIRED_CFG = [
    "BASE_SDXL", "CN_POSE", "CN_HED",
    "COND_POSE", "COND_HED",
    "FACE_IMG", "POSE_IMG", "STYLE_IMG",
    "PROMPT", "NEG", "CFG", "STEPS", "SEED",
    "OUTDIR", "STYLE_ENC", "STYLE_IP", "STYLE_SCALE",
    "use_style"
]
_missing = [k for k in _REQUIRED_CFG if k not in globals()]
if _missing:
    raise RuntimeError(f"[config.py] 다음 키가 필요합니다: {_missing}")

# ───────── 유틸 ─────────
def to_mask_L(mask01: np.ndarray) -> Image.Image:
    """0~1 float(H,W) -> 8bit 'L' PIL Image"""
    m = (np.clip(mask01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(m, mode="L")

def _ensure_3c(arr: np.ndarray) -> np.ndarray:
    """(H,W) → (H,W,3) 안전 변환"""
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    return arr

def _safe_name(s: str) -> str:
    # 파일명 안전화: 영숫자/._-만 허용
    return re.sub(r"[^A-Za-z0-9._-]+", "", s)

def ellipse_face_masks(W:int, H:int, x1:int, y1:int, x2:int, y2:int,
                       inner_bw:float, outer_bw:float, overlap_ratio:float):
    """
    타원 기반 얼굴 가중치(face_w: 0~1), 겹침 포함 몸 가중치(body_w)
    """
    face_mask_bin = np.zeros((H, W), dtype=np.uint8)
    cx, cy = (x1 + x2)//2, (y1 + y2)//2
    rx, ry = int((x2 - x1)*0.42), int((y2 - y1)*0.48)
    cv2.ellipse(face_mask_bin, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    # 거리 변환(안/밖)
    dist_in  = cv2.distanceTransform(face_mask_bin, cv2.DIST_L2, 5)      # 내부: 중심→경계
    # dist_out = cv2.distanceTransform(255 - face_mask_bin, cv2.DIST_L2, 5) # 외부 사용시

    inner_w = np.clip(dist_in / inner_bw, 0, 1)     # 중심 1 → 경계 0
    face_w  = inner_w.astype(np.float32)
    body_w  = np.clip(1.0 - overlap_ratio * face_w, 0.0, 1.0).astype(np.float32)
    return face_w, body_w

def soften_hed(face_hed_np: np.ndarray, edge_alpha: float) -> np.ndarray:
    """HED 결과를 얇게/블러/대비↓"""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    out = cv2.erode(face_hed_np, ker, iterations=1)
    out = cv2.GaussianBlur(out, (5,5), sigmaX=1.2)
    out = (out * edge_alpha).clip(0,255).astype(np.float32)
    return out

# ───────── 메인 ─────────
def main(gpu_idx: int):
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 최종 파일명 조합: <face>_<pose>_<style>.png
    face_stem  = Path(FACE_IMG).stem
    pose_stem  = Path(POSE_IMG).stem
    style_stem = Path(STYLE_IMG).stem
    fname_base = _safe_name(f"{face_stem}_{pose_stem}_{style_stem}")
    final_path = OUTDIR / f"{fname_base}.png"

    # 입력 로드/리사이즈
    face_im   = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im   = to_sdxl_res(load_rgb(POSE_IMG))
    style_pil = load_rgb(STYLE_IMG)
    W, H = pose_im.size

    if SAVE_INTERMEDIATES:
        face_im.save(OUTDIR / "0_face_input.png")
        pose_im.save(OUTDIR / "1_pose_input.png")
        style_pil.save(OUTDIR / "2_style_input.png")

    # 얼굴 검출기 + 컨디셔너
    face_det = FaceAnalysis(
        name="antelopev2",
        root=str(FACE_DET_ROOT),
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=gpu_idx, det_size=(640, 640))

    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    hed      = HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)

    # pose에서 얼굴 bbox
    pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
    p_faces = face_det.get(pose_cv)
    if not p_faces:
        raise RuntimeError("[m4] pose 이미지에서 얼굴을 찾지 못했습니다.")
    p_info = max(p_faces, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    x1, y1, x2, y2 = map(int, p_info['bbox'])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    pw, ph = x2 - x1, y2 - y1

    # 전체 OpenPose (전신+손+얼굴) → 몸 중심
    pose_openpose_pil = openpose(pose_im, hand_and_face=True).resize((W, H), Image.LANCZOS)
    pose_openpose_np  = _ensure_3c(np.array(pose_openpose_pil).astype(np.float32))

    # face 이미지에서 얼굴 bbox
    face_cv_full = cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)
    f_faces = face_det.get(face_cv_full)
    if not f_faces:
        raise RuntimeError("[m4] face 이미지에서 얼굴을 찾지 못했습니다.")
    f_info = max(f_faces, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    fx1, fy1, fx2, fy2 = map(int, f_info['bbox'])
    face_crop_pil = face_im.crop((fx1, fy1, fx2, fy2))

    # 얼굴 HED 생성 → 리사이즈 → 소프트닝
    face_hed_crop_pil = hed(face_crop_pil, safe=False, scribble=False)
    face_hed_resized  = face_hed_crop_pil.resize((pw, ph), Image.LANCZOS)
    face_hed_np       = _ensure_3c(np.array(face_hed_resized).astype(np.float32))
    face_hed_np       = soften_hed(face_hed_np, EDGE_ALPHA)

    # 얼굴 HED 캔버스(전역)
    face_hed_canvas_np = np.zeros_like(pose_openpose_np, dtype=np.float32)
    face_hed_canvas_np[y1:y2, x1:x2] = face_hed_np
    face_hed_canvas_pil = Image.fromarray(face_hed_canvas_np.clip(0,255).astype(np.uint8)).convert("RGB")

    if SAVE_INTERMEDIATES:
        Image.fromarray(pose_openpose_np.clip(0,255).astype(np.uint8)).save(OUTDIR/"cond_body_openpose_raw.png")
        face_hed_canvas_pil.save(OUTDIR/"cond_hed_soft.png")

    # 타원+거리기반 페더 마스크
    face_w, body_w = ellipse_face_masks(W, H, x1, y1, x2, y2, INNER_BW, OUTER_BW, OVERLAP_RATIO)
    face_mask_L = to_mask_L(face_w)
    body_mask_L = to_mask_L(body_w)

    if SAVE_INTERMEDIATES:
        face_mask_L.save(OUTDIR/"mask_face_elliptic.png")
        body_mask_L.save(OUTDIR/"mask_body_overlap.png")

    # ── Multi-ControlNet: A) OpenPose(몸, body_mask) + B) HED(얼굴, face_mask)
    controlnets, images, scales, masks = [], [], [], []

    # A) OpenPose (몸)
    controlnets.append(ControlNetModel.from_pretrained(CN_POSE, torch_dtype=DTYPE))
    images.append(pose_openpose_pil)
    scales.append(COND_POSE)
    masks.append(body_mask_L)   # 몸 영역만 활성

    # B) HED (얼굴)
    controlnets.append(ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE))
    images.append(face_hed_canvas_pil)
    scales.append(COND_HED)
    masks.append(face_mask_L)   # 얼굴 영역만 활성

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
        prompt=PROMPT,
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        image=images,
        controlnet_conditioning_scale=scales,
        control_mask=masks,  # 멀티 ControlNet과 1:1 매칭
    )

    if use_style:
        ip = IPAdapterXL(
            pipe, STYLE_ENC, STYLE_IP, DEVICE,
            target_blocks=["up_blocks.0.attentions.1"]
        )
        out = ip.generate(pil_image=style_pil, scale=STYLE_SCALE, seed=SEED, **gen_args)[0]
    else:
        out = pipe(**gen_args).images[0]

    out.save(final_path)
    print(f"✅ saved → {final_path}")

# ───────── CLI ─────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES 안에서 논리 GPU 번호")
    args = ap.parse_args()
    main(args.gpu)
