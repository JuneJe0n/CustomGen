"""
4. m3 (face HED_bbox + pose kps)
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
    SAVE_INTERMEDIATES = False  # 요청: 기본적으로 중간 산출물 저장하지 않음

use_style = True  # False면 IP-Adapter 없이 pipe(**gen_args)

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
def to_mask_image(mask01: np.ndarray) -> Image.Image:
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

    # pose 이미지에서 얼굴 bbox 추출
    pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
    p_faces = face_det.get(pose_cv)
    if not p_faces:
        raise RuntimeError("[m2] pose 이미지에서 얼굴을 찾지 못했습니다.")
    p_info = max(p_faces, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    x1, y1, x2, y2 = map(int, p_info['bbox'])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    pw, ph = x2 - x1, y2 - y1

    # 전신 OpenPose (몸/손/얼굴 포함)
    pose_openpose_pil = openpose(pose_im, hand_and_face=True).resize((W, H), Image.LANCZOS)
    pose_openpose_np  = _ensure_3c(np.array(pose_openpose_pil).astype(np.float32))

    # face 이미지에서 얼굴 bbox 추출
    face_cv_full = cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)
    f_faces = face_det.get(face_cv_full)
    if not f_faces:
        raise RuntimeError("[m2] face 이미지에서 얼굴을 찾지 못했습니다.")
    f_info = max(f_faces, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    fx1, fy1, fx2, fy2 = map(int, f_info['bbox'])
    face_crop_pil = face_im.crop((fx1, fy1, fx2, fy2))

    # 얼굴 HED → pose 얼굴 bbox 크기 → 전체 캔버스에 위치
    face_hed_crop_pil = hed(face_crop_pil, safe=False, scribble=False)
    face_hed_resized  = face_hed_crop_pil.resize((pw, ph), Image.LANCZOS)
    face_hed_np       = _ensure_3c(np.array(face_hed_resized).astype(np.float32))

    face_hed_canvas_np = np.zeros_like(pose_openpose_np, dtype=np.float32)  # (H,W,3)
    face_hed_canvas_np[y1:y2, x1:x2] = face_hed_np
    face_hed_canvas_pil = Image.fromarray(face_hed_canvas_np.clip(0,255).astype(np.uint8)).convert("RGB")

    if SAVE_INTERMEDIATES:
        pose_openpose_pil.save(OUTDIR / "cond_body_openpose_raw.png")
        face_hed_canvas_pil.save(OUTDIR / "cond_hed.png")

    # 마스크(소프트 경계): face / body
    face_mask = np.zeros((H, W), dtype=np.float32)
    face_mask[y1:y2, x1:x2] = 1.0
    face_mask = cv2.GaussianBlur(face_mask, (31,31), sigmaX=10, sigmaY=10)
    body_mask = (1.0 - face_mask).astype(np.float32)

    if SAVE_INTERMEDIATES:
        to_mask_image(face_mask).save(OUTDIR / "mask_face.png")
        to_mask_image(body_mask).save(OUTDIR / "mask_body.png")

    # ── Multi-ControlNet 구성: A) OpenPose(몸, body_mask) + B) HED(얼굴, face_mask)
    controlnets, images, scales, masks = [], [], [], []

    # A) OpenPose (몸)
    controlnets.append(ControlNetModel.from_pretrained(CN_POSE, torch_dtype=DTYPE))
    images.append(pose_openpose_pil)
    scales.append(COND_POSE)
    masks.append(to_mask_image(body_mask))   # 몸 영역만 활성

    # B) HED (얼굴)
    controlnets.append(ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE))
    images.append(face_hed_canvas_pil)
    scales.append(COND_HED)
    masks.append(to_mask_image(face_mask))   # 얼굴 영역만 활성

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
        control_mask=masks,  # 사용자 환경의 패치된 SDXL-ControlNet 파이프에서 사용
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
    print(f"saved -> {final_path}")

# ───────── CLI ─────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES 안에서 논리 GPU 번호")
    args = ap.parse_args()
    main(args.gpu)
