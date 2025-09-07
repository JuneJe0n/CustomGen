"""
3. m2 (face kps + pose kps)
"""

import argparse, cv2, torch, numpy as np, re
from pathlib import Path
from PIL import Image

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import OpenposeDetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL

# 공통 설정/유틸
from config import *        
from utils import *         

# ───────── 이 파일 내부에서 직접 정의하는 추가/재정의 설정 ─────────

FACE_DET_ROOT = "/data2/jiyoon/InstantID"
ERASE_ORIGINAL_FACE = True
use_style = True  # False면 IP-Adapter 없이 pipe(**gen_args)

# 중간 산출물 저장 여부 (입력/머지 이미지 저장)
SAVE_INTERMEDIATES = False  # ← 요청사항: 기본 False

# 방어적 체크
_required_cfg = ["BASE_SDXL", "CN_POSE", "COND_POSE", "FACE_IMG", "POSE_IMG",
                 "STYLE_IMG", "PROMPT", "NEG", "CFG", "STEPS", "SEED", "OUTDIR",
                 "STYLE_ENC", "STYLE_IP", "STYLE_SCALE", "use_style"]
_missing = [k for k in _required_cfg if k not in globals()]
if _missing:
    raise RuntimeError(f"[config] 필요한 키 누락: {_missing}")

def _safe_name(s: str) -> str:
    # 파일명 안전화: 영숫자/._-만 허용
    return re.sub(r"[^A-Za-z0-9._-]+", "", s)

def main(gpu_idx: int):
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 파일명 조합용 stem 추출
    face_stem  = Path(FACE_IMG).stem
    pose_stem  = Path(POSE_IMG).stem
    style_stem = Path(STYLE_IMG).stem
    fname_base = _safe_name(f"{face_stem}_{pose_stem}_{style_stem}")

    # 입력 로드 & 리사이즈
    face_im  = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im  = to_sdxl_res(load_rgb(POSE_IMG))
    if SAVE_INTERMEDIATES:
        face_im.save(OUTDIR / "0_face_input.png")
        pose_im.save(OUTDIR / "1_pose_input.png")
        load_rgb(STYLE_IMG).save(OUTDIR / "2_style_input.png")  # 보기용 저장

    w_pose, h_pose = pose_im.size

    # 얼굴 검출기 & OpenPose Detector
    face_det = FaceAnalysis(
        name="antelopev2",
        root=str(FACE_DET_ROOT),
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=gpu_idx, det_size=(640, 640))

    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)

    # pose 이미지 얼굴 bbox
    pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
    p_faces = face_det.get(pose_cv)
    if not p_faces:
        raise RuntimeError("[m2] pose 이미지에서 얼굴을 찾지 못했습니다.")
    p_info = max(p_faces, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    x1, y1, x2, y2 = map(int, p_info['bbox'])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_pose, x2), min(h_pose, y2)
    pw, ph = x2 - x1, y2 - y1

    # pose 전체 OpenPose (전신+손+얼굴)
    pose_openpose_pil = openpose(pose_im, hand_and_face=True).resize(pose_im.size, Image.LANCZOS)
    pose_openpose_np  = np.array(pose_openpose_pil).astype(np.float32)   # (H, W, 3)

    # face 이미지 → 얼굴 bbox → 얼굴만 OpenPose → pose 크기에 맞게 리사이즈
    face_cv_full = cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)
    f_faces = face_det.get(face_cv_full)
    if not f_faces:
        raise RuntimeError("[m2] face 이미지에서 얼굴을 찾지 못했습니다.")
    f_info = max(f_faces, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    fx1, fy1, fx2, fy2 = map(int, f_info['bbox'])
    face_crop_pil = face_im.crop((fx1, fy1, fx2, fy2))

    face_openpose_crop_pil = openpose(face_crop_pil, hand_and_face=True)
    face_openpose_resized  = face_openpose_crop_pil.resize((pw, ph), Image.LANCZOS)
    face_openpose_np       = np.array(face_openpose_resized).astype(np.float32)

    # 소프트 마스크(가우시안)
    mask = np.zeros((h_pose, w_pose), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 31), sigmaX=10, sigmaY=10)[..., None]   # (H, W, 1)

    # 얼굴 OpenPose를 pose OpenPose에 합성
    blended_src    = pose_openpose_np.copy()
    face_canvas_np = np.zeros_like(pose_openpose_np, dtype=np.float32)
    face_canvas_np[y1:y2, x1:x2] = face_openpose_np

    if ERASE_ORIGINAL_FACE:
        blended_src[y1:y2, x1:x2] = 0.0  # 기존 얼굴 라인 제거

    blended_np = mask * face_canvas_np + (1.0 - mask) * blended_src
    blended_np = blended_np.clip(0, 255).astype(np.uint8)
    merged_pose_pil = Image.fromarray(blended_np).convert("RGB")
    if SAVE_INTERMEDIATES:
        merged_pose_pil.save(OUTDIR / "3_merged.png")

    # ControlNet(OpenPose)
    controlnets, images, scales, masks = [], [], [], []
    controlnets.append(ControlNetModel.from_pretrained(CN_POSE, torch_dtype=DTYPE))
    images.append(merged_pose_pil)
    scales.append(COND_POSE)
    masks.append(None)

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
        control_mask=masks,
    )

    if use_style:
        style_pil = load_rgb(STYLE_IMG)
        ip = IPAdapterXL(
            pipe, STYLE_ENC, STYLE_IP, DEVICE,
            target_blocks=["up_blocks.0.attentions.1"]
        )
        out = ip.generate(pil_image=style_pil, scale=STYLE_SCALE, seed=SEED, **gen_args)[0]
    else:
        out = pipe(**gen_args).images[0]

    # 최종 파일명: <face>_<pose>_<style>.png
    final_path = OUTDIR / f"{fname_base}.png"
    out.save(final_path)
    print("saved →", final_path)

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES 안에서 논리 GPU 번호")
    args = ap.parse_args()
    main(args.gpu)
