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
                 "STYLE_IMG", "NEG", "CFG", "STEPS", "SEED", "OUTDIR",
                 "STYLE_ENC", "STYLE_IP", "STYLE_SCALE", "use_style"]
_missing = [k for k in _required_cfg if k not in globals()]
if _missing:
    raise RuntimeError(f"[config] 필요한 키 누락: {_missing}")

def _safe_name(s: str) -> str:
    # 파일명 안전화: 영숫자/._-만 허용
    return re.sub(r"[^A-Za-z0-9._-]+", "", s)

def main(face_img_path: str, pose_img_path: str, style_img_path: str, output_path: str, gpu_idx: int = 0):
    # Set GPU - use cuda:0 when CUDA_VISIBLE_DEVICES is set, otherwise use specified gpu_idx
    import os
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        DEVICE = "cuda:0"
    else:
        DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)
    
    # Set output path
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # 입력 로드 & 리사이즈
    face_im  = to_sdxl_res(load_rgb(face_img_path))
    pose_im  = to_sdxl_res(load_rgb(pose_img_path))
    
    # Generate prompt based on input images
    from utils import PromptGenerator
    generator = PromptGenerator()
    prompt = generator.generate_combined_prompt(face_img_path, pose_img_path)
    if SAVE_INTERMEDIATES:
        face_im.save(OUTDIR / "0_face_input.png")
        pose_im.save(OUTDIR / "1_pose_input.png")
        load_rgb(style_img_path).save(OUTDIR / "2_style_input.png")  # 보기용 저장

    w_pose, h_pose = pose_im.size

    # 얼굴 검출기 & OpenPose Detector
    # Use device 0 when CUDA_VISIBLE_DEVICES is set, otherwise use gpu_idx
    device_id = 0 if 'CUDA_VISIBLE_DEVICES' in os.environ else gpu_idx
    face_det = FaceAnalysis(
        name="antelopev2",
        root=str(FACE_DET_ROOT),
        providers=[('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=device_id, det_size=(640, 640))

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
        prompt=prompt,
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        image=images,
        controlnet_conditioning_scale=scales,
        control_mask=masks,
    )

    if use_style:
        style_pil = load_rgb(style_img_path)
        ip = IPAdapterXL(
            pipe, STYLE_ENC, STYLE_IP, DEVICE,
            target_blocks=["up_blocks.0.attentions.1"]
        )
        out = ip.generate(pil_image=style_pil, scale=STYLE_SCALE, seed=SEED, **gen_args)[0]
    else:
        out = pipe(**gen_args).images[0]

    out.save(final_path)
    print(f"✅ Saved final result to {final_path}")
    
    # Clear GPU memory
    if 'ip' in locals():
        del ip
    del pipe
    torch.cuda.empty_cache()
    
    return True

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--face_img", type=str, required=True, help="Path to face image")
    ap.add_argument("--pose_img", type=str, required=True, help="Path to pose image")
    ap.add_argument("--style_img", type=str, required=True, help="Path to style image")
    ap.add_argument("--output_path", type=str, required=True, help="Output path for final result")
    ap.add_argument("--gpu", type=int, default=2, help="GPU index to use")
    args = ap.parse_args()
    
    try:
        success = main(args.face_img, args.pose_img, args.style_img, args.output_path, args.gpu)
        if success:
            exit(0)
        else:
            exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
