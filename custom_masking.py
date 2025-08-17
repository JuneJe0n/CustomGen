"""
가장 단순한 접근법:
1. 원본 face 이미지에서 HED 추출
2. pose 이미지에서 HED 추출  
3. face kps 기준으로 마스킹만 하고 HED 삽입
워핑 없음, 변형 없음, 그냥 마스킹만!
"""
import argparse, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import HEDdetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL
import mediapipe as mp

# ─────────────────── 설정 ───────────────────
PROMPT = "a baby sitting, clear facial features, detailed, realistic, smooth colors"
NEG = "(lowres, bad quality, watermark, disjointed, strange limbs, cut off, bad anatomy, missing limbs, fused fingers)"
FACE_IMG  = Path("/data2/jeesoo/FFHQ/00000/00000.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/pose/p2.jpeg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/style/s3.png")

CN_HED     = "/data2/jiyoon/custom/ckpts/controlnet-union-sdxl-1.0"
BASE_SDXL  = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE_ENC  = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP   = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

COND_HED     = 0.8
STYLE_SCALE  = 0.8
CFG, STEPS   = 7.0, 50
SEED         = 42
OUTDIR       = Path("/data2/jiyoon/custom/results/claude/00000/1")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────── 유틸 ───────────────────
def load_rgb(p): 
    return Image.open(p).convert("RGB")

def to_sdxl_res(img, base=64, short=1024, long=1024):
    w, h = img.size
    r = short / min(w, h); w, h = int(w*r), int(h*r)
    r = long  / max(w, h); w, h = int(w*r), int(h*r)
    return img.resize(((w//base)*base, (h//base)*base), Image.LANCZOS)

def create_face_kps_mask(img, face_det, save_path=None):
    """얼굴 kps 기준으로 간단한 마스크 생성"""
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = face_det.get(img_cv)
    
    if not faces:
        print("⚠️ 얼굴 감지 실패")
        return None
    
    h, w = img.size[::-1]
    mask = np.zeros((h, w), dtype=np.float32)
    
    # 가장 큰 얼굴 선택
    face_info = max(faces, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    
    # 방법 1: bbox 기반 (가장 단순)
    bbox = face_info['bbox']
    x1, y1, x2, y2 = map(int, bbox)
    
    # 약간 확장
    margin_w = int((x2 - x1) * 0.2)
    margin_h = int((y2 - y1) * 0.2)
    
    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(w, x2 + margin_w)
    y2 = min(h, y2 + margin_h)
    
    # 사각형 마스크 생성
    mask[y1:y2, x1:x2] = 1.0
    
    # 부드러운 경계
    mask = cv2.GaussianBlur(mask, (51, 51), sigmaX=20, sigmaY=20)
    
    # 시각화 저장
    if save_path:
        mask_vis = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_vis, mode='L').save(save_path)
        print(f"💾 KPS 마스크 저장: {save_path}")
    
    return mask[..., None], (x1, y1, x2, y2)

def create_mediapipe_kps_mask(img, save_path=None):
    """MediaPipe 기반 얼굴 마스크 (더 정밀)"""
    
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
            h, w = img_cv.shape[:2]
            
            # 얼굴 윤곽선
            face_oval = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            
            mask = np.zeros((h, w), dtype=np.float32)
            points = []
            
            for idx in face_oval:
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                points.append([x, y])
            
            cv2.fillPoly(mask, [np.array(points)], 1.0)
            
            # 멀티 스테이지 블러링
            mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=7)
            mask = cv2.GaussianBlur(mask, (41, 41), sigmaX=15)
            
            # bbox도 계산
            points_np = np.array(points)
            x_min, y_min = points_np.min(axis=0)
            x_max, y_max = points_np.max(axis=0)
            
            margin = 30
            bbox = (
                max(0, x_min - margin),
                max(0, y_min - margin), 
                min(w, x_max + margin),
                min(h, y_max + margin)
            )
            
            if save_path:
                mask_vis = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_vis, mode='L').save(save_path)
                print(f"💾 MediaPipe KPS 마스크 저장: {save_path}")
            
            return mask[..., None], bbox
    
    return None, None

def simple_hed_insert(face_img, pose_img, face_det, hed_detector):
    """가장 단순한 HED 삽입 - 워핑 없음!"""
    
    print("🎯 단순 HED 삽입 (워핑 없음)")
    
    # 1. 각각에서 HED 추출
    face_hed = hed_detector(face_img, safe=False, scribble=False)
    pose_hed = hed_detector(pose_img, safe=False, scribble=False)
    
    # 크기 맞춤
    face_hed = face_hed.resize(pose_img.size, Image.LANCZOS)
    pose_hed = pose_hed.resize(pose_img.size, Image.LANCZOS)
    
    face_hed.save(OUTDIR / "face_hed_original.png")
    pose_hed.save(OUTDIR / "pose_hed_original.png")
    
    # 2. pose 이미지에서 얼굴 마스크 생성
    # 먼저 MediaPipe 시도
    mp_mask, mp_bbox = create_mediapipe_kps_mask(pose_img, OUTDIR / "mp_mask.png")
    
    if mp_mask is not None:
        print("✅ MediaPipe 마스크 사용")
        mask = mp_mask
        bbox = mp_bbox
    else:
        print("⚠️ MediaPipe 실패, InsightFace 마스크 사용")
        if_mask, if_bbox = create_face_kps_mask(pose_img, face_det, OUTDIR / "if_mask.png")
        if if_mask is not None:
            mask = if_mask
            bbox = if_bbox
        else:
            print("❌ 모든 마스크 생성 실패")
            return pose_hed
    
    # 3. 단순 블렌딩 (워핑 없음!)
    face_hed_np = np.array(face_hed).astype(np.float32)
    pose_hed_np = np.array(pose_hed).astype(np.float32)
    
    # 마스크 강도 조절
    mask_strength = 0.8
    adjusted_mask = mask * mask_strength
    
    # 블렌딩
    result = adjusted_mask * face_hed_np + (1 - adjusted_mask) * pose_hed_np
    result = result.astype(np.uint8)
    
    result_pil = Image.fromarray(result).convert("RGB")
    result_pil.save(OUTDIR / "simple_hed_insert.png")
    print(f"💾 단순 HED 삽입 결과: {OUTDIR / 'simple_hed_insert.png'}")
    
    return result_pil

# ─────────────────── 메인 함수 ───────────────────
def main(use_style, gpu_idx):
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE = torch.float16
    torch.manual_seed(SEED)

    print("🚀 초단순 KPS 마스킹 파이프라인")

    # 도구들 로드
    face_det = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID", 
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=gpu_idx, det_size=(640, 640))
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)

    # 이미지 로드
    face_im = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im = to_sdxl_res(load_rgb(POSE_IMG))
    style_pil = load_rgb(STYLE_IMG)

    # ───── 핵심: 단순 HED 삽입
    final_hed = simple_hed_insert(face_im, pose_im, face_det, hed)

    # ───── ControlNet 생성
    controlnets = [ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE)]
    
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_SDXL,
        controlnet=MultiControlNetModel(controlnets),
        torch_dtype=DTYPE,
        add_watermarker=False
    ).to(DEVICE)

    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()

    # 메모리 최적화 (스타일 사용 여부에 관계없이)
    if not use_style:
        pipe.enable_sequential_cpu_offload()
    # else: GPU로 이동은 IPAdapterXL에서 처리

    # ───── 생성
    gen_args = dict(
        prompt=PROMPT,
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        image=[final_hed],
        controlnet_conditioning_scale=[COND_HED],
        control_mask=[None],
    )

    if use_style:
        ip = IPAdapterXL(
            pipe, STYLE_ENC, STYLE_IP, DEVICE,
            target_blocks=["up_blocks.0.attentions.1"]
        )
        out = ip.generate(pil_image=style_pil, scale=STYLE_SCALE, seed=SEED, **gen_args)[0]
    else:
        out = pipe(**gen_args).images[0]

    fname = OUTDIR / "simple_kps_result.png"
    out.save(fname)
    print(f"🎉 초단순 KPS 결과: {fname}")
    
    print("\n📁 생성된 파일들:")
    print(f"   • 원본 Face HED: {OUTDIR / 'face_hed_original.png'}")
    print(f"   • 원본 Pose HED: {OUTDIR / 'pose_hed_original.png'}")
    print(f"   • 마스크: {OUTDIR / 'mp_mask.png'} 또는 {OUTDIR / 'if_mask.png'}")
    print(f"   • HED 삽입 결과: {OUTDIR / 'simple_hed_insert.png'}")
    print(f"   • 최종 결과: {fname}")

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simple KPS Masking Pipeline")
    ap.add_argument("--style", action="store_true", help="IP-Adapter 스타일 주입")
    ap.add_argument("--gpu", type=int, default=0, help="GPU 번호")
    args = ap.parse_args()
    main(args.style, args.gpu)