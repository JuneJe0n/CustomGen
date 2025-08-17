"""
개선된 버전: pose image에서는 pose keypoints만, face image에서는 face alignment 사용
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

# ─────────────────── 설정 ───────────────────
PROMPT = "a baby sitting, clear facial features, detailed, realistic, smooth colors"
NEG = "(lowres, bad quality, watermark, disjointed, strange limbs, cut off, bad anatomymissing limbs, fused fingers)"
FACE_IMG  = Path("/data2/jeesoo/FFHQ/10000/10000.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/pose/p2.jpeg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/style/s3.png")

CN_HED     = "/data2/jiyoon/custom/ckpts/controlnet-union-sdxl-1.0"
CN_POSE    = "/data2/jiyoon/custom/ckpts/controlnet-openpose-sdxl-1.0"  # pose controlnet 추가
BASE_SDXL  = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE_ENC  = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP   = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

COND_HED     = 0.8
COND_POSE    = 0.6  # pose conditioning strength
STYLE_SCALE  = 0.8
CFG, STEPS   = 7.0, 50
SEED         = 42
OUTDIR       = Path("/data2/jiyoon/custom/results/warp/10000")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────── 유틸 ───────────────────
def load_rgb(p): return Image.open(p).convert("RGB")

def to_sdxl_res(img, base=64, short=1024, long=1024):
    w, h = img.size
    r = short / min(w, h); w, h = int(w*r), int(h*r)
    r = long  / max(w, h); w, h = int(w*r), int(h*r)
    return img.resize(((w//base)*base, (h//base)*base), Image.LANCZOS)

def extract_pose_keypoints(pose_img, pose_detector):
    """Pose image에서 pose keypoints만 추출"""
    print("🤸 Pose keypoints 추출 중...")
    pose_kps = pose_detector(pose_img)
    
    # 결과 저장
    pose_kps.save(OUTDIR / "pose_keypoints.png")
    print(f"💾 Pose keypoints 저장: {OUTDIR / 'pose_keypoints.png'}")
    
    return pose_kps

def align_face_for_hed(face_img, target_size=(512, 512)):
    """Face image에서만 face alignment 수행하여 HED 추출"""
    print("👤 Face alignment 및 HED 추출 중...")
    
    # 얼굴 감지
    face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    face_infos = face_det.get(face_cv)
    
    if not face_infos:
        print("⚠️  얼굴 감지 실패")
        return None, None
    
    # 가장 큰 얼굴 선택
    face_info = max(face_infos, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    bbox = face_info['bbox']
    kps = face_info['kps']
    
    # 얼굴 영역 crop 및 정렬
    x1, y1, x2, y2 = map(int, bbox)
    face_crop = face_img.crop((x1, y1, x2, y2))
    
    # 표준 얼굴 landmarks (정면 얼굴 기준)
    standard_kps = np.array([
        [0.31556875000000000, 0.4615741071428571],  # left eye
        [0.68262291666666670, 0.4615741071428571],  # right eye  
        [0.50026249999999990, 0.6405053571428571],  # nose
        [0.34947187500000004, 0.8246919642857142],  # left mouth
        [0.65343645833333330, 0.8246919642857142]   # right mouth
    ]) * np.array(target_size)
    
    try:
        # crop된 얼굴에서의 상대적 keypoints
        crop_kps = kps - np.array([x1, y1])
        
        # Similarity transform 계산
        tform = SimilarityTransform()
        tform.estimate(crop_kps, standard_kps)
        
        # 얼굴 정렬
        aligned_face = warp(
            np.array(face_crop),
            tform.inverse,
            output_shape=target_size,
            preserve_range=True
        )
        
        aligned_face_pil = Image.fromarray(aligned_face.astype(np.uint8))
        
        # HED 추출
        face_hed = hed(aligned_face_pil, safe=False, scribble=False)
        
        # 결과 저장
        aligned_face_pil.save(OUTDIR / "aligned_face.png")
        face_hed.save(OUTDIR / "face_hed.png")
        print(f"💾 정렬된 얼굴 저장: {OUTDIR / 'aligned_face.png'}")
        print(f"💾 Face HED 저장: {OUTDIR / 'face_hed.png'}")
        
        return aligned_face_pil, face_hed
        
    except Exception as e:
        print(f"⚠️  Face alignment 실패: {e}")
        return None, None

def create_face_region_mask(pose_img, face_det, save_path=None):
    """Pose image에서 얼굴 영역 마스크 생성 (HED 블렌딩용)"""
    
    # MediaPipe로 정밀한 얼굴 마스크 시도
    mp_mask = create_mediapipe_face_mask(pose_img, save_path)
    
    if mp_mask is not None:
        return mp_mask
    
    # MediaPipe 실패시 InsightFace bbox 사용
    print("⚠️  MediaPipe 실패, InsightFace bbox 사용")
    pose_cv = cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR)
    pose_infos = face_det.get(pose_cv)
    
    if not pose_infos:
        print("⚠️  얼굴 감지 실패")
        return None
    
    face_info = max(pose_infos, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    x1, y1, x2, y2 = map(int, face_info['bbox'])
    
    # 부드러운 마스크 생성
    h, w = pose_img.size[::-1]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    
    # 가우시안 블러
    blur_size = max(31, int((x2-x1) * 0.15))
    if blur_size % 2 == 0:
        blur_size += 1
    
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), sigmaX=blur_size//3, sigmaY=blur_size//3)
    
    return mask[..., None]

def create_mediapipe_face_mask(img, save_path=None):
    """MediaPipe Face Mesh로 정밀한 얼굴 마스크 생성"""
    
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
            
            # 얼굴 외곽선 인덱스
            face_oval = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            
            # 마스크 생성
            mask = np.zeros(img_cv.shape[:2], dtype=np.float32)
            points = []
            
            for idx in face_oval:
                x = int(landmarks.landmark[idx].x * img_cv.shape[1])
                y = int(landmarks.landmark[idx].y * img_cv.shape[0])
                points.append([x, y])
            
            cv2.fillPoly(mask, [np.array(points)], 1.0)
            
            # 부드러운 블러
            mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=7, sigmaY=7)
            
            if save_path:
                mask_vis = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_vis, mode='L').save(save_path)
                print(f"💾 Face mask 저장: {save_path}")
            
            return mask[..., None]
    
    return None

def blend_face_hed_with_pose(face_hed, pose_img, face_mask):
    """Face HED를 pose image 얼굴 영역에만 블렌딩"""
    
    # pose image와 같은 크기로 face HED 리사이즈
    face_hed_resized = face_hed.resize(pose_img.size, Image.LANCZOS)
    
    # numpy 배열로 변환
    face_hed_np = np.array(face_hed_resized).astype(np.float32)
    pose_np = np.array(pose_img).astype(np.float32)
    
    # 마스크를 사용하여 블렌딩
    if len(face_hed_np.shape) == 3 and len(face_mask.shape) == 3:
        blended_np = face_mask * face_hed_np + (1 - face_mask) * pose_np
    else:
        # 채널 차원 맞추기
        if len(face_mask.shape) == 3 and face_mask.shape[2] == 1:
            face_mask = np.repeat(face_mask, 3, axis=2)
        blended_np = face_mask * face_hed_np + (1 - face_mask) * pose_np
    
    blended_np = blended_np.clip(0, 255).astype(np.uint8)
    blended_pil = Image.fromarray(blended_np)
    
    # 결과 저장
    blended_pil.save(OUTDIR / "blended_face_pose.png")
    print(f"💾 블렌딩 결과 저장: {OUTDIR / 'blended_face_pose.png'}")
    
    return blended_pil

# ─────────────────── 메인 (개선된 버전) ───────────────────
def main(use_style, gpu_idx):
    global face_det, hed
    
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)

    # ───── 모델 로드
    face_det = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=gpu_idx, det_size=(640, 640))
    
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)

    # ───── 이미지 불러오기
    face_im = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im = to_sdxl_res(load_rgb(POSE_IMG))
    style_pil = load_rgb(STYLE_IMG)

    print("🔄 개선된 파이프라인: Face alignment + Pose keypoints")
    
    # ───── 1. Pose image에서 keypoints 추출
    pose_keypoints = extract_pose_keypoints(pose_im, openpose)
    
    # ───── 2. Face image에서 정렬된 HED 추출
    aligned_face, face_hed = align_face_for_hed(face_im, target_size=(512, 512))
    
    if aligned_face is None or face_hed is None:
        print("❌ Face alignment 실패")
        return
    
    # ───── 3. Pose image에서 얼굴 영역 마스크 생성
    face_mask = create_face_region_mask(
        pose_im, 
        face_det, 
        save_path=OUTDIR / "face_region_mask.png"
    )
    
    if face_mask is None:
        print("❌ Face mask 생성 실패")
        return
    
    # ───── 4. Face HED를 pose image 얼굴 영역에 블렌딩
    blended_result = blend_face_hed_with_pose(face_hed, pose_im, face_mask)
    
    # ───── 5. Multi-ControlNet 구성
    controlnets = [
        ControlNetModel.from_pretrained(CN_POSE, torch_dtype=DTYPE),  # pose
        ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE)    # face HED
    ]
    
    control_images = [pose_keypoints, blended_result]
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

    # ───── 6. 이미지 생성
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

    fname = OUTDIR / "result_face_pose_separated.png"
    out.save(fname)
    print(f"✅ 최종 결과 저장: {fname}")

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", action="store_true", help="IP-Adapter 스타일 주입")
    ap.add_argument("--gpu", type=int, default=0, help="GPU 번호")
    args = ap.parse_args()
    main(args.style, args.gpu)