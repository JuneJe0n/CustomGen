"""
Use face lmks, face mesh to align face
Extract HED from aligned face & pose
"""
import argparse, cv2, torch, numpy as np
import gc
from pathlib import Path
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import HEDdetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL
from skimage.transform import SimilarityTransform, warp
import mediapipe as mp

# ─────────────────── 설정 ───────────────────
PROMPT = "a baby sitting, clear facial features, detailed, realistic, smooth colors"
NEG = "(lowres, bad quality, watermark, disjointed, strange limbs, cut off, bad anatomymissing limbs, fused fingers)"
FACE_IMG  = Path("/data2/jeesoo/FFHQ/64000/64000.png")
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
OUTDIR       = Path("/data2/jiyoon/custom/results/warp/64000")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────── 유틸 ───────────────────
def load_rgb(p): return Image.open(p).convert("RGB")

def to_sdxl_res(img, base=64, short=1024, long=1024):
    w, h = img.size
    r = short / min(w, h); w, h = int(w*r), int(h*r)
    r = long  / max(w, h); w, h = int(w*r), int(h*r)
    return img.resize(((w//base)*base, (h//base)*base), Image.LANCZOS)

def align_face_with_landmarks(face_img, pose_img, face_det):
    """InsightFace 랜드마크를 이용한 얼굴 정렬"""
    
    face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    pose_cv = cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR)
    
    # 얼굴 정보 추출
    face_infos = face_det.get(face_cv)
    pose_infos = face_det.get(pose_cv)
    
    if not face_infos or not pose_infos:
        print("⚠️  얼굴 감지 실패, 기존 방법 사용")
        return face_img, None
    
    # 가장 큰 얼굴 선택
    face_info = max(face_infos, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    pose_info = max(pose_infos, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    
    # 5개 핵심 랜드마크 추출
    face_kps = face_info['kps']  # (5, 2)
    pose_kps = pose_info['kps']
    
    try:
        # Similarity transform 계산 (rotation, scaling, translation)
        tform = SimilarityTransform()
        tform.estimate(face_kps, pose_kps)
        
        # face 이미지를 pose에 맞게 변형
        h, w = pose_img.size[::-1]
        aligned_face = warp(
            np.array(face_img), 
            tform.inverse, 
            output_shape=(h, w),
            preserve_range=True
        )
        
        aligned_face_pil = Image.fromarray(aligned_face.astype(np.uint8))
        
        # 변형된 얼굴의 새로운 bbox 계산
        transformed_kps = tform(face_kps)
        x_coords = transformed_kps[:, 0]
        y_coords = transformed_kps[:, 1]
        
        margin = 30  # bbox 확장
        new_bbox = [
            max(0, int(x_coords.min() - margin)),
            max(0, int(y_coords.min() - margin)),
            min(w, int(x_coords.max() + margin)),
            min(h, int(y_coords.max() + margin))
        ]
        
        return aligned_face_pil, new_bbox
        
    except Exception as e:
        print(f"⚠️  랜드마크 정렬 실패: {e}, 기존 방법 사용")
        return face_img, None

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
            
            # 얼굴 외곽선 인덱스 (FACEMESH_FACE_OVAL)
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
            
            # 가우시안 블러로 부드럽게
            mask = cv2.GaussianBlur(mask, (15, 15), sigmaX=5, sigmaY=5)
            
            # 마스크 시각화 저장
            if save_path:
                mask_vis = (mask * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_vis, mode='L')
                mask_pil.save(save_path)
                print(f"💾 MediaPipe 마스크 저장: {save_path}")
            
            return mask[..., None]  # (H, W, 1)
    
    return None

def create_enhanced_soft_mask(pose_img, bbox, save_dir=None):
    """향상된 소프트 마스크 생성"""
    
    h, w = pose_img.size[::-1]
    x1, y1, x2, y2 = bbox
    
    # MediaPipe로 정밀한 얼굴 마스크 시도
    mp_mask_path = save_dir / "00_mediapipe_mask.png" if save_dir else None
    mp_mask = create_mediapipe_face_mask(pose_img, mp_mask_path)
    
    if mp_mask is not None:
        # 얼굴 영역만 추출하여 반환
        roi_mask = np.zeros_like(mp_mask)
        roi_mask[y1:y2, x1:x2] = mp_mask[y1:y2, x1:x2]
        return roi_mask
    
    # MediaPipe 실패시 기존 방식 사용 (개선된 버전)
    print("⚠️  MediaPipe 마스크 생성 실패, 기존 방식 사용")
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    
    # 더 부드러운 블러링
    blur_size = max(31, int((x2-x1) * 0.1))  # 얼굴 크기에 비례
    if blur_size % 2 == 0:
        blur_size += 1
    
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), sigmaX=blur_size//3, sigmaY=blur_size//3)
    
    return mask[..., None]

# ─────────────────── 메인 (개선된 버전) ───────────────────
def main(use_style, gpu_idx):
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)
    
    # Clear GPU memory at start
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85, device=gpu_idx)
        print(f"📊 GPU {gpu_idx} 메모리 사용량: {torch.cuda.memory_allocated(gpu_idx)/1024**3:.1f}GB / {torch.cuda.max_memory_allocated(gpu_idx)/1024**3:.1f}GB")

    # ───── 얼굴 감지 & HED detector
    face_det = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=gpu_idx, det_size=(640, 640), det_thresh=0.3)
    
    # Load HED on CPU first
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    print(f"💾 HED detector loaded on CPU")

    # ───── 이미지 불러오기
    face_im = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im = to_sdxl_res(load_rgb(POSE_IMG))
    style_pil = load_rgb(STYLE_IMG)
    w_pose, h_pose = pose_im.size
    
    # 입력 이미지들 저장
    face_im.save(OUTDIR/"01_input_face.png")
    pose_im.save(OUTDIR/"02_input_pose.png")
    style_pil.save(OUTDIR/"03_input_style.png")
    print(f"💾 입력 이미지 저장: {OUTDIR}")

    print("🔄 랜드마크 기반 얼굴 정렬 사용")
    print(f"Face image size: {face_im.size}")
    print(f"Pose image size: {pose_im.size}")
    # ───── 랜드마크 기반 얼굴 정렬
    aligned_face, aligned_bbox = align_face_with_landmarks(face_im, pose_im, face_det)
    
    # 정렬된 얼굴 저장
    if aligned_face:
        aligned_face.save(OUTDIR/"04_aligned_face.png")
        print(f"💾 정렬된 얼굴 저장: {OUTDIR/'04_aligned_face.png'}")
    
    if aligned_bbox is None:
        # 랜드마크 정렬 실패시 기존 방법 사용
        print("📦 기존 bbox 방법 사용")
        pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
        p_info = max(face_det.get(pose_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
        x1, y1, x2, y2 = map(int, p_info['bbox'])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_pose, x2), min(h_pose, y2)
        
        # face HED 추출 
        face_cv = cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)
        f_info = max(face_det.get(face_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
        fx1, fy1, fx2, fy2 = map(int, f_info['bbox'])
        face_crop = face_im.crop((fx1, fy1, fx2, fy2))
        face_crop.save(OUTDIR/"05_face_crop_bbox.png")
        # Move HED to GPU temporarily
        hed = hed.to(DEVICE)
        face_hed = hed(face_crop, safe=False, scribble=False)
        # Move back to CPU to free memory
        hed = hed.to('cpu')
        torch.cuda.empty_cache()
        face_hed.save(OUTDIR/"06_face_hed_bbox.png")
        face_hed_resized = face_hed.resize((x2-x1, y2-y1), Image.LANCZOS)
        face_hed_resized.save(OUTDIR/"07_face_hed_resized_bbox.png")
        face_hed_np = np.array(face_hed_resized).astype(np.float32)
        print(f"💾 bbox 방식 HED 이미지 저장: {OUTDIR}")
        
    else:
        print("✨ 랜드마크 정렬 성공")
        x1, y1, x2, y2 = aligned_bbox
        
        # 정렬된 얼굴에서 HED 추출
        face_crop = aligned_face.crop((x1, y1, x2, y2))
        face_crop.save(OUTDIR/"05_face_crop_aligned.png")
        # Move HED to GPU temporarily
        hed = hed.to(DEVICE)
        face_hed = hed(face_crop, safe=False, scribble=False)
        # Move back to CPU to free memory
        hed = hed.to('cpu')
        torch.cuda.empty_cache()
        face_hed.save(OUTDIR/"06_face_hed_aligned.png")
        
        # 크기를 target region에 맞게 조정
        target_w, target_h = x2 - x1, y2 - y1
        face_hed_resized = face_hed.resize((target_w, target_h), Image.LANCZOS)
        face_hed_resized.save(OUTDIR/"07_face_hed_resized_aligned.png")
        face_hed_np = np.array(face_hed_resized).astype(np.float32)
        print(f"💾 랜드마크 방식 HED 이미지 저장: {OUTDIR}")

    # ───── pose 전체 HED
    # Move HED to GPU temporarily
    hed = hed.to(DEVICE)
    pose_hed_pil = hed(pose_im, safe=False, scribble=False).resize(pose_im.size, Image.LANCZOS)
    # Move back to CPU and clear memory
    hed = hed.to('cpu')
    del hed
    torch.cuda.empty_cache()
    pose_hed_pil.save(OUTDIR/"08_pose_hed.png")
    pose_hed_np = np.array(pose_hed_pil).astype(np.float32)
    print(f"💾 pose HED 저장: {OUTDIR/'08_pose_hed.png'}")

    # ───── 향상된 소프트 마스킹 (MediaPipe 마스크 저장)
    mask = create_enhanced_soft_mask(pose_im, (x1, y1, x2, y2), save_dir=OUTDIR)

    # face HED를 전체 canvas에 배치
    face_canvas_np = np.zeros_like(pose_hed_np).astype(np.float32)
    face_canvas_np[y1:y2, x1:x2] = face_hed_np
    face_canvas_pil = Image.fromarray(face_canvas_np.astype(np.uint8))
    face_canvas_pil.save(OUTDIR/"09_face_canvas.png")
    print(f"💾 face canvas 저장: {OUTDIR/'09_face_canvas.png'}")

    # 소프트 블렌딩
    pose_np = pose_hed_np.astype(np.float32)
    blended_np = mask * face_canvas_np + (1 - mask) * pose_np
    blended_np = blended_np.clip(0, 255).astype(np.uint8)
    merged_hed_pil = Image.fromarray(blended_np).convert("RGB")
    
    # 블렌딩 시각화를 위한 중간 이미지들 저장
    mask_vis = Image.fromarray((mask.squeeze() * 255).astype(np.uint8), mode='L')
    mask_vis.save(OUTDIR/"10_soft_mask.png")
    print(f"💾 소프트 마스크 저장: {OUTDIR/'10_soft_mask.png'}")
    
    # 결과 저장
    merged_hed_pil.save(OUTDIR/"11_merged_hed_enhanced.png")
    print(f"💾 HED 저장: {OUTDIR/'11_merged_hed_enhanced.png'}")

    # ───── ControlNet 구성
    controlnets, images, scales, masks = [], [], [], []
    controlnets.append(ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE))
    images.append(merged_hed_pil)
    scales.append(COND_HED)
    masks.append(None)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_SDXL,
        controlnet=MultiControlNetModel(controlnets),
        torch_dtype=DTYPE,
        add_watermarker=False
    ).to(DEVICE)

    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    
    # Enable CPU offloading for memory efficiency
    if not use_style:
        pipe.enable_sequential_cpu_offload()
    torch.cuda.empty_cache()

    # ───── 이미지 생성
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
        # Clear memory before loading IP-Adapter
        torch.cuda.empty_cache()
        
        # Temporarily move pipe to correct device for IP-Adapter
        pipe.to(DEVICE)
        
        ip = IPAdapterXL(
            pipe, STYLE_ENC, STYLE_IP, DEVICE,
            target_blocks=["up_blocks.0.attentions.1"]
        )
        out = ip.generate(pil_image=style_pil, scale=STYLE_SCALE,
                          seed=SEED, **gen_args)[0]
        # Clean up IP-Adapter
        del ip
        torch.cuda.empty_cache()
    else:
        out = pipe(**gen_args).images[0]

    # Final cleanup
    del pipe
    torch.cuda.empty_cache()
    
    fname = OUTDIR/"12_final_result.png"
    out.save(fname)
    print(f"✅ 최종 결과 저장: {fname}")
    print(f"🧼 GPU 메모리 정리 완료")

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", action="store_true", help="IP-Adapter 스타일 주입")
    ap.add_argument("--gpu", type=int, default=0, help="GPU 번호")
    args = ap.parse_args()
    main(args.style, args.gpu)