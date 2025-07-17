#!/usr/bin/env python
# custom_canny_face.py  (2025-07-16 rev-D)

"""
예시
CUDA_VISIBLE_DEVICES=1,2 python custom_canny_face.py --ctrl edge
CUDA_VISIBLE_DEVICES=1,2 python custom_canny_face.py --ctrl both --style --gpu 1
"""
import argparse, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import MidasDetector, HEDdetector # HEDdetector 추가
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL

# ─────────────────── 기본 설정 ───────────────────
PROMPT, NEG = "a baby with clear facial features", "(lowres, bad quality, watermark)"
FACE_IMG  = Path("/data2/jiyoon/custom/data/face/00000.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/pose/p2.jpeg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/style/s3.png")

# CN_EDGE  = "diffusers/controlnet-canny-sdxl-1.0"
CN_HED = "/data2/jiyoon/custom/ckpts/controlnet-union-sdxl-1.0"
CN_DEPTH = "diffusers/controlnet-depth-sdxl-1.0-small"
BASE_SDXL= "stabilityai/stable-diffusion-xl-base-1.0"

STYLE_ENC = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP  = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"


# COND_EDGE, COND_DEPTH = 0.8, 0.6
COND_HED, COND_DEPTH = 0.8, 0.6
STYLE_SCALE, CFG      = 0.8, 7.0
STEPS, SEED           = 50, 42

OUTDIR = Path("/data2/jiyoon/custom/results/mode/8/HED")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────── 유틸 ───────────────────
def load_rgb(p): return Image.open(p).convert("RGB")

def to_sdxl_res(img, base=64, short=1024, long=1024):
    w, h = img.size
    r = short / min(w, h); w, h = int(w*r), int(h*r)
    r = long  / max(w, h); w, h = int(w*r), int(h*r)
    return img.resize(((w//base)*base, (h//base)*base), Image.BILINEAR)

def depth_to_rgb(arr):
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    if torch.is_tensor(arr):
        arr = arr.squeeze().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    g   = (arr*255).astype("uint8")
    return Image.fromarray(np.repeat(g[..., None], 3, -1))

# ─────────────────── 메인 ───────────────────
def main(ctrl, use_style, gpu_idx):
    DEVICE = f"cuda:{gpu_idx}"
    DTYPE  = torch.float16
    torch.manual_seed(SEED)

    # detector & depth & HED
    face_det = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=gpu_idx, det_size=(640, 640))
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE) # HED detector 초기화

    # 0. 이미지 로딩
    face_im   = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im   = to_sdxl_res(load_rgb(POSE_IMG))
    style_pil = load_rgb(STYLE_IMG)

    w_pose, h_pose = pose_im.size

    # 1-A. pose 얼굴 bbox & mask
    pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
    p_info  = max(face_det.get(pose_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    x1, y1, x2, y2 = map(int, p_info['bbox'])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_pose, x2), min(h_pose, y2)

    mask_canvas = np.zeros((h_pose, w_pose, 3), dtype=np.uint8)
    mask_canvas[y1:y2, x1:x2] = 255
    mask_pil = Image.fromarray(mask_canvas)

    # 1-B. face bbox → HED edge 추출
    face_cv = cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)
    f_info  = max(face_det.get(face_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    fx1, fy1, fx2, fy2 = map(int, f_info['bbox'])
    face_crop_pil = face_im.crop((fx1, fy1, fx2, fy2)) # PIL Image로 크롭
    f_edges_pil = hed(face_crop_pil, safe=False, scribble=False) # HED 적용

    # pose 얼굴 bbox 사이즈로 리사이즈
    pw, ph = x2 - x1, y2 - y1
    edge_resized_pil = f_edges_pil.resize((pw, ph), Image.BILINEAR)

    edge_canvas_np = np.zeros_like(mask_canvas)
    edge_canvas_np[y1:y2, x1:x2] = np.array(edge_resized_pil)
    edge_pil = Image.fromarray(edge_canvas_np)
    edge_pil.save(OUTDIR/"hed_edge.png") # 파일명 변경
    mask_pil.save(OUTDIR/"mask.png")

    # 2. pose depth
    depth_pil = depth_to_rgb(midas(pose_im)).resize(pose_im.size, Image.BILINEAR)
    depth_pil.save(OUTDIR/"depth.png")

    # 3. ControlNet 설정
    controlnets, images, scales, masks = [], [], [], []
    if ctrl in ("edge", "both"):
        controlnets.append(ControlNetModel.from_pretrained(CN_HED, torch_dtype=DTYPE))
        images.append(edge_pil);  scales.append(COND_HED);  masks.append(mask_pil)
    if ctrl in ("depth", "both"):
        controlnets.append(ControlNetModel.from_pretrained(CN_DEPTH, torch_dtype=DTYPE))
        images.append(depth_pil); scales.append(COND_DEPTH); masks.append(None)

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

    # 4. 생성
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
        ip = IPAdapterXL(
            pipe, STYLE_ENC, STYLE_IP, DEVICE,
            target_blocks=["up_blocks.0.attentions.1"]
        )
        out = ip.generate(pil_image=style_pil, scale=STYLE_SCALE,
                          seed=SEED, **gen_args)[0]
    else:
        out = pipe(**gen_args).images[0]

    fname = OUTDIR/"2_hedmodel.png" # 출력 파일명도 HED로 변경
    out.save(fname); print("✅ saved →", fname)

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl",  choices=["edge", "depth", "both"], required=True,
                    help="'edge' = 얼굴 HED, 'depth' = depth, 'both' = 둘 다") # 도움말도 HED로 변경
    ap.add_argument("--style", action="store_true", help="IP-Adapter 스타일 주입")
    ap.add_argument("--gpu",   type=int, default=0,
                    help="CUDA_VISIBLE_DEVICES 안 논리 GPU 번호 (default=0)")
    args = ap.parse_args()
    main(args.ctrl, args.style, args.gpu)