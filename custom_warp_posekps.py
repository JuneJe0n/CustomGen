"""
Aligned face HED (face-only) + body pose keypoints (no face/hands)
Fixed: use pose face bbox (expanded) for cropping/resizing/pasting
Added: save all intermediate images to OUTDIR
"""
import argparse, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import HEDdetector, OpenposeDetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL
from skimage.transform import SimilarityTransform, warp
import mediapipe as mp

# ─────────────────── Settings ───────────────────
PROMPT = "a baby sitting, clear facial features, detailed, realistic, smooth colors"
NEG = "(lowres, bad quality, watermark, disjointed, strange limbs, cut off, bad anatomymissing limbs, fused fingers)"
FACE_IMG  = Path("/data2/jeesoo/FFHQ/00000/00000.png")
POSE_IMG  = Path("/data2/jiyoon/custom/data/pose/p5.jpg")
STYLE_IMG = Path("/data2/jiyoon/custom/data/style/s3.png")

CN_HED     = "/data2/jiyoon/custom/ckpts/controlnet-union-sdxl-1.0"       
CN_POSE    = "/data2/jiyoon/custom/ckpts/controlnet-openpose-sdxl-1.0"    
BASE_SDXL  = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE_ENC  = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP   = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

COND_HED     = 0.8
COND_POSE    = 0.6
STYLE_SCALE  = 0.8
CFG, STEPS   = 7.0, 50
SEED         = 42
OUTDIR       = Path("/data2/jiyoon/custom/results/pose_kps/00000/p5")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Globals
face_det = None
hed = None

# ─────────────────── Utils ───────────────────
def load_rgb(p): 
    return Image.open(p).convert("RGB")

def to_sdxl_res(img, base=64, short=1024, long=1024, low_mem=False):
    if low_mem:
        short, long = 768, 768
    w, h = img.size
    r = short / min(w, h); w, h = int(w*r), int(h*r)
    r = long  / max(w, h); w, h = int(w*r), int(h*r)
    return img.resize(((w//base)*base, (h//base)*base), Image.LANCZOS)

def expand_bbox(bbox, scale, W, H):
    x1, y1, x2, y2 = map(int, bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    nx1, ny1 = int(max(0, cx - w/2)), int(max(0, cy - h/2))
    nx2, ny2 = int(min(W, cx + w/2)), int(min(H, cy + h/2))
    return [nx1, ny1, nx2, ny2]

def extract_pose_keypoints(pose_img, pose_detector):
    pose_kps = pose_detector(pose_img, include_body=True, include_hand=False, include_face=False)
    pose_kps = pose_kps.resize(pose_img.size, Image.LANCZOS)
    pose_kps.save(OUTDIR / "01_pose_keypoints.png")
    print("💾 Saved: 01_pose_keypoints.png")
    return pose_kps

def align_face_with_landmarks(face_img, pose_img, face_det):
    face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    pose_cv = cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR)
    face_infos = face_det.get(face_cv)
    pose_infos = face_det.get(pose_cv)
    if not face_infos or not pose_infos:
        return face_img, None
    face_info = max(face_infos, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    pose_info = max(pose_infos, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
    face_kps, pose_kps = face_info['kps'], pose_info['kps']
    try:
        tform = SimilarityTransform()
        tform.estimate(face_kps, pose_kps)
        h, w = pose_img.size[::-1]
        aligned_face = warp(np.array(face_img), tform.inverse, output_shape=(h, w), preserve_range=True)
        aligned_face_pil = Image.fromarray(aligned_face.astype(np.uint8))
        aligned_face_pil.save(OUTDIR/"02_aligned_face.png")
        pose_bbox_expanded = expand_bbox(pose_info['bbox'], scale=1.35, W=w, H=h)
        return aligned_face_pil, pose_bbox_expanded
    except:
        return face_img, None

def create_mediapipe_face_mask(img):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = fm.process(img_cv)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            face_oval = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            mask = np.zeros(img_cv.shape[:2], dtype=np.float32)
            pts = [(int(lm.landmark[i].x*img_cv.shape[1]), int(lm.landmark[i].y*img_cv.shape[0])) for i in face_oval]
            cv2.fillPoly(mask, [np.array(pts)], 1.0)
            mask = cv2.GaussianBlur(mask, (21,21), 7)
            mask_img = Image.fromarray((mask*255).astype(np.uint8))
            mask_img.save(OUTDIR/"03_mediapipe_mask.png")
            return mask[...,None]
    return None

def create_enhanced_soft_mask(pose_img, bbox):
    h, w = pose_img.size[::-1]
    x1,y1,x2,y2 = bbox
    mp_mask = create_mediapipe_face_mask(pose_img)
    if mp_mask is not None:
        roi = np.zeros_like(mp_mask)
        roi[y1:y2,x1:x2] = mp_mask[y1:y2,x1:x2]
        return roi
    mask = np.zeros((h,w),dtype=np.float32)
    mask[y1:y2,x1:x2]=1.0
    mask = cv2.GaussianBlur(mask,(51,51),15)
    Image.fromarray((mask*255).astype(np.uint8)).save(OUTDIR/"03b_bbox_mask.png")
    return mask[...,None]

def blend_face_hed_face_only(face_hed, pose_img, face_mask, bbox):
    x1,y1,x2,y2 = bbox
    tw,th = x2-x1,y2-y1
    face_hed_resized = face_hed.resize((tw,th), Image.LANCZOS)
    face_hed_resized.save(OUTDIR/"04_face_hed_resized.png")
    hed_np = np.array(face_hed_resized)
    if hed_np.ndim==2: hed_np = np.stack([hed_np]*3,axis=2)
    H,W = pose_img.height, pose_img.width
    canvas = np.zeros((H,W,3),dtype=np.float32)
    canvas[y1:y2,x1:x2]=hed_np[:th,:tw]
    if face_mask is not None:
        if face_mask.ndim==3 and face_mask.shape[2]==1:
            face_mask=np.repeat(face_mask,3,axis=2)
        canvas *= face_mask
    result = Image.fromarray(np.clip(canvas,0,255).astype(np.uint8)).convert("RGB")
    result.save(OUTDIR/"05_hed_face_only.png")
    return result

# ─────────────────── Main ───────────────────
def main(use_style, gpu_idx, low_memory=False):
    global face_det, hed
    DEVICE=f"cuda:{gpu_idx}"
    DTYPE=torch.float16
    torch.manual_seed(SEED)

    face_det = FaceAnalysis(name="antelopev2", root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider'])
    face_det.prepare(ctx_id=gpu_idx, det_size=(640,640), det_thresh=0.3)
    
    face_im=to_sdxl_res(load_rgb(FACE_IMG), low_mem=low_memory)
    pose_im=to_sdxl_res(load_rgb(POSE_IMG), low_mem=low_memory)
    style_pil=load_rgb(STYLE_IMG)
    face_im.save(OUTDIR/"00_face_input.png")
    pose_im.save(OUTDIR/"00_pose_input.png")
    style_pil.save(OUTDIR/"00_style_input.png")

    openpose=OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    pose_kps=extract_pose_keypoints(pose_im,openpose)
    del openpose; torch.cuda.empty_cache()

    aligned_face,bbox=align_face_with_landmarks(face_im,pose_im,face_det)
    if bbox is None:
        pose_cv=cv2.cvtColor(np.array(pose_im),cv2.COLOR_RGB2BGR)
        p_info=max(face_det.get(pose_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
        bbox=expand_bbox(p_info['bbox'],1.35,pose_im.width,pose_im.height)
        face_crop=face_im
    else:
        x1,y1,x2,y2=bbox
        face_crop=aligned_face.crop((x1,y1,x2,y2))
    face_crop.save(OUTDIR/"02b_face_crop.png")

    hed=HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    face_hed=hed(face_crop,safe=False,scribble=False)
    face_hed.save(OUTDIR/"04_face_hed_raw.png")
    del hed; torch.cuda.empty_cache()

    mask=create_enhanced_soft_mask(pose_im,bbox)
    hed_face_only=blend_face_hed_face_only(face_hed,pose_im,mask,bbox)

    controlnets=[
        ControlNetModel.from_pretrained(CN_POSE,torch_dtype=DTYPE),
        ControlNetModel.from_pretrained(CN_HED,torch_dtype=DTYPE)]
    images=[pose_kps,hed_face_only]
    scales=[COND_POSE,COND_HED]

    pipe=StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_SDXL, controlnet=MultiControlNetModel(controlnets),
        torch_dtype=DTYPE, add_watermarker=False).to(DEVICE)
    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing(1)
    pipe.enable_model_cpu_offload()

    args=dict(prompt=PROMPT,negative_prompt=NEG,
        num_inference_steps=STEPS,guidance_scale=CFG,
        image=images,controlnet_conditioning_scale=scales)

    if use_style:
        ip=IPAdapterXL(pipe,STYLE_ENC,STYLE_IP,DEVICE,
            target_blocks=["up_blocks.0.attentions.1"])
        out=ip.generate(pil_image=style_pil,scale=STYLE_SCALE,
            seed=SEED,**args)[0]
        del ip
    else:
        out=pipe(**args).images[0]

    out.save(OUTDIR/"06_final_result.png")
    print(f"✅ Saved all intermediates in {OUTDIR}")

# ─────────────────── CLI ───────────────────
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--style",action="store_true")
    ap.add_argument("--gpu",type=int,default=0)
    ap.add_argument("--low-memory",action="store_true")
    args=ap.parse_args()
    main(args.style,args.gpu,args.low_memory)
