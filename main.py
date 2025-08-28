"""
Align by warping
Face : HED + openpose kps 
Pose : openpose kps

Example Command
python main.py --gpu 5 --low-memory

conda env
instantstlye
"""
import argparse
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import HEDdetector, OpenposeDetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL

from utils import *


# (load_rgb, to_sdxl_res, expand_bbox, extract_pose_keypoints,
# align_face_with_landmarks, create_mediapipe_face_mask, create_enhanced_soft_mask,
# blend_face_hed_face_only, paste_face_into_pose)
from config import *

# --- Main ---
def main(gpu_idx, low_memory=False):

    # Set GPU
    global face_det
    DEVICE=f"cuda:{gpu_idx}"
    DTYPE=torch.float16
    torch.manual_seed(SEED)

    # Face detector
    face_det = FaceAnalysis(name="antelopev2", root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': gpu_idx}), 'CPUExecutionProvider'])
    face_det.prepare(ctx_id=gpu_idx, det_size=(640,640), det_thresh=0.1)

    # Load input imgs
    face_im = to_sdxl_res(load_rgb(FACE_IMG), low_mem=low_memory) 
    pose_im = to_sdxl_res(load_rgb(POSE_IMG), low_mem=low_memory)
    style_pil = load_rgb(STYLE_IMG)
    face_im.save(OUTDIR/"0_face_input.png")
    pose_im.save(OUTDIR/"1_pose_input.png")
    style_pil.save(OUTDIR/"2_style_input.png")

    # Align face 
    aligned_face, bbox = align_face_with_landmarks(face_im, pose_im, face_det)
    if bbox is None:
        pose_cv = cv2.cvtColor(np.array(pose_im), cv2.COLOR_RGB2BGR)
        p_info = max(face_det.get(pose_cv), key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
        bbox = expand_bbox(p_info['bbox'], 1.35, pose_im.width, pose_im.height)
        face_crop = face_im
    else:
        x1,y1,x2,y2 = bbox
        face_crop = aligned_face.crop((x1,y1,x2,y2))

    # Face HED from aligned face crop
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    face_hed = hed(face_crop, safe=False, scribble=False)
    face_hed.save(OUTDIR/"4_face_hed_raw.png")
    del hed; torch.cuda.empty_cache()

    # Mediapipe face mask
    mask = create_enhanced_soft_mask(pose_im, bbox)

    # HED face condition on pose canvas
    hed_face_only = blend_face_hed_face_only(face_hed, pose_im, mask, bbox)

    # Build composite canvas 
    composite = paste_face_into_pose(pose_im, aligned_face, mask, bbox)

    # Extract kps from composite canvas
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    pose_kps = extract_pose_keypoints(
        composite, openpose,
        include_body=True, include_hand=True, include_face=True,  
        save_name="7_kps.png"
    )
    del openpose; torch.cuda.empty_cache()

    # ControlNet setup
    controlnets = [
        ControlNetModel.from_pretrained(CN_POSE, torch_dtype=DTYPE),
        ControlNetModel.from_pretrained(CN_HED,  torch_dtype=DTYPE),
    ]
    images = [pose_kps, hed_face_only]
    scales = [COND_POSE, COND_HED]

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_SDXL,
        controlnet=MultiControlNetModel(controlnets),
        torch_dtype=DTYPE,
        add_watermarker=False
    ).to(DEVICE)

    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing(1)
    pipe.enable_model_cpu_offload()

    args = dict(
        prompt=PROMPT,
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        image=images,
        controlnet_conditioning_scale=scales,
        generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    )

    # IP-Adapter
    ip = IPAdapterXL(
        pipe, STYLE_ENC, STYLE_IP, DEVICE,
        target_blocks=["up_blocks.0.attentions.1"]
    )
    pipe_args = {k: v for k, v in args.items() if k != "generator"}
    out = ip.generate(
        pil_image=style_pil,
        scale=STYLE_SCALE,
        seed=SEED,                 
        **pipe_args
    )[0]
    del ip
    
    out.save(OUTDIR/"8_final_result.png")
    print(f"✅ Saved all intermediates in {OUTDIR}")

# ─────────────────── CLI ───────────────────
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--low-memory", action="store_true")
    args = ap.parse_args()
    main(args.gpu, args.low_memory)