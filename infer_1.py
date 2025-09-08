"""
1. Baseline (InstantID → InstantStyle)

Stage 1) InstantID  : face image + pose image (kps + depth) + prompt
Stage 2) InstantStyle: Stage1 output as control image + style image + prompt
"""

import cv2
import torch
import numpy as np
from PIL import Image
import tempfile
import os

from diffusers.utils import load_image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.models import ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline, draw_kps
from controlnet_aux import MidasDetector
from ip_adapter import IPAdapterXL

from config import *   


# --- Model paths ---

base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

# InstantID
face_adapter = f'/data2/jiyoon/InstantID/checkpoints/ip-adapter.bin'
controlnet_path = f'/data2/jiyoon/InstantID/checkpoints/ControlNetModel'
controlnet_depth_path = f'/data2/jiyoon/instantstyle/checkpoints/controlnet-depth-sdxl-1.0-small'

# InstantStyle
controlnet_path = "/data2/jiyoon/instantstyle/checkpoints/controlnet-canny-sdxl-1.0"


# --- utils ---
def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image




# --- Stage1 : InstantID ---
def stage1_generate_with_face_and_pose(face_image_path, pose_image_path, prompt):
    """Stage 1: Generate image using face and pose (from infer_full.py)"""
    
    # Load face encoder
    app = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load depth detector
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

    # Load pipeline
    controlnet_list = [controlnet_path, controlnet_depth_path]
    controlnet_model_list = []
    for controlnet_path in controlnet_list:
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        controlnet_model_list.append(controlnet)
    controlnet = MultiControlNetModel(controlnet_model_list)


    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)

    # Load and process face image
    face_image = load_image(str(face_image_path))
    face_image = resize_img(face_image)
    
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    face_emb = face_info['embedding']

    # Load and process pose image
    pose_image = load_image(str(pose_image_path))
    pose_image = resize_img(pose_image)

    face_info = app.get(cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR))
    pose_image_cv2 = convert_from_image_to_cv2(pose_image)
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    face_kps = draw_kps(pose_image, face_info['kps'])

    width, height = face_kps.size

    # Use depth control
    processed_image_midas = midas(pose_image)
    processed_image_midas = processed_image_midas.resize(pose_image.size)
    
    # Enhance face region
    control_mask = np.zeros([height, width, 3])
    x1, y1, x2, y2 = face_info["bbox"]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    control_mask[y1:y2, x1:x2] = 255
    control_mask = Image.fromarray(control_mask.astype(np.uint8))

    # Generate image
    image = pipe(
        prompt=prompt,
        negative_prompt=config.NEG,
        image_embeds=face_emb,
        control_mask=control_mask,
        image=[face_kps, processed_image_midas],
        controlnet_conditioning_scale=[0.8, 0.8],
        control_guidance_start=[0.0, 0.0],
        control_guidance_end=[1.0, 1.0],
        ip_adapter_scale=0.8,
        num_inference_steps=30,
        guidance_scale=5,
        generator=torch.Generator().manual_seed(config.SEED),
    ).images[0]

    return image



# --- Stage2 : InstantStyle ---
def stage2_apply_style_transfer(input_image, style_image_path, prompt):
    """Stage 2: Apply style transfer using controlnet (from infer_style_controlnet.py)"""
    
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    image_encoder_path = config.STYLE_ENC
    ip_ckpt = config.STYLE_IP
    device = "cuda"

    controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)

    # Load SDXL pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()

    # Load ip-adapter
    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

    # Load style image
    style_image = Image.open(style_image_path)
    style_image = style_image.resize((512, 512))

    # Create canny edge map from input image
    input_image_cv2 = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    detected_map = cv2.Canny(input_image_cv2, 50, 200)
    canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

    # Generate final image
    images = ip_model.generate(
        pil_image=style_image,
        prompt=prompt,
        negative_prompt=config.NEG,
        scale=config.STYLE_SCALE,
        guidance_scale=config.CFG,
        num_samples=1,
        num_inference_steps=config.STEPS,
        seed=config.SEED,
        image=canny_map,
        controlnet_conditioning_scale=0.6,
    )

    return images[0]



# --- main ---
def main():
    """Main pipeline that chains the two stages"""
    
    # Stage 1
    print("Stage 1: Generating using InstantID...")
    stage1_result = stage1_generate_with_face_and_pose(
        face_image_path=config.FACE_IMG,
        pose_image_path=config.POSE_IMG, 
        prompt=config.PROMPT
    )
    
    # Save intermediate result
    intermediate_path = config.OUTDIR / "0_stage1_result.jpg"
    stage1_result.save(intermediate_path)
    print(f"Stage 1 complete. Intermediate result saved to: {intermediate_path}")
    

    # Stage 2
    print("Stage 2: Applying style transfer using InstantStyle...")
    final_result = stage2_apply_style_transfer(
        input_image=stage1_result,
        style_image_path=config.STYLE_IMG,
        prompt=config.PROMPT
    )
    
    # Save final result
    final_path = config.OUTDIR / "1_final_result.jpg"
    final_result.save(final_path)
    print(f"Pipeline complete! Final result saved to: {final_path}")

if __name__ == "__main__":
    main()
Revise the codes for me