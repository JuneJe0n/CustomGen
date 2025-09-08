"""
Modified Pipeline:
1. Generate prompt from pose image only
2. Stage 1) InstantID: face image + pose image + generated prompt
3. Stage 2) InstantStyle: Stage1 output + style image + generated prompt
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from diffusers.utils import load_image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline, draw_kps
from controlnet_aux import MidasDetector
from ip_adapter import IPAdapterXL

from config import *

# --- Model Configuration ---
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# InstantID paths
INSTANTID_IP_ADAPTER = "/data2/jiyoon/InstantID/checkpoints/ip-adapter.bin"
INSTANTID_CONTROLNET_FACEKPS = "/data2/jiyoon/InstantID/checkpoints/ControlNetModel"
INSTANTID_CONTROLNET_DEPTH = "/data2/jiyoon/instantstyle/checkpoints/controlnet-depth-sdxl-1.0-small"

# InstantStyle paths
INSTANTSTYLE_CONTROLNET_CANNY = "/data2/jiyoon/instantstyle/checkpoints/controlnet-canny-sdxl-1.0"

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32


# --- Step 1: Generate Prompt from Pose Image Only ---
def generate_prompt_from_pose(pose_image_path):
    """Generate prompt using only the pose image with POSE_PROMPT"""
    print("Generating prompt from pose image...")

    generator = PromptGenerator()
    
    # Use POSE_PROMPT to analyze pose image
    pose_prompt = generator.analyze_image(pose_image_path, POSE_PROMPT)
    
    print(f"Generated pose prompt: {pose_prompt}")
    return pose_prompt


# --- Utility Functions ---
def convert_from_image_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (RGB -> BGR)"""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    """Resize image while maintaining aspect ratio (from infer_full.py)"""
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        w2, h2 = round(ratio * w), round(ratio * h)
        w_resize_new = (w2 // base_pixel_number) * base_pixel_number
        h_resize_new = (h2 // base_pixel_number) * base_pixel_number

    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


# --- Stage 1: InstantID Generation (from infer_full.py) ---
@torch.inference_mode()
def stage1_instantid_generation(face_image_path, pose_image_path, prompt):
    """
    Stage 1: Generate image using InstantID with face identity and pose control
    Based on infer_full.py logic
    """
    print("="*50)
    print("STAGE 1: InstantID Generation")
    print("="*50)
    print("Initializing InstantID pipeline...")
    
    app = FaceAnalysis(name="antelopev2", root="/data2/jiyoon/InstantID", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

    controlnet_list = [INSTANTID_CONTROLNET_FACEKPS, INSTANTID_CONTROLNET_DEPTH]
    controlnet_model_list = []
    for controlnet_path in controlnet_list:
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        controlnet_model_list.append(controlnet)
    multi_controlnet = MultiControlNetModel(controlnet_model_list)

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=multi_controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda() 
    pipe.load_ip_adapter_instantid(INSTANTID_IP_ADAPTER)

    print("Processing face image...")
    face_image = load_image(str(face_image_path))
    face_image = resize_img(face_image)
    
    faces = app.get(convert_from_image_to_cv2(face_image))
    if not faces:
        raise RuntimeError(f"No face found in face image: {face_image_path}")
    
    face_info = max(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
    face_emb = face_info["embedding"]
    print("Processing pose image...")
    pose_image = load_image(str(pose_image_path))
    pose_image = resize_img(pose_image)
    
    pose_faces = app.get(convert_from_image_to_cv2(pose_image))
    if not pose_faces:
        raise RuntimeError(f"No face found in pose image: {pose_image_path}")
    
    pose_face_info = max(pose_faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
    face_kps = draw_kps(pose_image, pose_face_info["kps"])

    processed_depth = midas(pose_image).resize(pose_image.size)

    w, h = face_kps.size
    x1, y1, x2, y2 = map(int, pose_face_info["bbox"])
    control_mask = np.zeros((h, w, 3), dtype=np.uint8)
    control_mask[y1:y2, x1:x2] = 255
    control_mask = Image.fromarray(control_mask)

    print(f"Face embedding shape: {face_emb.shape}")
    print(f"Control images size: {face_kps.size}")
    print(f"Prompt: '{prompt}'")

    print("Generating image with InstantID...")
    
    result = pipe(
        prompt=prompt,
        negative_prompt=NEG,
        image_embeds=face_emb,
        control_mask=control_mask,
        image=[face_kps, processed_depth],
        controlnet_conditioning_scale=[0.8, 0.8],
        control_guidance_start=[0.0, 0.0],
        control_guidance_end=[1.0, 1.0],
        ip_adapter_scale=0.8,
        num_inference_steps=30,
        guidance_scale=5,  
    )

    generated_image = result.images[0]
    
    # Validate output
    img_array = np.array(generated_image)
    min_val, max_val, mean_val = img_array.min(), img_array.max(), img_array.mean()
    print(f"Generated image stats: min={min_val}, max={max_val}, mean={mean_val:.2f}")
    
    if max_val == 0:
        print("⚠️  WARNING: Generated image appears to be completely black!")
        print("Retrying with reduced controlnet strength...")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.startswith("cuda")):
            result = pipe(
                prompt=prompt,
                negative_prompt=NEG,
                image_embeds=face_emb,
                image=[face_kps, processed_depth],  # Remove control_mask
                controlnet_conditioning_scale=[0.5, 0.5],  # Reduced strength
                control_guidance_start=[0.0, 0.0],
                control_guidance_end=[1.0, 1.0],
                ip_adapter_scale=0.6,  # Reduced IP adapter strength
                num_inference_steps=30,
                guidance_scale=7.0,  # Increased guidance
                generator=generator,
            )
        generated_image = result.images[0]
        
        # Check again
        img_array = np.array(generated_image)
        min_val, max_val, mean_val = img_array.min(), img_array.max(), img_array.mean()
        print(f"Retry image stats: min={min_val}, max={max_val}, mean={mean_val:.2f}")
        
        if max_val == 0:
            print("❌ Still generating black images after retry. This may be a model compatibility issue.")

    return generated_image


# --- Stage 2: InstantStyle Transfer (from infer_style_controlnet.py) ---
@torch.inference_mode()
def stage2_instantstyle_transfer(input_image, style_image_path, prompt):
    """
    Stage 2: Apply style transfer using InstantStyle with ControlNet
    Based on infer_style_controlnet.py logic
    """
    print("="*50)
    print("STAGE 2: InstantStyle Transfer")
    print("="*50)
    print("Initializing InstantStyle pipeline...")
    
    controlnet = ControlNetModel.from_pretrained(
        INSTANTSTYLE_CONTROLNET_CANNY, 
        use_safetensors=False, 
        torch_dtype=DTYPE
    ).to(DEVICE)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        add_watermarker=False,
    )
    pipe.to(DEVICE)
    
    if DEVICE.startswith("cuda"):
        pipe.enable_vae_tiling()
        pipe.enable_xformers_memory_efficient_attention()

    ip_model = IPAdapterXL(
        pipe,
        image_encoder_path=STYLE_ENC,
        ip_ckpt=STYLE_IP,
        device=DEVICE,
        target_blocks=["up_blocks.0.attentions.1"],
    )

    print("Processing style image...")
    style_image = Image.open(style_image_path).convert("RGB").resize((512, 512), Image.BILINEAR)

    print("Creating canny edge map...")
    input_cv2 = convert_from_image_to_cv2(input_image)
    edges = cv2.Canny(input_cv2, threshold1=50, threshold2=200)
    canny_map = Image.fromarray(edges)  

    print(f"Input image size: {input_image.size}")
    print(f"Style image size: {style_image.size}")
    print(f"Canny map size: {canny_map.size}")

    print("Generating styled image...")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.startswith("cuda")):
        images = ip_model.generate(
            pil_image=style_image,
            prompt=prompt,
            negative_prompt=NEG,
            scale=STYLE_SCALE,
            guidance_scale=CFG,
            num_samples=1,
            num_inference_steps=STEPS,
            seed=SEED,
            image=canny_map,
            controlnet_conditioning_scale=0.6,
        )

    result_image = images[0]
    
    result_array = np.array(result_image)
    min_val, max_val, mean_val = result_array.min(), result_array.max(), result_array.mean()
    print(f"Final image stats: min={min_val}, max={max_val}, mean={mean_val:.2f}")

    return result_image


# --- Main Pipeline ---
def main():
    """Execute the complete pipeline"""
    print("🚀 Starting Modified InstantID → InstantStyle Pipeline")
    print(f"Using device: {DEVICE}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    try:
        stage1_result = stage1_instantid_generation(
            face_image_path=FACE_IMG,
            pose_image_path=POSE_IMG,
            prompt=PROMPT,  
        )
        
        intermediate_path = OUTDIR / "0_stage1_instantid_result.jpg"
        stage1_result.save(intermediate_path)
        print(f"✅ Stage 1 complete! Result saved to: {intermediate_path}")


        final_result = stage2_instantstyle_transfer(
            input_image=stage1_result,
            style_image_path=STYLE_IMG,
            prompt=PROMPT, 
        )
        
        final_path = OUTDIR / "1_final_styled_result.jpg"
        final_result.save(final_path)
        print(f"✅ Stage 2 complete! Final result saved to: {final_path}")
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Prompt used for both stages: '{PROMPT}'")
        print(f"Intermediate result: {intermediate_path}")
        print(f"Final result: {final_path}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()