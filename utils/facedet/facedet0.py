"""
Face extraction script using InsightFace AntelopeV2
Extracts faces from all images in a folder and saves them with the same filename

Usage: python facedet.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from insightface.app import FaceAnalysis
from tqdm import tqdm

def main():
    # Configuration
    input_folder = Path("/data2/jiyoon/custom/results/final/infer_8") 
    output_folder = Path("/data2/jiyoon/custom/results/final/metric/CLIP_ID_facedet0/generated_face/infer_8")  
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize face detector
    gpu_idx = 0
    device_id = 0 if 'CUDA_VISIBLE_DEVICES' in os.environ else gpu_idx
    face_det = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
    )
    face_det.prepare(ctx_id=device_id, det_size=(640, 640))
    
    # Supported image extensions
    img_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    
    # Get all image files from input folder
    image_files = []
    for ext in img_extensions:
        image_files.extend(input_folder.glob(f"*{ext}"))
        image_files.extend(input_folder.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to process")
    
    success_count = 0
    failed_count = 0
    
    for img_path in tqdm(image_files, desc="Extracting faces"):
        try:
            # Load image
            face_im = Image.open(img_path).convert("RGB")
            
            # Face bbox from face img
            face_cv_full = cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)
            f_faces = face_det.get(face_cv_full)
            if not f_faces:
                print(f"No face detected in {img_path.name}")
                failed_count += 1
                continue
                
            # Get the largest face
            f_info = max(f_faces, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
            fx1, fy1, fx2, fy2 = map(int, f_info['bbox'])
            fw, fh = fx2 - fx1, fy2 - fy1
            face_crop_pil = face_im.crop((fx1, fy1, fx2, fy2))
            
            # Save extracted face with same filename
            output_path = output_folder / img_path.name
            face_crop_pil.save(output_path)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            failed_count += 1
    
    print(f"\nFace extraction completed!")
    print(f"Successfully extracted: {success_count} faces")
    print(f"Failed: {failed_count} images")
    print(f"Output saved to: {output_folder}")

if __name__ == "__main__":
    main()