
"""
SSIM score computation:
- <target-folder>  vs  FFHQ/{bucket}/{id5}.{ext}

Usage: python SSIM.py
(Modify the hardcoded values in main() function as needed)
"""

from pathlib import Path
import re
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim
import cv2

ID_ALPHA_RE = re.compile(r"^(\d{5})_([A-Za-z])")
ALT_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
ANCHOR_EXTS = [".png", ".jpg", ".jpeg"] 

def resolve_ffhq_anchor(ffhq_root: Path, id5: str, bucket_size: int, recursive_fallback: bool = False) -> Optional[Path]:
    """
    Find FFHQ anchor image.
    Priority:
      1) Bucket layout ({id5//bucket_size:05d}/{id5}.{ext})
      2) Flat layout ({ffhq_root}/{id5}.{ext})
      3) (Optional) Recursive rglb search
    """
    idx = int(id5)

    # 1) Specified bucket layout
    if bucket_size and bucket_size > 0:
        folder = f"{idx // bucket_size:05d}"
        for ext in ANCHOR_EXTS:
            p = ffhq_root / folder / f"{id5}{ext}"
            if p.exists():
                return p

    # 2) Flat layout
    for ext in ANCHOR_EXTS:
        p = ffhq_root / f"{id5}{ext}"
        if p.exists():
            return p

    # 3) (Optional) Recursive search - disabled by default as it can be slow
    if recursive_fallback:
        for ext in ANCHOR_EXTS:
            hits = list(ffhq_root.rglob(f"{id5}{ext}"))
            if hits:
                return hits[0]

    return None

def resolve_target_image(folder: Path, target_name: Optional[str]) -> Optional[Path]:
    if target_name:
        base = target_name.strip()
        # With extension
        if Path(base).suffix:
            p = folder / base
            if p.exists():
                return p
        else:
            # Without extension
            for ext in ALT_EXTS:
                p = folder / f"{base}{ext}"
                if p.exists():
                    return p
        # Prefix matching
        cands = sorted([q for q in folder.glob(f"{base}*") if q.suffix.lower() in ALT_EXTS])
        if cands:
            return cands[0]
        return None

    # Auto selection
    imgs = sorted([q for q in folder.iterdir() if q.is_file() and q.suffix.lower() in ALT_EXTS])
    return imgs[0] if imgs else None

class SSIMScorer:
    def __init__(self, target_size: tuple = (256, 256)):
        self.target_size = target_size

    def calculate_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate SSIM score between two images
        """
        # Resize images to same size
        img1_resized = img1.resize(self.target_size, Image.LANCZOS)
        img2_resized = img2.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy arrays
        img1_array = np.array(img1_resized)
        img2_array = np.array(img2_resized)
        
        # Convert to grayscale if images are RGB
        if len(img1_array.shape) == 3:
            img1_gray = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2_array, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1_array
            img2_gray = img2_array
        
        # Calculate SSIM
        ssim_score = ssim(img1_gray, img2_gray, data_range=255)
        return float(ssim_score)

def main():
    # Hardcoded values - modify these as needed
    target_folder = "/data2/jiyoon/custom/results/final/metric/CLIP_ID_facedet1/generated_face/infer_7"
    out_file = "/data2/jiyoon/custom/results/final/metric/SSIM/SSIM_infer7.csv"
    ffhq_root_path = "/data2/jiyoon/custom/results/final/metric/CLIP_ID_facedet0/ffhq_face"
    bucket_size = 1000
    target_name = ""
    target_size = (256, 256)  # Size to resize images for SSIM calculation

    root_dir = Path(target_folder)
    ffhq_root = Path(ffhq_root_path)
    assert root_dir.is_dir(), f"target-folder path is not a directory: {root_dir}"
    assert ffhq_root.is_dir(), f"FFHQ root does not exist: {ffhq_root}"

    ssim_scorer = SSIMScorer(target_size)
    print(f"[INFO] model=SSIM, target_size={target_size}")
    print(f"[INFO] target_folder={root_dir} | ffhq_root={ffhq_root} | bucket_size={bucket_size} | target_name={target_name or '(auto)'}")

    rows: List[Dict] = []
    
    # Check if we have subdirectories or individual files
    all_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    all_files = sorted([p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() in ALT_EXTS])
    
    if all_dirs:
        # Original behavior: process subdirectories
        print(f"[INFO] total_subdirs={len(all_dirs)}")
        items_to_process = all_dirs
        process_mode = "subdirs"
    else:
        # New behavior: process individual files
        print(f"[INFO] total_files={len(all_files)}")
        items_to_process = all_files
        process_mode = "files"

    # Store SSIM scores for statistics
    ssim_scores = []

    for item in tqdm(items_to_process, desc="Evaluating", ncols=100):
        if process_mode == "subdirs":
            # Original logic for subdirectories
            m = ID_ALPHA_RE.match(item.name)
            if not m:
                continue
            id5 = m.group(1)
            
            row = {
                "folder": item.name,
                "id5": id5,
                "anchor_path": "",
                "target_path": "",
                "ssim_score": np.nan,
                "notes": ""
            }
            
            # Target image
            target_path = resolve_target_image(item, target_name or None)
            if not target_path:
                row["notes"] = "target_missing"; rows.append(row); continue
            row["target_path"] = str(target_path)
        else:
            # New logic for individual files
            m = ID_ALPHA_RE.match(item.name)
            if not m:
                continue
            id5 = m.group(1)
            
            row = {
                "folder": item.name,
                "id5": id5,
                "anchor_path": "",
                "target_path": str(item),
                "ssim_score": np.nan,
                "notes": ""
            }
            target_path = item

        # Anchor image
        anchor_path = resolve_ffhq_anchor(ffhq_root, id5, bucket_size, recursive_fallback=False)
        if not anchor_path:
            row["notes"] = "anchor_missing"; rows.append(row); continue
        row["anchor_path"] = str(anchor_path)

        # Calculate SSIM score between anchor and target
        try:
            # Load images
            ref_img = Image.open(anchor_path).convert("RGB")
            gen_img = Image.open(target_path).convert("RGB")
            
            # Calculate SSIM score
            ssim_score = ssim_scorer.calculate_ssim(ref_img, gen_img)
            row["ssim_score"] = ssim_score
            ssim_scores.append(ssim_score)
            
        except Exception as e:
            row["notes"] = "image_open_or_ssim_error"
        
        rows.append(row)

    # Save CSV
    df = pd.DataFrame(rows)
    out = Path(out_file)
    df.to_csv(out, index=False)
    print(f"\n[DONE] saved: {out.resolve()}")

    # Statistics/Aggregation
    if not df.empty:
        print("[DEBUG] notes counts:", dict(Counter(df["notes"].fillna(""))))
        if ssim_scores:
            ssim_array = np.array(ssim_scores)
            print(f"\n[SSIM STATS] count={len(ssim_scores)}, mean={ssim_array.mean():.6f}, min={ssim_array.min():.6f}, max={ssim_array.max():.6f}, std={ssim_array.std():.6f}")
        else:
            print("[STAT] No valid SSIM scores calculated.")
    else:
        print("[DEBUG] No target folders to evaluate, empty CSV generated.")

if __name__ == "__main__":
    main()