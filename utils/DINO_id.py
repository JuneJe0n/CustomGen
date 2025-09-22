
"""
DINO score computation:
- <target-folder>  vs  FFHQ/{bucket}/{id5}.{ext}

Usage: python DINO_id.py
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

import torch
import torchvision.transforms as transforms
from torch.nn import functional as F

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

class DINOScorer:
    def __init__(self, device: str, model_name: str = 'dino_vits16'):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        
        # Load DINO model from torch hub
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(self.device)
        self.model.eval()
        
        # DINO preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def extract_features(self, img: Image.Image) -> torch.Tensor:
        """
        Extract DINO features from an image
        """
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model(x)
        # Normalize features
        features = F.normalize(features, dim=1)
        return features.squeeze(0)
    
    def calculate_dino_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate DINO similarity score between two images
        """
        # Extract features for both images
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0))
        return float(similarity.item())

def main():
    # Hardcoded values - modify these as needed
    target_folder = "/data2/jiyoon/custom/results/final/metric/CLIP_ID_facedet1/generated_face/infer_7"
    out_file = "/data2/jiyoon/custom/results/final/metric/DINO_ID/DINO_infer7.csv"
    ffhq_root_path = "/data2/jiyoon/custom/results/final/metric/CLIP_ID_facedet0/ffhq_face"
    bucket_size = 1000
    target_name = ""
    device = "cuda"
    dino_model = "dino_vits16"  # Options: 'dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8'

    root_dir = Path(target_folder)
    ffhq_root = Path(ffhq_root_path)
    assert root_dir.is_dir(), f"target-folder path is not a directory: {root_dir}"
    assert ffhq_root.is_dir(), f"FFHQ root does not exist: {ffhq_root}"

    dino_scorer = DINOScorer(device, dino_model)
    print(f"[INFO] device={dino_scorer.device}, model={dino_model}")
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

    # Store DINO scores for statistics
    dino_scores = []

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
                "dino_score": np.nan,
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
                "dino_score": np.nan,
                "notes": ""
            }
            target_path = item

        # Anchor image
        anchor_path = resolve_ffhq_anchor(ffhq_root, id5, bucket_size, recursive_fallback=False)
        if not anchor_path:
            row["notes"] = "anchor_missing"; rows.append(row); continue
        row["anchor_path"] = str(anchor_path)

        # Calculate DINO similarity score between anchor and target
        try:
            # Load images
            ref_img = Image.open(anchor_path).convert("RGB")
            gen_img = Image.open(target_path).convert("RGB")
            
            # Calculate DINO similarity score
            dino_score = dino_scorer.calculate_dino_similarity(ref_img, gen_img)
            row["dino_score"] = dino_score
            dino_scores.append(dino_score)
            
        except Exception as e:
            row["notes"] = "image_open_or_dino_error"
        
        rows.append(row)

    # Save CSV
    df = pd.DataFrame(rows)
    out = Path(out_file)
    df.to_csv(out, index=False)
    print(f"\n[DONE] saved: {out.resolve()}")

    # Statistics/Aggregation
    if not df.empty:
        print("[DEBUG] notes counts:", dict(Counter(df["notes"].fillna(""))))
        if dino_scores:
            dino_array = np.array(dino_scores)
            print(f"\n[DINO STATS] count={len(dino_scores)}, mean={dino_array.mean():.6f}, min={dino_array.min():.6f}, max={dino_array.max():.6f}, std={dino_array.std():.6f}")
        else:
            print("[STAT] No valid DINO scores calculated.")
    else:
        print("[DEBUG] No target folders to evaluate, empty CSV generated.")

if __name__ == "__main__":
    main()