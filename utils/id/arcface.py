
"""
ArcFace similarity computation:
- <target-folder>  vs  FFHQ/{bucket}/{id5}.{ext}

Usage: python arface.py
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
import torch.nn as nn
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

class ArcFaceScorer:
    def __init__(self, device: str, model_path: str = None):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        
        # Load ArcFace model - you may need to adjust this based on your model
        # This is a placeholder - replace with actual ArcFace model loading
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, 512)  # ArcFace embedding dimension
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        
        # If you have a pretrained ArcFace model, load it here
        if model_path and Path(model_path).exists():
            self.backbone.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.preprocess = transforms.Compose([
            transforms.Resize((112, 112)),  # Standard ArcFace input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] normalization
        ])

    @torch.inference_mode()
    def img_feat(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        embedding = self.backbone(x)
        # L2 normalize the embedding for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0)
    
    def calculate_cosine_similarity(self, anchor_embedding, target_embedding):
        """
        Calculate cosine similarity between two embeddings
        """
        # Both embeddings are already L2 normalized, so dot product gives cosine similarity
        similarity = torch.dot(anchor_embedding, target_embedding).item()
        return similarity
    
    def calculate_average_similarity(self, real_embeddings, fake_embeddings):
        """
        Calculate average cosine similarity across all pairs
        """
        if len(real_embeddings) != len(fake_embeddings):
            raise ValueError("Number of real and fake embeddings must match")
        
        similarities = []
        for real_emb, fake_emb in zip(real_embeddings, fake_embeddings):
            sim = self.calculate_cosine_similarity(real_emb, fake_emb)
            similarities.append(sim)
        
        return np.mean(similarities)

def main():
    # Hardcoded values - modify these as needed
    target_folder = "/data2/jiyoon/custom/results/final/metric/CLIP_ID_facedet1/generated_face/infer_6"
    out_file = "/data2/jiyoon/custom/results/final/metric/arcface/ArcFace_ID_infer6.csv"
    ffhq_root_path = "/data2/jiyoon/custom/results/final/metric/CLIP_ID_facedet0/ffhq_face"
    bucket_size = 1000
    target_name = ""
    device = "cuda"
    arcface_model_path = None  # Path to pretrained ArcFace model (optional)

    root_dir = Path(target_folder)
    ffhq_root = Path(ffhq_root_path)
    assert root_dir.is_dir(), f"target-folder path is not a directory: {root_dir}"
    assert ffhq_root.is_dir(), f"FFHQ root does not exist: {ffhq_root}"

    arcface_scorer = ArcFaceScorer(device, arcface_model_path)
    print(f"[INFO] device={arcface_scorer.device}, model=ArcFace-ResNet50")
    print(f"[INFO] target_folder={root_dir} | ffhq_root={ffhq_root} | bucket_size={bucket_size} | target_name={target_name or '(auto)'}")
    
    # Collect all embeddings for similarity calculation
    anchor_embeddings = []
    target_embeddings = []

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

    # Anchor embedding cache (to handle duplicate id5s)
    anchor_feat_cache: Dict[str, Optional[torch.Tensor]] = {}

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
                "arcface_similarity": np.nan,
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
                "arcface_similarity": np.nan,
                "notes": ""
            }
            target_path = item

        # Anchor image
        anchor_path = resolve_ffhq_anchor(ffhq_root, id5, bucket_size, recursive_fallback=False)
        if not anchor_path:
            row["notes"] = "anchor_missing"; rows.append(row); continue
        row["anchor_path"] = str(anchor_path)

        # Extract embeddings for ArcFace similarity calculation
        try:
            # Anchor image embedding
            if id5 not in anchor_feat_cache:
                try:
                    ref_img = Image.open(anchor_path).convert("RGB")
                    anchor_feat_cache[id5] = arcface_scorer.img_feat(ref_img)
                except Exception:
                    anchor_feat_cache[id5] = None
            
            a_emb = anchor_feat_cache[id5]
            if a_emb is None:
                row["notes"] = "anchor_open_or_feat_error"; rows.append(row); continue
            
            # Target image embedding
            gen_img = Image.open(target_path).convert("RGB")
            g_emb = arcface_scorer.img_feat(gen_img)
            
            # Calculate cosine similarity between the two embeddings
            similarity = arcface_scorer.calculate_cosine_similarity(a_emb, g_emb)
            row["arcface_similarity"] = similarity
            
            # Store embeddings for average calculation
            anchor_embeddings.append(a_emb)
            target_embeddings.append(g_emb)
            
        except Exception as e:
            row["notes"] = "target_open_or_feat_error"
        
        rows.append(row)

    # Save CSV
    df = pd.DataFrame(rows)
    out = Path(out_file)
    df.to_csv(out, index=False)
    print(f"\n[DONE] saved: {out.resolve()}")

    # Calculate overall average similarity
    if anchor_embeddings and target_embeddings:
        overall_avg_similarity = arcface_scorer.calculate_average_similarity(anchor_embeddings, target_embeddings)
        print(f"\n[ARCFACE SIMILARITY] Average similarity: {overall_avg_similarity:.6f}")
        
        # Calculate individual statistics
        valid_similarities = [row["arcface_similarity"] for row in rows if not np.isnan(row["arcface_similarity"])]
        if valid_similarities:
            min_sim = min(valid_similarities)
            max_sim = max(valid_similarities)
            print(f"[ARCFACE SIMILARITY] Min: {min_sim:.6f}, Max: {max_sim:.6f}")
    
    # Statistics/Aggregation
    if not df.empty:
        print("[DEBUG] notes counts:", dict(Counter(df["notes"].fillna(""))))
        valid_pairs = len([emb for emb in anchor_embeddings if emb is not None])
        print(f"[STAT] Valid image pairs processed: {valid_pairs}")
        if valid_pairs == 0:
            print("[STAT] No valid image pairs for similarity calculation.")
    else:
        print("[DEBUG] No target folders to evaluate, empty CSV generated.")

if __name__ == "__main__":
    main()