#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArtFID style similarity:
  <target-folder>/<image>  vs  style image inferred from image filename suffix

- File ending with `_sNN`          →  {style_root}/sNN.jpg
- File ending with `_wikiart_XXX`  →  {style_root}/wikiart_100/wikiart_XXX.jpg
- Evaluate all image files (.png/.jpg/.jpeg/.webp/.bmp)
- Results: CSV save + overall/style-specific statistics + failure reason aggregation
"""

"""
Usage: python ArtFID_style.py
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
import sys
import os
# Add the art_fid directory to path for importing
sys.path.append(os.path.join(os.path.dirname(__file__), 'art_fid'))
from art_fid.art_fid import compute_art_fid, get_activations
from art_fid.inception import Inception3

# Filename suffix patterns
SUFFIX_S_RE = re.compile(r"_s(\d{1,3})$")          # ..._s12
SUFFIX_WIKI_RE = re.compile(r"_wikiart_(\d{3})$")  # ..._wikiart_006

ALT_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

def resolve_style_image(style_root: Path, filename: str) -> Optional[Path]:
    # sNN series
    m = SUFFIX_S_RE.search(filename)
    if m:
        num = int(m.group(1))  # s12.jpg (no zero padding)
        for ext in [".jpg", ".png", ".jpeg"]:
            p = style_root / f"s{num}{ext}"
            if p.exists(): return p
        return None
    # wikiart_XXX series
    m = SUFFIX_WIKI_RE.search(filename)
    if m:
        idx3 = m.group(1)  # '006'
        for ext in [".jpg", ".png", ".jpeg"]:
            p = style_root / "wikiart_100" / f"wikiart_{idx3}{ext}"
            if p.exists(): return p
        return None
    return None

class ArtFIDScorer:
    def __init__(self, device: str):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # Load the art-trained Inception3 model for feature extraction
        self.model = Inception3().to(self.device)
        self.model.eval()
        
    def calculate_artfid_pairwise(self, img1_path: str, img2_path: str) -> float:
        """
        Calculate art-domain feature similarity between two individual images
        Uses the art-trained Inception3 model for feature extraction and cosine similarity
        """
        try:
            # Extract features using the art-trained model
            feat1 = get_activations([img1_path], self.model, batch_size=1, device=self.device, num_workers=1)
            feat2 = get_activations([img2_path], self.model, batch_size=1, device=self.device, num_workers=1)
            
            import numpy as np
            # Normalize features to unit vectors
            feat1_norm = feat1[0] / (np.linalg.norm(feat1[0]) + 1e-8)
            feat2_norm = feat2[0] / (np.linalg.norm(feat2[0]) + 1e-8)
            
            # Calculate cosine similarity (ranges from -1 to 1, higher is better)
            cosine_sim = np.dot(feat1_norm, feat2_norm)
            
            # Convert to distance: 1 - cosine_similarity (ranges from 0 to 2, lower is better)
            distance = 1.0 - cosine_sim
            
            return float(distance)
            
        except Exception as e:
            print(f"Art feature extraction failed: {e}")
            # Return a high distance value for failed calculations  
            return 2.0

def main():
    # Hardcoded values - modify these as needed
    target_folder = "/data2/jiyoon/custom/results/final/infer_8"
    out_file = "/data2/jiyoon/custom/results/final/metric/ArtFID/ArtFID_style_infer8.csv"
    style_root_path = "/data2/jiyoon/custom/data/ablation/style"
    device = "cuda"

    root_dir = Path(target_folder)
    style_root = Path(style_root_path)
    assert root_dir.is_dir(), f"target-folder path is not a directory: {root_dir}"
    assert style_root.is_dir(), f"style root does not exist: {style_root}"

    artfid_scorer = ArtFIDScorer(device)
    print(f"[INFO] device={device}, model=ArtFID")
    print(f"[INFO] target_folder={root_dir} | style_root={style_root}")

    rows: List[Dict] = []
    all_images = sorted([p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() in ALT_EXTS])
    print(f"[INFO] total_images={len(all_images)}")

    # Note: ArtFID works directly with image paths, so no feature caching needed

    for img_path in tqdm(all_images, desc="Evaluating", ncols=100):
        row = {
            "image": img_path.name,
            "style_key": "",
            "style_path": "",
            "target_path": str(img_path),
            "artfid_style": np.nan,
            "notes": ""
        }

        # Style image path (extracted from filename)
        style_path = resolve_style_image(style_root, img_path.stem)  # .stem removes extension
        if not style_path:
            row["notes"] = f"style_missing_for_{img_path.stem}"; rows.append(row); continue
        row["style_path"] = str(style_path)

        # Style key (for logging/group statistics)
        if SUFFIX_S_RE.search(img_path.stem):
            row["style_key"] = f"s{SUFFIX_S_RE.search(img_path.stem).group(1)}"
        elif SUFFIX_WIKI_RE.search(img_path.stem):
            row["style_key"] = f"wikiart_{SUFFIX_WIKI_RE.search(img_path.stem).group(1)}"

        # Calculate ArtFID score between target and style images
        try:
            # Verify that both images exist and are readable
            if not img_path.exists() or not style_path.exists():
                row["notes"] = "image_missing"
            else:
                # Calculate ArtFID score using image paths
                artfid_score = artfid_scorer.calculate_artfid_pairwise(str(img_path), str(style_path))
                row["artfid_style"] = artfid_score
        except Exception as e:
            row["notes"] = f"artfid_calculation_error: {str(e)}"
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(out_file); df.to_csv(out, index=False)
    print(f"\n[DONE] saved: {out.resolve()}")

    # Statistics/Aggregation
    if not df.empty:
        print("[DEBUG] notes counts:", dict(Counter(df["notes"].fillna(""))))
        if df["artfid_style"].notna().any():
            s = df["artfid_style"].dropna()
            # Filter out infinite values (failed calculations)
            s_finite = s[s != float('inf')]
            if len(s_finite) > 0:
                print(f"[STAT | overall] count={len(s_finite)}, mean={s_finite.mean():.6f}, min={s_finite.min():.6f}, max={s_finite.max():.6f}")
                try:
                    df_finite = df[df["artfid_style"].notna() & (df["artfid_style"] != float('inf'))]
                    grp = df_finite.groupby("style_key")["artfid_style"].mean().sort_values(ascending=True)
                    print("\n[STAT | by style_key] mean artfid_style (lower is better):")
                    for k, v in grp.items():
                        print(f"  {k:>12s}: {v:.6f}")
                except Exception:
                    pass
            else:
                print("[STAT] No valid ArtFID values (all calculations failed).")
        else:
            print("[STAT] No valid ArtFID values.")
    else:
        print("[DEBUG] No target images to evaluate, empty CSV generated.")

if __name__ == "__main__":
    main()