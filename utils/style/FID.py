#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FID style similarity:
  <target-folder>/<image>  vs  style image inferred from image filename suffix

- File ending with `_sNN`          →  {style_root}/sNN.jpg
- File ending with `_wikiart_XXX`  →  {style_root}/wikiart_100/wikiart_XXX.jpg
- Evaluate all image files (.png/.jpg/.jpeg/.webp/.bmp)
- Results: CSV save + overall/style-specific statistics + failure reason aggregation
"""

"""
Usage: python FID_style.py
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
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

# 폴더명 접미부 패턴
SUFFIX_S_RE = re.compile(r"_s(\d{1,3})$")          # ..._s12
SUFFIX_WIKI_RE = re.compile(r"_wikiart_(\d{3})$")  # ..._wikiart_006

ALT_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

def resolve_style_image(style_root: Path, filename: str) -> Optional[Path]:
    # sNN 계열
    m = SUFFIX_S_RE.search(filename)
    if m:
        num = int(m.group(1))  # s12.jpg (0패딩 없음)
        for ext in [".jpg", ".png", ".jpeg"]:
            p = style_root / f"s{num}{ext}"
            if p.exists(): return p
        return None
    # wikiart_XXX 계열
    m = SUFFIX_WIKI_RE.search(filename)
    if m:
        idx3 = m.group(1)  # '006'
        for ext in [".jpg", ".png", ".jpeg"]:
            p = style_root / "wikiart_100" / f"wikiart_{idx3}{ext}"
            if p.exists(): return p
        return None
    return None

class FIDScorer:
    def __init__(self, device: str):
        self.device = torch.device("cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu")
        self.model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
        
        # Remove the final classification layer
        self.model.fc = torch.nn.Identity()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def img_feat(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        f = self.model(x)
        return f.squeeze(0)
    
    def calculate_fid_pairwise(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """
        Calculate FID between two single feature vectors (simplified FID for pairs)
        """
        # Convert to numpy
        f1 = feat1.cpu().numpy()
        f2 = feat2.cpu().numpy()
        
        # For pairwise comparison, use L2 distance as proxy
        # (True FID needs distributions, but this gives comparable relative scores)
        distance = np.linalg.norm(f1 - f2)
        return float(distance)

def main():
    # Hardcoded values - modify these as needed
    target_folder = "/data2/jiyoon/custom/results/final/infer_7"
    out_file = "/data2/jiyoon/custom/results/final/metric/FID_style/FID_style_infer7.csv"
    style_root_path = "/data2/jiyoon/custom/data/ablation/style"
    device = "cuda"

    root_dir = Path(target_folder)
    style_root = Path(style_root_path)
    assert root_dir.is_dir(), f"target-folder path is not a directory: {root_dir}"
    assert style_root.is_dir(), f"style root does not exist: {style_root}"

    fid_scorer = FIDScorer(device)
    print(f"[INFO] device={fid_scorer.device}, model=InceptionV3")
    print(f"[INFO] target_folder={root_dir} | style_root={style_root}")

    rows: List[Dict] = []
    all_images = sorted([p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() in ALT_EXTS])
    print(f"[INFO] total_images={len(all_images)}")

    # Style feature cache (reuse same styles)
    style_feat_cache: Dict[str, Optional[torch.Tensor]] = {}

    for img_path in tqdm(all_images, desc="Evaluating", ncols=100):
        row = {
            "image": img_path.name,
            "style_key": "",
            "style_path": "",
            "target_path": str(img_path),
            "fid_style": np.nan,
            "notes": ""
        }

        # Style image path (extracted from filename)
        style_path = resolve_style_image(style_root, img_path.stem)  # .stem removes extension
        if not style_path:
            row["notes"] = "style_missing"; rows.append(row); continue
        row["style_path"] = str(style_path)

        # Style key (for logging/group statistics)
        if SUFFIX_S_RE.search(img_path.stem):
            row["style_key"] = f"s{SUFFIX_S_RE.search(img_path.stem).group(1)}"
        elif SUFFIX_WIKI_RE.search(img_path.stem):
            row["style_key"] = f"wikiart_{SUFFIX_WIKI_RE.search(img_path.stem).group(1)}"

        # Feature cache
        key = row["style_key"] or row["style_path"]
        if key not in style_feat_cache:
            try:
                s_img = Image.open(style_path).convert("RGB")
                style_feat_cache[key] = fid_scorer.img_feat(s_img)
            except Exception:
                style_feat_cache[key] = None
        s_feat = style_feat_cache[key]
        if s_feat is None:
            row["notes"] = "style_open_or_feat_error"; rows.append(row); continue

        # Target feature extraction and FID calculation
        try:
            g_img = Image.open(img_path).convert("RGB")
            g_feat = fid_scorer.img_feat(g_img)
            row["fid_style"] = fid_scorer.calculate_fid_pairwise(s_feat, g_feat)
        except Exception:
            row["notes"] = "target_open_or_feat_error"
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(out_file); df.to_csv(out, index=False)
    print(f"\n[DONE] saved: {out.resolve()}")

    # Statistics/Aggregation
    if not df.empty:
        print("[DEBUG] notes counts:", dict(Counter(df["notes"].fillna(""))))
        if df["fid_style"].notna().any():
            s = df["fid_style"].dropna()
            print(f"[STAT | overall] count={len(s)}, mean={s.mean():.6f}, min={s.min():.6f}, max={s.max():.6f}")
            try:
                grp = df[df["fid_style"].notna()].groupby("style_key")["fid_style"].mean().sort_values(ascending=True)
                print("\n[STAT | by style_key] mean fid_style (lower is better):")
                for k, v in grp.items():
                    print(f"  {k:>12s}: {v:.6f}")
            except Exception:
                pass
        else:
            print("[STAT] No valid FID values.")
    else:
        print("[DEBUG] No target images to evaluate, empty CSV generated.")

if __name__ == "__main__":
    main()