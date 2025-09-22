#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LPIPS style similarity:
  <target-folder>/<image>  vs  style image inferred from image filename suffix

- File ending with `_sNN`          →  {style_root}/sNN.jpg
- File ending with `_wikiart_XXX`  →  {style_root}/wikiart_100/wikiart_XXX.jpg
- Evaluate all image files (.png/.jpg/.jpeg/.webp/.bmp)
- Results: CSV save + overall/style-specific statistics + failure reason aggregation
"""

"""
Usage: python LPIPS_style.py
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
import lpips

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

class LPIPSScorer:
    def __init__(self, device: str, net: str = 'alex'):
        self.device = torch.device("cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    @torch.inference_mode()
    def calculate_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate LPIPS score between two images
        """
        # Preprocess images
        x1 = self.preprocess(img1).unsqueeze(0).to(self.device)
        x2 = self.preprocess(img2).unsqueeze(0).to(self.device)
        
        # Calculate LPIPS distance
        distance = self.model(x1, x2)
        return float(distance.item())

def main():
    # Hardcoded values - modify these as needed
    target_folder = "/data2/jiyoon/custom/results/final/infer_7"
    out_file = "/data2/jiyoon/custom/results/final/metric/LPIPS_style/LPIPS_style_infer7.csv"
    style_root_path = "/data2/jiyoon/custom/data/ablation/style"
    device = "cuda"
    lpips_net = "alex"  # Options: 'alex', 'vgg', 'squeeze'

    root_dir = Path(target_folder)
    style_root = Path(style_root_path)
    assert root_dir.is_dir(), f"target-folder path is not a directory: {root_dir}"
    assert style_root.is_dir(), f"style root does not exist: {style_root}"

    lpips_scorer = LPIPSScorer(device, lpips_net)
    print(f"[INFO] device={lpips_scorer.device}, model=LPIPS-{lpips_net}")
    print(f"[INFO] target_folder={root_dir} | style_root={style_root}")

    rows: List[Dict] = []
    all_images = sorted([p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() in ALT_EXTS])
    print(f"[INFO] total_images={len(all_images)}")

    # Style image cache (reuse same styles to avoid reloading)
    style_image_cache: Dict[str, Optional[Image.Image]] = {}

    for img_path in tqdm(all_images, desc="Evaluating", ncols=100):
        row = {
            "image": img_path.name,
            "style_key": "",
            "style_path": "",
            "target_path": str(img_path),
            "lpips_style": np.nan,
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

        # Image cache
        key = row["style_key"] or row["style_path"]
        if key not in style_image_cache:
            try:
                s_img = Image.open(style_path).convert("RGB")
                style_image_cache[key] = s_img
            except Exception:
                style_image_cache[key] = None
        s_img = style_image_cache[key]
        if s_img is None:
            row["notes"] = "style_open_error"; rows.append(row); continue

        # Calculate LPIPS score between target and style images
        try:
            g_img = Image.open(img_path).convert("RGB")
            row["lpips_style"] = lpips_scorer.calculate_lpips(g_img, s_img)
        except Exception:
            row["notes"] = "target_open_or_lpips_error"
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(out_file); df.to_csv(out, index=False)
    print(f"\n[DONE] saved: {out.resolve()}")

    # Statistics/Aggregation
    if not df.empty:
        print("[DEBUG] notes counts:", dict(Counter(df["notes"].fillna(""))))
        if df["lpips_style"].notna().any():
            s = df["lpips_style"].dropna()
            print(f"[STAT | overall] count={len(s)}, mean={s.mean():.6f}, min={s.min():.6f}, max={s.max():.6f}")
            try:
                grp = df[df["lpips_style"].notna()].groupby("style_key")["lpips_style"].mean().sort_values(ascending=True)
                print("\n[STAT | by style_key] mean lpips_style (lower is better):")
                for k, v in grp.items():
                    print(f"  {k:>12s}: {v:.6f}")
            except Exception:
                pass
        else:
            print("[STAT] No valid LPIPS values.")
    else:
        print("[DEBUG] No target images to evaluate, empty CSV generated.")

if __name__ == "__main__":
    main()