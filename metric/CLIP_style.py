#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP style similarity:
  <target-folder>/<folder>/<image>  vs  style image inferred from folder name suffix

- 폴더명 끝이 `_sNN`          →  {style_root}/sNN.jpg
- 폴더명 끝이 `_wikiart_XXX`  →  {style_root}/wikiart_100/wikiart_XXX.jpg
- 전체 폴더 평가(알파벳 필터 없음)
- 생성 이미지:
    * 기본: 폴더 내 이미지(.png/.jpg/.jpeg/.webp/.bmp) 중 사전순 첫 번째 자동 선택
    * 옵션: --target-name 로 파일 이름 지정(확장자 생략/접두(prefix) 매칭 지원)
- 결과: CSV 저장 + 전체/스타일별 통계 + 실패 사유 집계
"""

"""
실행 명령어 예시
python CLIP_style.py \
  --target-folder /data2/jeesoo/custom/results/final/infer_1 \
  --target-name 1_final_styled_result  
"""

import argparse
from pathlib import Path
import re
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from tqdm import tqdm

import torch
import open_clip

# 폴더명 접미부 패턴
SUFFIX_S_RE = re.compile(r"_s(\d{1,3})$")          # ..._s12
SUFFIX_WIKI_RE = re.compile(r"_wikiart_(\d{3})$")  # ..._wikiart_006

ALT_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

def resolve_target_image(folder: Path, target_name: Optional[str]) -> Optional[Path]:
    if target_name:
        base = target_name.strip()
        # 확장자 포함
        if Path(base).suffix:
            p = folder / base
            if p.exists():
                return p
        else:
            # 확장자 미포함
            for ext in ALT_EXTS:
                p = folder / f"{base}{ext}"
                if p.exists():
                    return p
        # prefix 매칭
        cands = sorted([q for q in folder.glob(f"{base}*") if q.suffix.lower() in ALT_EXTS])
        if cands:
            return cands[0]
        return None

    # 자동 선택
    imgs = sorted([q for q in folder.iterdir() if q.is_file() and q.suffix.lower() in ALT_EXTS])
    return imgs[0] if imgs else None

def resolve_style_image(style_root: Path, folder_name: str) -> Optional[Path]:
    # sNN 계열
    m = SUFFIX_S_RE.search(folder_name)
    if m:
        num = int(m.group(1))  # s12.jpg (0패딩 없음)
        for ext in [".jpg", ".png", ".jpeg"]:
            p = style_root / f"s{num}{ext}"
            if p.exists(): return p
        return None
    # wikiart_XXX 계열
    m = SUFFIX_WIKI_RE.search(folder_name)
    if m:
        idx3 = m.group(1)  # '006'
        for ext in [".jpg", ".png", ".jpeg"]:
            p = style_root / "wikiart_100" / f"wikiart_{idx3}{ext}"
            if p.exists(): return p
        return None
    return None

class CLIPScorer:
    def __init__(self, model_name: str, pretrained: str, device: str):
        self.device = torch.device("cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def img_feat(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.squeeze(0)

def main():
    ap = argparse.ArgumentParser(description="CLIP style similarity by folder suffix (no alpha filter; auto target)")
    ap.add_argument("--target-folder", required=True, help="생성 이미지 루트(직계 하위 폴더 순회)")
    ap.add_argument("--style-root", default="/data2/jiyoon/custom/data/style",
                    help="스타일 이미지 루트 (sNN.jpg, wikiart_100/wikiart_XXX.jpg)")
    ap.add_argument("--target-name", default="", help="폴더 내 사용할 이미지 이름(옵션). 미지정 시 자동 선택")
    ap.add_argument("--clip-model", default="ViT-L-14", help="open_clip 모델명 (ViT-L-14, ViT-L-14-336, ViT-B-32 등)")
    ap.add_argument("--clip-pretrained", default="openai", help="사전학습 가중치 (openai, laion2b_s32b_b82k 등)")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="CLIP 실행 디바이스")
    ap.add_argument("--out", default="clip_style_all.csv", help="결과 CSV 경로")
    args = ap.parse_args()

    root_dir = Path(args.target_folder)
    style_root = Path(args.style_root)
    assert root_dir.is_dir(), f"--target-folder 경로가 폴더가 아닙니다: {root_dir}"
    assert style_root.is_dir(), f"style 루트가 없습니다: {style_root}"

    clip = CLIPScorer(args.clip_model, args.clip_pretrained, args.device)
    print(f"[INFO] device={clip.device}, model={args.clip_model}:{args.clip_pretrained}")
    print(f"[INFO] target_folder={root_dir} | style_root={style_root} | target_name={args.target_name or '(auto)'}")

    rows: List[Dict] = []
    all_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    print(f"[INFO] total_subdirs={len(all_dirs)}")

    # 스타일 임베딩 캐시 (같은 스타일은 재사용)
    style_feat_cache: Dict[str, Optional[torch.Tensor]] = {}

    for sub in tqdm(all_dirs, desc="Evaluating", ncols=100):
        row = {
            "folder": sub.name,
            "style_key": "",
            "style_path": "",
            "target_path": "",
            "clip_style": np.nan,
            "notes": ""
        }

        # 타겟 이미지
        target_path = resolve_target_image(sub, args.target_name or None)
        if not target_path:
            row["notes"] = "target_missing"; rows.append(row); continue
        row["target_path"] = str(target_path)

        # 스타일 이미지 경로
        style_path = resolve_style_image(style_root, sub.name)
        if not style_path:
            row["notes"] = "style_missing"; rows.append(row); continue
        row["style_path"] = str(style_path)

        # 스타일 키 (로그/그룹 통계용)
        if SUFFIX_S_RE.search(sub.name):
            row["style_key"] = f"s{SUFFIX_S_RE.search(sub.name).group(1)}"
        elif SUFFIX_WIKI_RE.search(sub.name):
            row["style_key"] = f"wikiart_{SUFFIX_WIKI_RE.search(sub.name).group(1)}"

        # 임베딩 캐시
        key = row["style_key"] or row["style_path"]
        if key not in style_feat_cache:
            try:
                s_img = Image.open(style_path).convert("RGB")
                style_feat_cache[key] = clip.img_feat(s_img)
            except Exception:
                style_feat_cache[key] = None
        s_feat = style_feat_cache[key]
        if s_feat is None:
            row["notes"] = "style_open_or_feat_error"; rows.append(row); continue

        # 타겟 임베딩 및 유사도
        try:
            g_img = Image.open(target_path).convert("RGB")
            g_feat = clip.img_feat(g_img)
            row["clip_style"] = float((s_feat @ g_feat).item())
        except Exception:
            row["notes"] = "target_open_or_feat_error"
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.out); df.to_csv(out, index=False)
    print(f"\n[DONE] saved: {out.resolve()}")

    # 통계/집계
    if not df.empty:
        print("[DEBUG] notes counts:", dict(Counter(df["notes"].fillna(""))))
        if df["clip_style"].notna().any():
            s = df["clip_style"].dropna()
            print(f"[STAT | overall] count={len(s)}, mean={s.mean():.6f}, min={s.min():.6f}, max={s.max():.6f}")
            try:
                grp = df[df["clip_style"].notna()].groupby("style_key")["clip_style"].mean().sort_values(ascending=False)
                print("\n[STAT | by style_key] mean clip_style:")
                for k, v in grp.items():
                    print(f"  {k:>12s}: {v:.6f}")
            except Exception:
                pass
        else:
            print("[STAT] CLIP 유효값이 없습니다.")
    else:
        print("[DEBUG] 평가 대상 폴더가 없어 빈 CSV가 생성되었습니다.")

if __name__ == "__main__":
    main()
