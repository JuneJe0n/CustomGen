#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP image-image similarity:
  <target-folder>/<folder>/<image>  vs  FFHQ/{bucket}/{id5}.{ext}

- 폴더명: ^\\d{5}_[A-Za-z] ...  → 앞 5자리(id5)로 FFHQ 앵커 매칭
- 전체 폴더 평가(알파벳 필터 없음)
- FFHQ 앵커: {ffhq_root}/{id5//bucket_size:05d}/{id5}.png(.jpg)
- 생성 이미지:
    * 기본: 폴더 내 이미지(.png/.jpg/.jpeg/.webp/.bmp) 중 사전순 첫 번째 자동 선택
    * 옵션: --target-name 로 파일 이름 지정(확장자 생략/접두(prefix) 매칭 지원)
- 결과: CSV 저장 + 전체 통계(mean/min/max) + 실패 사유 집계
"""

"""
실행 명령어 예시
python CLIP_id.py \
--target-folder /data2/jiyoon/custom/results/final/infer_6 \
--out /data2/jiyoon/custom/results/final/metric/CLIP_ID_infer6.csv
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

ID_ALPHA_RE = re.compile(r"^(\d{5})_([A-Za-z])")
ALT_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
ANCHOR_EXTS = [".png", ".jpg", ".jpeg"]  # FFHQ 앵커 확장자 후보

def resolve_ffhq_anchor(ffhq_root: Path, id5: str, bucket_size: int, recursive_fallback: bool = False) -> Optional[Path]:
    """
    FFHQ 앵커 이미지를 찾는다.
    우선순위:
      1) 버킷 레이아웃({id5//bucket_size:05d}/{id5}.{ext})
      2) 평면({ffhq_root}/{id5}.{ext})
      3) (옵션) 재귀 rglob 탐색
    """
    idx = int(id5)

    # 1) 지정된 버킷 레이아웃
    if bucket_size and bucket_size > 0:
        folder = f"{idx // bucket_size:05d}"
        for ext in ANCHOR_EXTS:
            p = ffhq_root / folder / f"{id5}{ext}"
            if p.exists():
                return p

    # 2) 평면 레이아웃
    for ext in ANCHOR_EXTS:
        p = ffhq_root / f"{id5}{ext}"
        if p.exists():
            return p

    # 3) (선택) 재귀 탐색 — 느릴 수 있어 기본 비활성
    if recursive_fallback:
        for ext in ANCHOR_EXTS:
            hits = list(ffhq_root.rglob(f"{id5}{ext}"))
            if hits:
                return hits[0]

    return None

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

class CLIPScorer:
    def __init__(self, model_name: str, pretrained: str, device: str):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
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
    ap = argparse.ArgumentParser(description="CLIP image-image similarity (no alpha filter; auto target)")
    ap.add_argument("--target-folder", required=True, help="생성 이미지 루트(직계 하위 폴더 순회)")
    ap.add_argument("--ffhq-root", default="/data2/jeesoo/FFHQ", help="FFHQ 루트")
    ap.add_argument("--bucket-size", type=int, default=1000, help="FFHQ 버킷 크기")
    ap.add_argument("--target-name", default="", help="폴더 내 사용할 이미지 이름(옵션). 미지정 시 자동 선택")
    ap.add_argument("--clip-model", default="ViT-L-14", help="open_clip 모델명 (ViT-L-14, ViT-L-14-336, ViT-B-32 등)")
    ap.add_argument("--clip-pretrained", default="openai", help="사전학습 가중치 (openai, laion2b_s32b_b82k 등)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="CLIP 실행 디바이스")
    ap.add_argument("--out", default="clip_identity_similarity_all.csv", help="결과 CSV 경로")
    args = ap.parse_args()

    root_dir = Path(args.target_folder)
    ffhq_root = Path(args.ffhq_root)
    assert root_dir.is_dir(), f"--target-folder 경로가 폴더가 아닙니다: {root_dir}"
    assert ffhq_root.is_dir(), f"FFHQ 루트가 없습니다: {ffhq_root}"

    clip = CLIPScorer(args.clip_model, args.clip_pretrained, args.device)
    print(f"[INFO] device={clip.device}, model={args.clip_model}:{args.clip_pretrained}")
    print(f"[INFO] target_folder={root_dir} | ffhq_root={ffhq_root} | bucket_size={args.bucket_size} | target_name={args.target_name or '(auto)'}")

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

    # 앵커 임베딩 캐시(같은 id5 중복 대비)
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
                "clip_i": np.nan,
                "notes": ""
            }
            
            # 타겟 이미지
            target_path = resolve_target_image(item, args.target_name or None)
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
                "clip_i": np.nan,
                "notes": ""
            }
            target_path = item

        # 앵커 이미지
        anchor_path = resolve_ffhq_anchor(ffhq_root, id5, args.bucket_size, recursive_fallback=False)
        if not anchor_path:
            row["notes"] = "anchor_missing"; rows.append(row); continue
        row["anchor_path"] = str(anchor_path)

        # 앵커 임베딩
        if id5 not in anchor_feat_cache:
            try:
                ref_img = Image.open(anchor_path).convert("RGB")
                anchor_feat_cache[id5] = clip.img_feat(ref_img)
            except Exception:
                anchor_feat_cache[id5] = None
        a_feat = anchor_feat_cache[id5]
        if a_feat is None:
            row["notes"] = "anchor_open_or_feat_error"; rows.append(row); continue

        # 타겟 임베딩 및 유사도
        try:
            gen_img = Image.open(target_path).convert("RGB")
            g_feat = clip.img_feat(gen_img)
            row["clip_i"] = float((a_feat @ g_feat).item())
        except Exception:
            row["notes"] = "target_open_or_feat_error"
        rows.append(row)

    # CSV 저장
    df = pd.DataFrame(rows)
    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"\n[DONE] saved: {out.resolve()}")

    # 통계/집계
    if not df.empty:
        print("[DEBUG] notes counts:", dict(Counter(df["notes"].fillna(""))))
        if df["clip_i"].notna().any():
            s = df["clip_i"].dropna()
            print(f"[STAT] count={len(s)}, mean={s.mean():.6f}, min={s.min():.6f}, max={s.max():.6f}")
        else:
            print("[STAT] CLIP 유효값이 없습니다.")
    else:
        print("[DEBUG] 평가 대상 폴더가 없어 빈 CSV가 생성되었습니다.")

if __name__ == "__main__":
    main()