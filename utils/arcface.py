#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArcFace: <target-folder>/<folder>/<image> vs FFHQ anchor
- 폴더명 패턴: ^\\d{5}_[A-Za-z] ... (예: 00000_b_6_wikiart_011)  → id5=앞 5자리
- 전체 폴더 평가
- FFHQ 앵커: {ffhq_root}/{id5//bucket_size:05d}/{id5}.png(.jpg)
- 생성 이미지:
    * 기본: 해당 폴더 내 이미지 파일(확장자: png/jpg/jpeg/webp/bmp) 중 하나 자동 선택(사전순)
    * 옵션: --target-name 로 이름 지정 가능 (확장자 생략 가능, prefix 매칭 지원)
- 결과 CSV 저장 + 평균/최솟값/최댓값 출력
"""

"""
실행 명령어 예시
python arcface.py \
--target-folder /data2/jiyoon/custom/results/final/infer_6 \
--out /data2/jiyoon/custom/results/final/metric/arcface_infer6.csv
"""

import argparse
from pathlib import Path
import re
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis

ID_ALPHA_RE = re.compile(r"^(\d{5})_([A-Za-z])")  # 앞 5자리 + '_' + 알파벳
ALT_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

def has_cuda_provider() -> bool:
    try:
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False

def bgr_from_pil(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def select_largest_face(faces):
    if not faces:
        return None
    areas = [max(1, int((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))) for f in faces]
    return faces[int(np.argmax(areas))]

def init_face_app(use_gpu: bool, det_size: int) -> FaceAnalysis:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))
    return app

def load_anchor_embedding(app: FaceAnalysis, anchor_path: Path) -> Optional[np.ndarray]:
    try:
        img = Image.open(anchor_path).convert("RGB")
    except Exception:
        return None
    faces = app.get(bgr_from_pil(img))
    face = select_largest_face(faces)
    if face is None or face.normed_embedding is None:
        return None
    return face.normed_embedding.astype(np.float32)  # L2-normalized

def best_sim_against_anchor(app: FaceAnalysis, img_path: Path, anchor: np.ndarray) -> Optional[float]:
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None
    faces = app.get(bgr_from_pil(img))
    if not faces:
        return None
    sims = []
    for f in faces:
        if f.normed_embedding is None:
            continue
        emb = f.normed_embedding.astype(np.float32)  # L2-normalized
        sims.append(float(np.dot(anchor, emb)))      # cosine == dot
    return max(sims) if sims else None

def resolve_ffhq_anchor(ffhq_root: Path, id5: str, bucket_size: int) -> Optional[Path]:
    idx = int(id5)
    folder = f"{idx // bucket_size:05d}"
    png = ffhq_root / folder / f"{id5}.png"
    if png.exists():
        return png
    jpg = ffhq_root / folder / f"{id5}.jpg"
    if jpg.exists():
        return jpg
    return None

def resolve_target_image(folder: Path, target_name: Optional[str]) -> Optional[Path]:
    """
    - target_name이 주어지면:
        1) 확장자 포함 시 그 파일 검사
        2) 확장자 미포함 시 ALT_EXTS 순회해 존재하면 반환
        3) prefix 매칭 (예: name -> name*, 이미지 확장자)
    - target_name이 없으면:
        폴더 내 ALT_EXTS 파일들을 사전순 정렬 후 첫 번째 반환
    """
    if target_name:
        base = target_name.strip()
        # 확장자 포함
        if Path(base).suffix:
            p = folder / base
            if p.exists():
                return p
        else:
            # 확장자 미포함 → 후보 탐색
            for ext in ALT_EXTS:
                p = folder / f"{base}{ext}"
                if p.exists():
                    return p
        # prefix 매칭
        cands = sorted([q for q in folder.glob(f"{base}*") if q.suffix.lower() in ALT_EXTS])
        if cands:
            return cands[0]
        return None

    # 자동 선택: 폴더 내 이미지 한 장 선택(사전순)
    imgs = sorted([q for q in folder.iterdir() if q.is_file() and q.suffix.lower() in ALT_EXTS])
    return imgs[0] if imgs else None

def main():
    ap = argparse.ArgumentParser(description="ArcFace vs FFHQ anchor (no alpha filter; auto-target)")
    ap.add_argument("--target-folder",
                    required=True,
                    help="생성 이미지 루트(직계 하위 폴더 순회)")
    ap.add_argument("--ffhq-root", default="/data2/jeesoo/FFHQ", help="FFHQ 루트 경로")
    ap.add_argument("--bucket-size", type=int, default=1000, help="FFHQ 버킷 크기(기본 1000)")
    ap.add_argument("--target-name", default="", help="폴더 내에서 사용할 이미지 이름(옵션). 미지정 시 자동 선택")
    ap.add_argument("--det-size", type=int, default=640, help="얼굴 검출 해상도 한변")
    ap.add_argument("--cpu", action="store_true", help="GPU 대신 CPU 사용(onnxruntime)")
    ap.add_argument("--out", default="arcface_all.csv", help="결과 CSV 경로")
    args = ap.parse_args()

    root_dir = Path(args.target_folder); ffhq_root = Path(args.ffhq_root)
    assert root_dir.is_dir(), f"--target-folder 경로가 폴더가 아닙니다: {root_dir}"
    assert ffhq_root.is_dir(), f"FFHQ 루트가 없습니다: {ffhq_root}"

    use_gpu = (not args.cpu) and has_cuda_provider()
    app = init_face_app(use_gpu, args.det_size)
    print(f"[INFO] providers={'GPU' if use_gpu else 'CPU'} | det_size={args.det_size}")
    print(f"[INFO] target_folder={root_dir} | ffhq_root={ffhq_root} | target_name={args.target_name or '(auto)'}")

    anchor_cache: Dict[str, Optional[np.ndarray]] = {}
    rows: List[Dict] = []

    # Check if we have subdirectories or direct files
    subdirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    direct_files = sorted([p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() in ALT_EXTS])
    
    if subdirs:
        print(f"[INFO] total_subdirs={len(subdirs)}")
        items_to_process = [(sub, sub.name, sub) for sub in subdirs]
    elif direct_files:
        print(f"[INFO] total_files={len(direct_files)}")
        items_to_process = [(f, f.name, root_dir) for f in direct_files]
    else:
        print(f"[INFO] No subdirs or direct image files found")
        items_to_process = []

    for item, item_name, target_dir in items_to_process:
        m = ID_ALPHA_RE.match(item_name)
        if not m:
            # 파일/폴더명에서 id5를 추출할 수 없으면 스킵
            continue

        id5 = m.group(1)

        row = {
            "folder": item_name,
            "id5": id5,
            "anchor_path": "",
            "target_path": "",
            "arcface": np.nan,
            "notes": ""
        }

        # 타겟 이미지 결정
        if item.is_file():
            # Direct file case
            target_path = item
        else:
            # Subdirectory case
            target_path = resolve_target_image(target_dir, args.target_name if args.target_name else None)
        
        if not target_path:
            row["notes"] = "target_missing"
            rows.append(row); continue
        row["target_path"] = str(target_path)

        # 앵커 이미지 결정
        anchor_path = resolve_ffhq_anchor(ffhq_root, id5, args.bucket_size)
        if not anchor_path:
            row["notes"] = "anchor_missing"
            rows.append(row); continue
        row["anchor_path"] = str(anchor_path)

        # 앵커 임베딩 캐시
        key = str(anchor_path)
        if key not in anchor_cache:
            anchor_cache[key] = load_anchor_embedding(app, anchor_path)
        anchor = anchor_cache[key]
        if anchor is None:
            row["notes"] = "anchor_no_face"
            rows.append(row); continue

        # 유사도
        sim = best_sim_against_anchor(app, target_path, anchor)
        if sim is None:
            row["notes"] = "gen_no_face"
        else:
            row["arcface"] = sim
        rows.append(row)

    # CSV 저장
    df = pd.DataFrame(rows)
    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"\n[DONE] saved: {out.resolve()}")

    # 통계
    if not df.empty and df["arcface"].notna().any():
        s = df["arcface"].dropna()
        print(f"[STAT] count={len(s)}, mean={s.mean():.6f}, min={s.min():.6f}, max={s.max():.6f}")
    else:
        print("[STAT] 유효 ArcFace 값이 없습니다.")

if __name__ == "__main__":
    main()