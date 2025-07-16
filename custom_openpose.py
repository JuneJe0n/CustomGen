#custom_openpose.py
"""
예시
python run_sdxl_multicn.py --ctrl kps
python run_sdxl_multicn.py --ctrl both --style
"""
import argparse, math
from pathlib import Path

import torch, cv2, numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import MidasDetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL

# ──────────────────────────────────────────────────────────────────────
PROMPT, NEG = "a baby", "(lowres, bad quality, watermark)"
FACE_IMG  = Path("/data2/jeesoo/FFHQ/00000/00000.png")            # 얼굴 키포인트용
POSE_IMG  = Path("/data2/jiyoon/InstantID/test/pose/p2.jpeg")     # Depth 추출용
STYLE_IMG = Path("/data2/jiyoon/StyleID/test/style/s1.png")

CN_KPS   = "/data2/jiyoon/custom/ckpts/controlnet-openpose-sdxl-1.0"
CN_DEPTH = "diffusers/controlnet-depth-sdxl-1.0-small"
BASE_SDXL= "stabilityai/stable-diffusion-xl-base-1.0"

STYLE_ENC= "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

COND_SCALE_KPS, COND_SCALE_DEPTH = 1.2, 0.7
STYLE_SCALE, GUIDE_SCALE = 1.0, 6.0
STEPS, SEED = 30, 42

GPU_ID = 5
DEVICE, DTYPE = f"cuda:{GPU_ID}", torch.float16
OUTDIR = Path("/data2/jiyoon/custom/results/mode/4/multi"); OUTDIR.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────
def load_rgb(p): return Image.open(p).convert("RGB")
def to_sdxl_res(im, base=64, short=1024, long=1280):
    w,h = im.size
    r = short/min(w,h); w,h = int(w*r), int(h*r)
    r = long /max(w,h); w,h = int(w*r), int(h*r)
    return im.resize(((w//base)*base,(h//base)*base), Image.BILINEAR)
def depth_to_rgb(arr):
    if isinstance(arr, Image.Image): return arr.convert("RGB")
    if torch.is_tensor(arr): arr = arr.squeeze().cpu().numpy()
    arr = arr.astype("float32")
    arr = (arr-arr.min())/(arr.max()-arr.min()+1e-8)
    g = (arr*255).astype("uint8")
    return Image.fromarray(np.repeat(g[...,None],3,-1))
def pil2tensor(pil): return torch.from_numpy(np.float32(pil)/255.).permute(2,0,1).unsqueeze(0).to(DEVICE,DTYPE)

def draw_kps(img,kps):
    import cv2
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    limb=np.array([[0,2],[1,2],[3,2],[4,2]])
    out=np.zeros([img.height,img.width,3],dtype=np.uint8)
    for p in limb:
        c=colors[p[0]]; x=kps[p][:,0]; y=kps[p][:,1]
        L=np.hypot(x[0]-x[1],y[0]-y[1]); ang=math.degrees(math.atan2(y[0]-y[1],x[0]-x[1]))
        poly=cv2.ellipse2Poly((int(np.mean(x)),int(np.mean(y))),(int(L/2),4),int(ang),0,360,1)
        out=cv2.fillConvexPoly(out.copy(),poly,c)
    out=(out*0.6).astype(np.uint8)
    for i,(x,y) in enumerate(kps):
        out=cv2.circle(out.copy(),(int(x),int(y)),10,colors[i%5],-1)
    return Image.fromarray(out)
# ──────────────────────────────────────────────────────────────────────
def main(ctrl,use_style):
    torch.manual_seed(SEED); torch.backends.cuda.matmul.allow_tf32=True

    face = FaceAnalysis(
        name="antelopev2",
        root="/data2/jiyoon/InstantID",
        providers=[('CUDAExecutionProvider', {'device_id': GPU_ID}),
                   'CPUExecutionProvider']
    )
    face.prepare(ctx_id=GPU_ID, det_size=(640,640))
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)

    # 0. 이미지 로드
    face_im = to_sdxl_res(load_rgb(FACE_IMG))    # 얼굴 키포인트 추출용
    pose_im = to_sdxl_res(load_rgb(POSE_IMG))    # Depth 맵용
    style_im= load_rgb(STYLE_IMG)

    # 1. 얼굴 키포인트 (FACE_IMG 기준)
    f_info = max(
        face.get(cv2.cvtColor(np.array(face_im), cv2.COLOR_RGB2BGR)),
        key=lambda d:(d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1])
    )
    kps_pil = draw_kps(face_im, f_info["kps"]).resize(pose_im.size, Image.BILINEAR)
    kps_pil.save(OUTDIR/"kps.png")

    # 2. Depth 맵 (POSE_IMG 기준)
    depth_pil = depth_to_rgb(midas(pose_im)).resize(pose_im.size, Image.BILINEAR)
    depth_pil.save(OUTDIR/"depth.png")

    # tensors
    kps_t, depth_t = pil2tensor(kps_pil), pil2tensor(depth_pil)
    dummy_t = torch.ones_like(kps_t)

    # 3. Multi-ControlNet 파이프라인
    cn_kps   = ControlNetModel.from_pretrained(CN_KPS,   torch_dtype=DTYPE).to(DEVICE)
    cn_depth = ControlNetModel.from_pretrained(CN_DEPTH, torch_dtype=DTYPE).to(DEVICE)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_SDXL, controlnet=MultiControlNetModel([cn_kps, cn_depth]),
        torch_dtype=DTYPE, add_watermarker=False
    ).to(DEVICE)
    pipe.enable_vae_tiling(); pipe.enable_vae_slicing(); pipe.enable_xformers_memory_efficient_attention()

    img_list   = [kps_t if ctrl in ("kps","both") else dummy_t,
                  depth_t if ctrl in ("depth","both") else dummy_t]
    scale_list = [COND_SCALE_KPS if ctrl in ("kps","both") else 0.0,
                  COND_SCALE_DEPTH if ctrl in ("depth","both") else 0.0]

    # 4. 생성
    if use_style:
        ip   = IPAdapterXL(pipe, image_encoder_path=STYLE_ENC, ip_ckpt=STYLE_IP,
                           device=DEVICE, target_blocks=["up_blocks.0.attentions.1"])
        out  = ip.generate(
            pil_image=style_im, prompt=PROMPT, negative_prompt=NEG,
            scale=STYLE_SCALE, guidance_scale=GUIDE_SCALE,
            num_inference_steps=STEPS, seed=SEED,
            image=img_list, controlnet_conditioning_scale=scale_list,
        )[0]
    else:
        out  = pipe(
            prompt=PROMPT, negative_prompt=NEG, num_inference_steps=STEPS,
            image=img_list, controlnet_conditioning_scale=scale_list
        ).images[0]

    out_path = OUTDIR/f"result_ctrl-{ctrl}_style-{use_style}.png"
    out.save(out_path); print("✅ saved →", out_path)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ctrl",  choices=["none","kps","depth","both"], required=True)
    p.add_argument("--style", action="store_true")
    args = p.parse_args()
    main(args.ctrl, args.style)
