# custom_infer_fixed_v3.py  ─────────────────────────────────────────────
from pathlib import Path
import torch, cv2, numpy as np
from PIL import Image, ImageDraw
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from controlnet_aux import MidasDetector
from insightface.app import FaceAnalysis
from ip_adapter import IPAdapterXL
from pipeline_stable_diffusion_xl_instantid_full import draw_kps
# ──────────────────────────────────────────────────────────────────────
PROMPT   = "a baby"
NEG      = "(lowres, bad quality, watermark)"

FACE_IMG = Path("/data2/jeesoo/FFHQ/00000/00000.png")
POSE_IMG = Path("/data2/jiyoon/InstantID/test/pose/p2.jpeg")
STYLE_IMG= Path("/data2/jiyoon/StyleID/test/style/s1.png")

CN_KPS   = "/data2/jiyoon/InstantID/checkpoints/ControlNetModel"
CN_DEPTH = "diffusers/controlnet-depth-sdxl-1.0-small"

STYLE_ENC= "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
BASE_SDXL= "stabilityai/stable-diffusion-xl-base-1.0"

COND_SCALE_KPS, COND_SCALE_DEPTH = 1.2, 0.7
STYLE_SCALE, GUIDE_SCALE = 1.0, 6.0
STEPS, SEED = 30, 42

DEVICE, DTYPE = "cuda:2", torch.float16
OUTDIR = Path("/data2/jiyoon/custom/5"); OUTDIR.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────
def load_rgb(p): return Image.open(p).convert("RGB")

def to_sdxl_res(im, base=64, short=1024, long=1280):
    w,h = im.size
    r = short/min(w,h); w,h = int(w*r), int(h*r)
    r = long /max(w,h); w,h = int(w*r), int(h*r)
    return im.resize(((w//base)*base, (h//base)*base), Image.BILINEAR)

def depth_to_rgb(arr:np.ndarray)->Image.Image:
    arr = (arr-arr.min())/(arr.max()-arr.min()+1e-8)
    g   = (arr*255).astype("uint8"); return Image.fromarray(np.repeat(g[...,None],3,-1))

def pil2tensor(pil:Image.Image)->torch.Tensor:
    t = torch.from_numpy(np.float32(pil)/255.).permute(2,0,1).unsqueeze(0)
    return t.to(DEVICE, DTYPE)

def make_openpose_map(size, kps):
    canvas = Image.new("RGB", size, "white"); d = ImageDraw.Draw(canvas)
    pts = [(int(x),int(y)) for x,y in kps]
    for p in pts: d.ellipse([p[0]-3,p[1]-3,p[0]+3,p[1]+3], fill="black")
    pairs=[(0,1),(0,2),(0,3),(0,4)]
    for i,j in pairs: d.line([pts[i],pts[j]], width=4, fill="black")
    return canvas
# ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True                          # 메모리 절약

    # detectors
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    face  = FaceAnalysis("antelopev2", root="/data2/jiyoon/InstantID",
                         providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    face.prepare(ctx_id=2, det_size=(640,640))

    face_im  = to_sdxl_res(load_rgb(FACE_IMG))
    pose_im  = to_sdxl_res(load_rgb(POSE_IMG))
    style_im = load_rgb(STYLE_IMG)

    # keypoints map
    info = max(face.get(cv2.cvtColor(np.array(face_im),cv2.COLOR_RGB2BGR)),
               key=lambda d:(d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]))
    kps_pil = make_openpose_map(pose_im.size, info["kps"])
    kps_pil.save(OUTDIR/"kps.png")

    # depth map
    d = midas(pose_im)
    depth_pil = d if isinstance(d, Image.Image) else depth_to_rgb(d)
    depth_pil = depth_pil.resize(pose_im.size, Image.BILINEAR)
    depth_pil.save(OUTDIR/"depth.png")

    # fp16 tensors
    kps_t, depth_t = pil2tensor(kps_pil), pil2tensor(depth_pil)

    # ControlNet pipeline
    cn_kps   = ControlNetModel.from_pretrained(CN_KPS,   torch_dtype=DTYPE).to(DEVICE)
    cn_depth = ControlNetModel.from_pretrained(CN_DEPTH, torch_dtype=DTYPE).to(DEVICE)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_SDXL,
        controlnet = MultiControlNetModel([cn_kps, cn_depth]),
        torch_dtype=DTYPE, add_watermarker=False
    ).to(DEVICE)

    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()                       # 추가 메모리 세이브
    pipe.enable_xformers_memory_efficient_attention()

    # IP-Adapter (style only)
    ip = IPAdapterXL(pipe, image_encoder_path=STYLE_ENC, ip_ckpt=STYLE_IP,
                     device=DEVICE, target_blocks=["up_blocks.0.attentions.1"])

    img = ip.generate(
        pil_image = style_im,
        prompt = PROMPT, negative_prompt = NEG,
        scale = STYLE_SCALE, guidance_scale = GUIDE_SCALE,
        num_inference_steps = STEPS, seed = SEED,
        image = [kps_t, depth_t],
        controlnet_conditioning_scale = [COND_SCALE_KPS, COND_SCALE_DEPTH],
    )[0]

    out = OUTDIR/"result.png"; img.save(out); print("✅ saved →", out)

if __name__ == "__main__":
    main()
