import os
import time
from glob import glob

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor

import importlib
from utils.option import args  # args.model, args.pre_train 사용


# ------------------ paths ------------------
# SLP dataset
IMG_DIR = "/mnt/d/Hyebin/AOT-GAN-for-Inpainting/aot-gan_recon_results/slp_orig"
MASK_DIR = "/mnt/d/Hyebin/AOT-GAN-for-Inpainting/aot-gan_recon_results/slp_mask/random_rec"
OUT_DIR = "/mnt/d/Hyebin/AOT-GAN-for-Inpainting/aot-gan_time_recon_results/500_000/slp/random_rec"
PRE_TRAIN = "/mnt/d/hyebin/AOT-GAN-for-Inpainting/experiments/slp/aotgan_slp_train_random_rec256/G0500000.pt"


os.makedirs(OUT_DIR, exist_ok=True)


def find_mask_path(basename: str) -> str | None:
    """mask는 보통 png로 저장돼 있으니 우선 png, 그 외 확장자도 fallback"""
    candidates = [
        os.path.join(MASK_DIR, f"{basename}.png"),
        os.path.join(MASK_DIR, f"{basename}.jpg"),
        os.path.join(MASK_DIR, f"{basename}.jpeg"),
        os.path.join(MASK_DIR, f"{basename}.bmp"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def postprocess_to_bgr_uint8(image_rgb_chw: torch.Tensor) -> np.ndarray:
    """tensor RGB CHW in [-1,1] -> uint8 BGR HWC"""
    x = torch.clamp(image_rgb_chw, -1.0, 1.0)
    x = (x + 1.0) / 2.0 * 255.0
    x = x.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)  # RGB HWC
    return rgb_to_bgr(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ------------------ load model ------------------
    net = importlib.import_module("model." + args.model)
    model = net.InpaintGenerator(args).to(device)
    model.load_state_dict(torch.load(PRE_TRAIN, map_location=device))
    model.eval()

    # ------------------ gather images ------------------
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    img_paths = []
    for e in exts:
        img_paths.extend(glob(os.path.join(IMG_DIR, e)))
    img_paths.sort()
    print(f"[INFO] images = {len(img_paths)}")

    ok, skipped = 0, 0

    # ===== 모델 로딩 이후, 이미지 생성 시간만 측정 시작 =====
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for p in img_paths:
            base = os.path.splitext(os.path.basename(p))[0]
            mask_path = find_mask_path(base)
            if mask_path is None:
                print(f"[SKIP] mask not found for: {base}")
                skipped += 1
                continue

            # ---- read image ----
            img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"[SKIP] failed to read image: {p}")
                skipped += 1
                continue

            # ---- read mask (grayscale) ----
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                print(f"[SKIP] failed to read mask: {mask_path}")
                skipped += 1
                continue

            # ---- assert / resize consistency ----
            Hm, Wm = mask_gray.shape[:2]
            Hi, Wi = img_bgr.shape[:2]
            if (Hi, Wi) != (Hm, Wm):
                img_bgr = cv2.resize(img_bgr, (Wm, Hm), interpolation=cv2.INTER_AREA)

            # ---- tensorize (trainer style) ----
            img_rgb = bgr_to_rgb(img_bgr)
            img_t = (ToTensor()(img_rgb) * 2.0 - 1.0).unsqueeze(0).to(device)  # [1,3,H,W], [-1,1]

            # mask: 0/255 -> 0/1 float, [1,1,H,W]
            mask_t = ToTensor()(mask_gray).unsqueeze(0).to(device)  # [1,1,H,W], [0,1]
            mask_t = (mask_t > 0.5).float()

            # ---- inpaint ----
            masked_t = img_t * (1.0 - mask_t) + mask_t
            pred_t = model(masked_t, mask_t)  # [1,3,H,W]
            comp_t = pred_t * mask_t + img_t * (1.0 - mask_t)

            # ---- save ----
            comp_bgr = postprocess_to_bgr_uint8(comp_t[0])  # uint8 BGR
            out_path = os.path.join(OUT_DIR, f"{base}.png")
            ok_write = cv2.imwrite(out_path, comp_bgr)
            if not ok_write:
                print(f"[SKIP] failed to write: {out_path}")
                skipped += 1
                continue

            ok += 1
            if ok % 50 == 0:
                print(f"[PROGRESS] ok={ok}, skipped={skipped}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    # ===== 시간 측정 종료 =====

    print(f"[DONE] ok={ok}, skipped={skipped}")
    print(f"[DONE] saved to: {OUT_DIR}")
    print(f"elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()