# eval_migan_metrics.py
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def load_rgb_255(path: Path) -> np.ndarray:
    """Load image as RGB float32 in [0,255], shape (H,W,3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def mse_pixel_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Standard MSE (pixel mean O, channel mean O) on 0–255 scale."""
    diff = a - b
    return float(np.mean(diff * diff))


def psnr_from_mse(mse_val: float, data_range: float = 255.0) -> float:
    if mse_val <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse_val))


def ssim_rgb(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    return float(ssim(a, b, data_range=data_range, channel_axis=-1))


def main():
    orig_dir  = Path(r"/mnt/d/hyebin/AOT-GAN-for-Inpainting/data/slp_256/slp_test")
    recon_dir = Path(r"/mnt/d/Hyebin/AOT-GAN-for-Inpainting/aot-gan_intermediate_recon_results/without_finetune_random_rec_recon")

    # orig_dir  = Path(r"/mnt/d/hyebin/AOT-GAN-for-Inpainting/data/ewha")
    # recon_dir = Path(r"/mnt/d/hyebin/AOT-GAN-for-Inpainting/aot-gan_recon_metrics/ewha/random_rec_recon")

    recon_paths = sorted(recon_dir.glob("*.png"))
    if not recon_paths:
        print(f"No recon png found in: {recon_dir}")
        return

    mse_list, psnr_list, ssim_list = [], [], []
    used = 0
    missing_pairs = []

    for recon_path in recon_paths:
        orig_path = orig_dir / recon_path.name  # same filename (e.g., ceh_001.png)

        if not orig_path.exists():
            missing_pairs.append((recon_path.name, str(orig_path)))
            continue

        a = load_rgb_255(orig_path)   # orig
        b = load_rgb_255(recon_path)  # recon

        if a.shape != b.shape:
            b_img = Image.open(recon_path).convert("RGB").resize(
                (a.shape[1], a.shape[0]), Image.BILINEAR
            )
            b = np.asarray(b_img, dtype=np.float32)

        mse_val = mse_pixel_mean(a, b)
        psnr_val = psnr_from_mse(mse_val, data_range=255.0)
        ssim_val = ssim_rgb(a, b, data_range=255.0)

        mse_list.append(mse_val)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        used += 1

    if used == 0:
        print("No valid (orig, recon) pairs found.")
        if missing_pairs:
            print(f"Missing orig for {len(missing_pairs)} recon files (examples): {missing_pairs[:5]}")
        return

    print(f"[Recon folder] {recon_dir}")
    print(f"[Orig  folder] {orig_dir}")
    print(f"[Samples used] {used}")

    if missing_pairs:
        print(f"[Warning] Missing orig pairs: {len(missing_pairs)} (showing up to 5)")
        for name, expected in missing_pairs[:5]:
            print("  -", name, "-> expected", expected)

    print("\n[Averages over samples]")
    print(f"MSE  (0–255 scale): {float(np.mean(mse_list)):.4f}")
    print(f"PSNR (dB):         {float(np.mean(psnr_list)):.4f}")
    print(f"SSIM:              {float(np.mean(ssim_list)):.4f}")


if __name__ == "__main__":
    main()
