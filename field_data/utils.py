import os
import math
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def seed_everything(seed: int = 123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def robust_v(img: np.ndarray, q: float = 99.0) -> float:
    return float(np.percentile(np.abs(img), q) + 1e-8)

def compute_psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> float:
    mse = F.mse_loss(x, y).item()
    if mse == 0.0:
        return float("inf")
    peak = data_range / 2.0
    return 10.0 * math.log10((peak**2) / mse)

def save_quad(clean, mask, masked, recon, out_png, psnr=None, title_suffix=""):
    plt.figure(figsize=(14, 4))

    def show(img, title, vclip=99.0):
        v = robust_v(img, vclip)
        plt.imshow(img, cmap="seismic", aspect="auto", vmin=-v, vmax=+v)
        plt.title(title)
        plt.colorbar()
        plt.xlabel("Trace")
        plt.ylabel("Time")

    plt.subplot(1, 4, 1)
    show(clean, "Clean" + title_suffix)

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap="gray", aspect="auto", vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Mask")

    plt.subplot(1, 4, 3)
    show(masked, "Masked")

    plt.subplot(1, 4, 4)
    if psnr is None:
        show(recon, "Reconstructed")
    else:
        show(recon, f"Recon | PSNR={psnr:.2f} dB")

    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
