import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from .utils import compute_psnr
from .config import DEVICE
from .sampler import ddim_sample

@torch.no_grad()
def evaluate_now(model, diffusion, val_loader,
                 steps_final=200, guidance_scale=1.6,
                 repaint_R=0, repaint_J=0, dc_lambda=0.3,
                 max_batches=1, save_dir="eval_out", prefix="eval"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    for bi, (inp, tgt) in enumerate(val_loader):
        if bi >= max_batches:
            break

        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        y_obs = inp[:, 0:1]
        mask = inp[:, 1:2]

        x_rec = ddim_sample(
            model, diffusion, y_obs, mask, tgt.shape,
            steps=steps_final, eta=0.0, guidance_scale=guidance_scale,
            repaint_R=repaint_R, repaint_J=repaint_J, dc_lambda=dc_lambda
        ).clamp(-1, 1)

        psnr = compute_psnr(tgt[:1], x_rec[:1])

        def show(img, title, vclip=99.0):
            v = np.percentile(np.abs(img), vclip) + 1e-8
            plt.imshow(img, cmap="seismic", aspect="auto", vmin=-v, vmax=+v)
            plt.title(title)
            plt.colorbar()
            plt.xlabel("Trace")
            plt.ylabel("Time")

        clean = tgt[0, 0].cpu().numpy()
        M = mask[0, 0].cpu().numpy()
        masked = y_obs[0, 0].cpu().numpy()
        recon = x_rec[0, 0].cpu().numpy()

        plt.figure(figsize=(14, 4))
        plt.subplot(1, 4, 1); show(clean, "Clean")
        plt.subplot(1, 4, 2); plt.imshow(M, cmap="gray", aspect="auto", vmin=0, vmax=1); plt.colorbar(); plt.title("Mask")
        plt.subplot(1, 4, 3); show(masked, "Masked")
        plt.subplot(1, 4, 4); show(recon, f"Final Recon | PSNR={psnr:.2f} dB")
        plt.tight_layout()

        out_png = os.path.join(save_dir, f"{prefix}_b{bi}.png")
        plt.savefig(out_png, dpi=140)
        plt.close()

        print(f"[EVAL] batch={bi} | PSNR={psnr:.2f} dB | saved: {out_png}")
