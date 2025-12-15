import os, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    USE_DIR_MODE, SINGLE_FILE_PATH, SEGY_DIR, SEGY_GLOB,
    CACHE_DIR, EXPECTED_TRACES, GATHER_LIMIT, TRAIN_SPLIT,
    CKPT_DIFFUSION, CKPT_CNN, CKPT_TDAE,
    OUT_ROOT,
    PATCH_HW, STRIDE_HW, DROP_RATIO, MASK_W_MIN_MAX, NUM_EXAMPLES,
    T_STEPS, STEPS_SLOW, GUIDE_SLOW, STEPS_FAST, GUIDE_FAST, DC_LAMBDA,
    DEVICE, AMP_CTX
)

from .utils import (
    ensure_dir, compute_psnr_np, compute_ssim_global_np,
    save_figure_2x4, save_diff_figure, save_fk_figure, save_amp_spectrum
)

from .segy_cache import cache_gathers
from .dataset import build_random_vertical_trace_mask, compute_patch_positions, MaskedPatchDataset
from .diffusion_core import Diffusion, diffusion_reconstruct
from .models_diffusion import ResUNet
from .baselines import SimpleUNet, ConvAE, hslr_reconstruct


def run_compare():
    print("[MODE] compare")
    ensure_dir(OUT_ROOT)
    ensure_dir(CACHE_DIR)

    # cache
    npy_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npy")))
    if len(npy_files) == 0:
        print("[INFO] building cache from SEGY ...")
        npy_files = cache_gathers(USE_DIR_MODE, SINGLE_FILE_PATH, SEGY_DIR, SEGY_GLOB,
                                  CACHE_DIR, EXPECTED_TRACES, GATHER_LIMIT)

    n = len(npy_files)
    Ni = int(TRAIN_SPLIT * n)
    train_files = npy_files[:Ni]
    val_files = npy_files[Ni:] if Ni < n else npy_files[max(0, n - NUM_EXAMPLES):]
    print(f"[SPLIT] train={len(train_files)}, val={len(val_files)}")

    # load diffusion (ours)
    diff_model = ResUNet(in_ch=3, base_ch=48, use_axial=True, ax_heads=2).to(DEVICE)
    diffusion = Diffusion(T=T_STEPS, device=DEVICE)

    sd = torch.load(CKPT_DIFFUSION, map_location="cpu")
    sd = sd["model"] if "model" in sd else sd
    diff_model.load_state_dict(sd, strict=False)
    diff_model.eval()
    print("[CKPT] diffusion loaded.")

    # CNN
    cnn = SimpleUNet(in_ch=2, base=32).to(DEVICE)
    if os.path.exists(CKPT_CNN):
        cnn.load_state_dict(torch.load(CKPT_CNN, map_location="cpu"))
        print("[CKPT] CNN loaded.")
    else:
        print("[WARN] CNN checkpoint not found, using untrained weights!")
    cnn.eval()

    # TDAE
    tdae = ConvAE(in_ch=2, base=32).to(DEVICE)
    if os.path.exists(CKPT_TDAE):
        tdae.load_state_dict(torch.load(CKPT_TDAE, map_location="cpu"))
        print("[CKPT] TDAE loaded.")
    else:
        print("[WARN] TDAE checkpoint not found, using untrained weights!")
    tdae.eval()

    # metrics file
    metrics_path = os.path.join(OUT_ROOT, "metrics_advanced.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("example,method,psnr_db,mse,ssim\n")

        rng = np.random.RandomState(123)

        # pick patches from different offsets
        example_specs = []
        H, W = PATCH_HW

        for fp in val_files:
            g_tmp = np.load(fp).astype(np.float32)
            ns, ntr = g_tmp.shape
            positions = compute_patch_positions(ns, ntr, PATCH_HW, STRIDE_HW)
            for (t0, x0) in positions:
                example_specs.append((fp, t0, x0))
                if len(example_specs) >= NUM_EXAMPLES:
                    break
            if len(example_specs) >= NUM_EXAMPLES:
                break

        print(f"[INFO] using {len(example_specs)} patches at various offsets for comparison.")

        for ex_idx, (fp, t0, x0) in enumerate(example_specs):
            g = np.load(fp).astype(np.float32)
            ns, ntr = g.shape
            if t0 + H > ns or x0 + W > ntr:
                print("[SKIP] invalid patch window", fp, t0, x0)
                continue

            patch = g[t0:t0+H, x0:x0+W]

            m1d = build_random_vertical_trace_mask(W, DROP_RATIO, MASK_W_MIN_MAX[0], MASK_W_MIN_MAX[1], rng)
            M2d = np.repeat(m1d[None, :], H, axis=0).astype(np.float32)
            masked = patch.copy()
            masked[:, m1d == 0] = 0.0

            clean_t = torch.from_numpy(patch[None, None, ...]).to(DEVICE)
            y_obs = torch.from_numpy(masked[None, None, ...]).to(DEVICE)
            M_t = torch.from_numpy(M2d[None, None, ...]).to(DEVICE)

            # ours slow
            with torch.no_grad(), AMP_CTX:
                x_rec = diffusion_reconstruct(diff_model, diffusion, y_obs, M_t, clean_t.shape,
                                              steps=STEPS_SLOW, guidance_scale=GUIDE_SLOW, dc_lambda=DC_LAMBDA)
            rec_ours = x_rec[0, 0].cpu().numpy()

            # fast diffusion (fdm)
            with torch.no_grad(), AMP_CTX:
                x_rec_fast = diffusion_reconstruct(diff_model, diffusion, y_obs, M_t, clean_t.shape,
                                                   steps=STEPS_FAST, guidance_scale=GUIDE_FAST, dc_lambda=DC_LAMBDA)
            rec_fdm = x_rec_fast[0, 0].cpu().numpy()

            # cnn
            with torch.no_grad(), AMP_CTX:
                pred_cnn = cnn(y_obs, M_t)
            rec_cnn = pred_cnn[0, 0].cpu().numpy()

            # tdae twice
            with torch.no_grad(), AMP_CTX:
                z1 = tdae(y_obs, M_t)
                z1_mask = z1 * M_t + y_obs * (1 - M_t)
                z2 = tdae(z1_mask, M_t)
            rec_tdae = z2[0, 0].cpu().numpy()

            # hslr
            rec_hslr = hslr_reconstruct(patch, M2d, win_t=20, rank=5)

            methods = {"ours": rec_ours, "cnn": rec_cnn, "tdae": rec_tdae, "fdm": rec_fdm, "hslr": rec_hslr}

            psnrs = {}
            for name, rec in methods.items():
                mse = float(np.mean((patch - rec) ** 2))
                psnr = compute_psnr_np(patch, rec, data_range=2.0)
                ssim = compute_ssim_global_np(patch, rec, data_range=2.0)
                psnrs[name] = psnr
                f.write(f"{ex_idx},{name},{psnr:.4f},{mse:.8e},{ssim:.4f}\n")
            f.flush()

            out_png = os.path.join(OUT_ROOT, f"example_{ex_idx:02d}_adv_offsets.png")
            save_figure_2x4(patch, M2d, masked, rec_ours, rec_cnn, rec_tdae, rec_fdm, rec_hslr, psnrs, out_png)

            out_fk = os.path.join(OUT_ROOT, f"example_{ex_idx:02d}_adv_offsets_fk.png")
            save_fk_figure(patch, methods, out_fk)

            out_spec = os.path.join(OUT_ROOT, f"example_{ex_idx:02d}_adv_offsets_spectrum.png")
            save_amp_spectrum(patch, methods, out_spec)

            out_diff = os.path.join(OUT_ROOT, f"example_{ex_idx:02d}_adv_offsets_diff.png")
            save_diff_figure(patch, rec_ours, rec_cnn, rec_tdae, rec_fdm, rec_hslr, out_diff)

    print("[DONE] metrics →", metrics_path)
