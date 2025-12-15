import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (
    MODE, USE_DIR_MODE, SEGY_DIR, SEGY_GLOB, SINGLE_FILE_PATH,
    CACHE_DIR, EXPECTED_TRACES, GATHER_LIMIT, TRAIN_SPLIT,
    OUT_ROOT, RATIO_DROP, PATCH_HW, STRIDE_HW,
    EPOCHS, EARLY_EPOCHS, GUIDE_EARLY, GUIDE_LATE, REPAINT_EARLY, REPAINT_LATE,
    BATCH_SIZE, ACCUM_STEPS, LR, PATIENCE, GRAD_CLIP, SEED,
    LAMBDA_SPEC, TV_LAMBDA, W_MISS, W_OBS, P_UNCOND,
    DEVICE, AMP, AMP_CTX, SCALER,
    EXPORT_SAMPLES_N, STEPS_FINAL, GUIDE_EVAL, REPAINT_EVAL, DC_LAMBDA, FULLVAL_STEPS,
    T_STEPS, USE_AXIAL, AX_HEADS, BASE_CH
)

from .utils import ensure_dir, seed_everything, compute_psnr, save_quad
from .segy_io import cache_gathers
from .dataset import NpyPatchDataset
from .diffusion import Diffusion
from .model import ResUNet
from .losses import spectral_l1, total_variation
from .sampler import ddim_sample


def build_model_and_diffusion():
    model = ResUNet(
        in_ch=3,
        base_ch=BASE_CH,
        time_dim=64,
        time_emb_dim=256,
        use_axial=USE_AXIAL,
        ax_heads=AX_HEADS
    ).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    diff = Diffusion(T=T_STEPS, device=DEVICE)
    return model, diff


def train_with_loaders(tr_loader, va_loader, save_dir):
    ensure_dir(save_dir)
    ckpt_path = os.path.join(save_dir, "field_inpaint_20pct.pth")

    model, diff = build_model_and_diffusion()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)

    best = float("inf")
    patience_ctr = 0

    for ep in range(1, EPOCHS + 1):
        if ep <= EARLY_EPOCHS:
            phase = "early"
            g_scale = GUIDE_EARLY
            R, J = REPAINT_EARLY
        else:
            phase = "late"
            g_scale = GUIDE_LATE
            R, J = REPAINT_LATE

        # --- Train ---
        model.train()
        opt.zero_grad(set_to_none=True)
        run = 0.0

        for i, (inp, tgt) in enumerate(tqdm(tr_loader, desc=f"train {ep}/{EPOCHS}", leave=True)):
            inp = inp.to(DEVICE).to(memory_format=torch.channels_last)
            tgt = tgt.to(DEVICE).to(memory_format=torch.channels_last)

            B = tgt.size(0)
            y_obs = inp[:, 0:1]
            mask = inp[:, 1:2]

            with torch.no_grad():
                drop = (torch.rand(B, device=DEVICE) < P_UNCOND).float().view(-1, 1, 1, 1)
                y_eff = y_obs * (1.0 - drop)
                m_eff = mask * (1.0 - drop)

            t = torch.randint(0, diff.T, (B,), device=DEVICE).long()
            noise = torch.randn_like(tgt)

            with AMP_CTX:
                x_t = diff.q_sample(tgt, t, noise)
                x_in = torch.cat([x_t, y_eff, m_eff], 1)

                v_tgt = diff.v_from_eps_x0(t, noise, tgt)
                v_pr = model(x_in, t)
                x0_pr = diff.x0_from_v_xt(t, v_pr, x_t).clamp(-1, 1)

                miss = (1.0 - mask)
                l_pred = F.mse_loss(v_pr, v_tgt)
                l_rec = F.l1_loss(x0_pr * miss, tgt * miss) * W_MISS + F.l1_loss(x0_pr * mask, tgt * mask) * W_OBS
                l_spc = spectral_l1(x0_pr, tgt) * LAMBDA_SPEC
                l_tv = total_variation(x0_pr) * TV_LAMBDA
                loss = (l_pred + l_rec + l_spc + l_tv) / float(ACCUM_STEPS)

            SCALER.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP:
                    SCALER.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                SCALER.step(opt)
                SCALER.update()
                opt.zero_grad(set_to_none=True)

            run += float((l_pred + l_rec + l_spc + l_tv).item())

        tr_loss = run / max(1, len(tr_loader))
        print(f"Epoch {ep:02d}/{EPOCHS} | Train {tr_loss:.5f} | Phase={phase}")

        # --- Val ---
        model.eval()
        vloss = 0.0
        with torch.no_grad(), AMP_CTX:
            for inp, tgt in tqdm(va_loader, desc=f"valid {ep}/{EPOCHS}", leave=False):
                inp = inp.to(DEVICE).to(memory_format=torch.channels_last)
                tgt = tgt.to(DEVICE).to(memory_format=torch.channels_last)

                B = tgt.size(0)
                y_obs = inp[:, 0:1]
                mask = inp[:, 1:2]

                t = torch.randint(0, diff.T, (B,), device=DEVICE).long()
                noise = torch.randn_like(tgt)

                x_t = diff.q_sample(tgt, t, noise)
                x_in = torch.cat([x_t, y_obs, mask], 1)

                v_tgt = diff.v_from_eps_x0(t, noise, tgt)
                v_pr = model(x_in, t)
                x0_pr = diff.x0_from_v_xt(t, v_pr, x_t).clamp(-1, 1)

                miss = (1.0 - mask)
                vloss += (
                    F.mse_loss(v_pr, v_tgt)
                    + F.l1_loss(x0_pr * miss, tgt * miss) * W_MISS
                    + F.l1_loss(x0_pr * mask, tgt * mask) * W_OBS
                    + spectral_l1(x0_pr, tgt) * LAMBDA_SPEC
                    + total_variation(x0_pr) * TV_LAMBDA
                ).item()

        vloss /= max(1, len(va_loader))
        sched.step(vloss)
        print(f"           Val  {vloss:.5f}")

        # --- Save best ---
        if vloss < best:
            best = vloss
            patience_ctr = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": best_state}, ckpt_path)
            print(f"  Saved best -> {ckpt_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("  Early stopping.")
                break

    return ckpt_path


def export_samples(ckpt_path):
    print(f"[INFO] export_samples from {ckpt_path}")

    npy_files = sorted([str(p) for p in Path(CACHE_DIR).glob("*_gid*.npy")])
    n = len(npy_files)
    Ni = int(TRAIN_SPLIT * n)
    val_files = npy_files[Ni:] if Ni < n else npy_files
    if len(val_files) == 0:
        print("[WARN] no val_files found.")
        return

    va_ds = NpyPatchDataset(val_files, patch_hw=PATCH_HW, stride_hw=STRIDE_HW, drop_ratio=RATIO_DROP, seed=456)
    va_loader = DataLoader(
        va_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=False
    )

    model, diff = build_model_and_diffusion()
    sd = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(sd["model"])
    model.eval()

    out_dir = os.path.join(OUT_ROOT, f"ratio_{int(RATIO_DROP*100)}", "eval_export_gpu")
    ensure_dir(out_dir)

    cnt = 0
    for inp, tgt in va_loader:
        if cnt >= EXPORT_SAMPLES_N:
            break

        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        y_obs = inp[:, 0:1]
        mask = inp[:, 1:2]

        with torch.no_grad():
            x_rec = ddim_sample(
                model, diff, y_obs, mask, tgt.shape,
                steps=STEPS_FINAL, eta=0.0, guidance_scale=GUIDE_EVAL,
                repaint_R=REPAINT_EVAL[0], repaint_J=REPAINT_EVAL[1],
                dc_lambda=DC_LAMBDA
            ).clamp(-1, 1)
            psnr = compute_psnr(tgt[:1], x_rec[:1])

        clean = tgt[0, 0].detach().cpu().numpy()
        M = mask[0, 0].detach().cpu().numpy()
        masked = y_obs[0, 0].detach().cpu().numpy()
        recon = x_rec[0, 0].detach().cpu().numpy()

        out_png = os.path.join(out_dir, f"sample_{cnt:02d}_PSNR{psnr:.2f}.png")
        save_quad(clean, M, masked, recon, out_png, psnr=psnr, title_suffix=" (val)")
        print(f"[EXPORT] saved {out_png}")
        cnt += 1

    torch.cuda.empty_cache()
    print("[EXPORT] done.")


def full_val_psnr(ckpt_path):
    print(f"[INFO] full_val_psnr from {ckpt_path}")

    npy_files = sorted([str(p) for p in Path(CACHE_DIR).glob("*_gid*.npy")])
    n = len(npy_files)
    Ni = int(TRAIN_SPLIT * n)
    val_files = npy_files[Ni:] if Ni < n else npy_files
    if len(val_files) == 0:
        print("[WARN] no val_files found.")
        return

    va_ds = NpyPatchDataset(val_files, patch_hw=PATCH_HW, stride_hw=STRIDE_HW, drop_ratio=RATIO_DROP, seed=789)
    va_loader = DataLoader(
        va_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=False
    )

    model, diff = build_model_and_diffusion()
    sd = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(sd["model"])
    model.eval()

    psnrs = []
    for inp, tgt in tqdm(va_loader, desc="full_val_psnr"):
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        y_obs = inp[:, 0:1]
        mask = inp[:, 1:2]

        with torch.no_grad():
            x_rec = ddim_sample(
                model, diff, y_obs, mask, tgt.shape,
                steps=FULLVAL_STEPS, eta=0.0, guidance_scale=GUIDE_EVAL,
                repaint_R=0, repaint_J=0,
                dc_lambda=DC_LAMBDA
            ).clamp(-1, 1)
            ps = compute_psnr(tgt[:1], x_rec[:1])
            psnrs.append(ps)

    psnrs = np.array(psnrs, dtype=np.float32)
    print(f"[FULL VAL] N={len(psnrs)} | PSNR mean={psnrs.mean():.2f} dB | std={psnrs.std():.2f} dB")

    out_dir = os.path.join(OUT_ROOT, f"ratio_{int(RATIO_DROP*100)}")
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "full_val_psnrs.npy"), psnrs)
    with open(os.path.join(out_dir, "full_val_psnr_stats.txt"), "w") as f:
        f.write(f"N={len(psnrs)}\nmean={psnrs.mean():.4f}\nstd={psnrs.std():.4f}\n")

    torch.cuda.empty_cache()
    print("[FULL VAL] saved stats.")


def main():
    seed_everything(SEED)
    ensure_dir(OUT_ROOT)
    ensure_dir(CACHE_DIR)

    print(f"[INFO] Device: {DEVICE.type.upper()}, AMP={AMP}, MODE={MODE}")

    npy_files = sorted([str(p) for p in Path(CACHE_DIR).glob("*.npy")])

    if len(npy_files) == 0 and MODE == "train":
        npy_files = cache_gathers(
            use_dir=USE_DIR_MODE,
            single_path=SINGLE_FILE_PATH,
            dir_path=SEGY_DIR, dir_glob=SEGY_GLOB,
            cache_dir=CACHE_DIR,
            expected_traces=EXPECTED_TRACES,
            limit=GATHER_LIMIT
        )
    else:
        if len(npy_files) == 0:
            npy_files = cache_gathers(
                use_dir=USE_DIR_MODE,
                single_path=SINGLE_FILE_PATH,
                dir_path=SEGY_DIR, dir_glob=SEGY_GLOB,
                cache_dir=CACHE_DIR,
                expected_traces=EXPECTED_TRACES,
                limit=GATHER_LIMIT
            )
        print(f"[INFO] Using existing cache: {len(npy_files)} npy files.")

    if MODE == "train":
        n = len(npy_files)
        Ni = int(TRAIN_SPLIT * n)
        train_files = npy_files[:Ni]
        val_files = npy_files[Ni:]
        print(f"[SPLIT] train_gathers={len(train_files)} | val_gathers={len(val_files)}")

        tr_ds = NpyPatchDataset(train_files, patch_hw=PATCH_HW, stride_hw=STRIDE_HW, drop_ratio=RATIO_DROP, seed=123)
        va_ds = NpyPatchDataset(val_files,   patch_hw=PATCH_HW, stride_hw=STRIDE_HW, drop_ratio=RATIO_DROP, seed=456)

        tr_loader = DataLoader(
            tr_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=False
        )
        va_loader = DataLoader(
            va_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=False
        )

        print(f"[LOADERS] train_patches={len(tr_ds)} | val_patches={len(va_ds)}")

        run_dir = os.path.join(OUT_ROOT, f"ratio_{int(RATIO_DROP*100)}")
        ckpt = train_with_loaders(tr_loader, va_loader, save_dir=run_dir)
        print(f"[DONE] best model: {ckpt}")

    else:
        ckpt_path = os.path.join(OUT_ROOT, f"ratio_{int(RATIO_DROP*100)}", "field_inpaint_20pct.pth")
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] checkpoint not found at {ckpt_path}. Run MODE='train' first.")
            return

        if MODE == "export_samples":
            export_samples(ckpt_path)
        elif MODE == "full_val_psnr":
            full_val_psnr(ckpt_path)
        else:
            print(f"[ERROR] Unknown MODE={MODE}")


if __name__ == "__main__":
    main()
