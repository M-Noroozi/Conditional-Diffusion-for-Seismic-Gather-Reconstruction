import os, glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .config import (
    SEGY_DIR, SEGY_GLOB, EXPECTED_TRACES, FILE_LIMIT,
    MASK_W_MIN_MAX,
    PATCH_HW, STRIDE_HW,
    SEED, EPOCHS, BATCH_SIZE, LR, PATIENCE, GRAD_CLIP, ACCUM_STEPS,
    EARLY_EPOCHS, CFG_EARLY, REPAINT_EARLY, CFG_LATE, REPAINT_LATE,
    T_STEPS, OBJECTIVE,
    STEPS_FINAL, DC_LAMBDA,
    LAMBDA_SPEC, TV_LAMBDA, P_UNCOND, W_MISS, W_OBS,
    RUNS_DIR,
    DEVICE, AMP_CTX, SCALER,
    BASE_CH, AX_HEADS, USE_AXIAL
)

from .utils import seed_everything, ensure_dir
from .dataset import BP2004PatchDataset
from .diffusion import Diffusion
from .model import ResUNet
from .losses import spectral_l1, total_variation
from .eval import evaluate_now


def train_one_ratio(ratio: float, resume: bool = True) -> str:
    """
    Train on one missing-trace ratio and save the best checkpoint.
    Returns checkpoint path.
    """
    seed_everything(SEED)
    ensure_dir(RUNS_DIR)

    files = sorted(glob.glob(os.path.join(SEGY_DIR, SEGY_GLOB)))
    if not files:
        raise FileNotFoundError("No SEGY files found. Please set SEGY_DIR/SEGY_GLOB in config.py.")

    ds = BP2004PatchDataset(
        files,
        patch_hw=PATCH_HW,
        stride_hw=STRIDE_HW,
        drop_ratio=ratio,
        min_width=MASK_W_MIN_MAX[0],
        max_width=MASK_W_MIN_MAX[1],
        expected_traces=EXPECTED_TRACES,
        seed=SEED,
        verbose=False,
        file_limit=FILE_LIMIT
    )

    # split patches (90/10)
    N = len(ds)
    Nv = max(1, int(0.1 * N))
    Ni = N - Nv
    idx = np.arange(N)
    np.random.shuffle(idx)
    tr_idx, va_idx = idx[:Ni], idx[Ni:]

    tr_loader = DataLoader(
        Subset(ds, tr_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    va_loader = DataLoader(
        Subset(ds, va_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # model / diffusion
    model = ResUNet(
        in_ch=3,  # [x_t, y_obs, mask]
        base_ch=BASE_CH,
        time_dim=64,
        time_emb_dim=256,
        use_axial=USE_AXIAL,
        ax_heads=AX_HEADS
    ).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)

    diff = Diffusion(T=T_STEPS, device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)

    ckpt_path = os.path.join(RUNS_DIR, f"inpaint_v2_clean_{int(ratio*100)}pct.pth")

    # resume weights
    if resume and os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(sd["model"])
            print(f"[resume] loaded {ckpt_path} (CPU-mapped)")
        except Exception as e:
            print(f"[resume] failed: {e}")

    print(f"==== Train ratio={int(ratio*100)}% | patches: train={len(tr_idx)}, val={len(va_idx)} ====")

    best = float("inf")
    patience_ctr = 0

    for ep in range(1, EPOCHS + 1):
        # two-phase guidance schedule
        if ep <= EARLY_EPOCHS:
            guidance_scale = CFG_EARLY
            R, J = REPAINT_EARLY
            phase = "early"
        else:
            guidance_scale = CFG_LATE
            R, J = REPAINT_LATE
            phase = "late"

            if ep == EARLY_EPOCHS + 1:
                print("[INFO] Phase switched to LATE — one-shot eval.")
                evaluate_now(
                    model, diff, va_loader,
                    steps_final=STEPS_FINAL,
                    guidance_scale=guidance_scale,
                    repaint_R=R, repaint_J=J,
                    dc_lambda=DC_LAMBDA,
                    max_batches=1,
                    save_dir=os.path.join(RUNS_DIR, "eval_phase_switch"),
                    prefix=f"{int(ratio*100)}pct_ep{ep}"
                )

        # ---- Train
        model.train()
        run = 0.0
        opt.zero_grad(set_to_none=True)

        for i, (inp, tgt) in enumerate(tqdm(tr_loader, desc=f"train {ep}/{EPOCHS}", leave=False)):
            inp = inp.to(DEVICE).to(memory_format=torch.channels_last)
            tgt = tgt.to(DEVICE).to(memory_format=torch.channels_last)

            B = tgt.size(0)
            y_obs = inp[:, 0:1]
            mask = inp[:, 1:2]

            # classifier-free dropout on conditioning
            with torch.no_grad():
                drop = (torch.rand(B, device=DEVICE) < P_UNCOND).float().view(-1, 1, 1, 1)
                y_eff = y_obs * (1.0 - drop)
                m_eff = mask * (1.0 - drop)

            t = torch.randint(0, diff.T, (B,), device=DEVICE).long()
            noise = torch.randn_like(tgt)

            with AMP_CTX:
                x_t = diff.q_sample(tgt, t, noise)
                x_in = torch.cat([x_t, y_eff, m_eff], dim=1)

                if OBJECTIVE == "v":
                    v_tgt = diff.v_from_eps_x0(t, noise, tgt)
                    v_pred = model(x_in, t)
                    x0_pred = diff.x0_from_v_xt(t, v_pred, x_t).clamp(-1, 1)
                    pred_loss = F.mse_loss(v_pred, v_tgt)
                else:  # eps
                    eps_pred = model(x_in, t)
                    x0_pred = torch.clamp(
                        (x_t - diff.sqrt_om[t].view(-1, 1, 1, 1) * eps_pred)
                        / diff.sqrt_acp[t].view(-1, 1, 1, 1),
                        -1, 1
                    )
                    pred_loss = F.mse_loss(eps_pred, noise)

                missing = (1.0 - mask)
                l_rec = F.l1_loss(x0_pred * missing, tgt * missing) * W_MISS + F.l1_loss(x0_pred * mask, tgt * mask) * W_OBS
                l_spec = spectral_l1(x0_pred, tgt) * LAMBDA_SPEC
                l_tv = total_variation(x0_pred) * TV_LAMBDA

                loss = (pred_loss + l_rec + l_spec + l_tv) / float(ACCUM_STEPS)

            SCALER.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP:
                    SCALER.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                SCALER.step(opt)
                SCALER.update()
                opt.zero_grad(set_to_none=True)

            run += float((pred_loss + l_rec + l_spec + l_tv).item())

        tr_loss = run / max(1, len(tr_loader))
        print(f"Epoch {ep:02d}/{EPOCHS} | Train {tr_loss:.5f} | Phase={phase}")

        # ---- Validation
        model.eval()
        vloss = 0.0
        with torch.no_grad(), AMP_CTX:
            for inp, tgt in va_loader:
                inp = inp.to(DEVICE).to(memory_format=torch.channels_last)
                tgt = tgt.to(DEVICE).to(memory_format=torch.channels_last)
                B = tgt.size(0)
                y_obs = inp[:, 0:1]
                mask = inp[:, 1:2]

                t = torch.randint(0, diff.T, (B,), device=DEVICE).long()
                noise = torch.randn_like(tgt)
                x_t = diff.q_sample(tgt, t, noise)
                x_in = torch.cat([x_t, y_obs, mask], 1)

                if OBJECTIVE == "v":
                    v_tgt = diff.v_from_eps_x0(t, noise, tgt)
                    v_pred = model(x_in, t)
                    x0_pred = diff.x0_from_v_xt(t, v_pred, x_t).clamp(-1, 1)
                    pred = F.mse_loss(v_pred, v_tgt)
                else:
                    eps_pred = model(x_in, t)
                    x0_pred = torch.clamp(
                        (x_t - diff.sqrt_om[t].view(-1, 1, 1, 1) * eps_pred)
                        / diff.sqrt_acp[t].view(-1, 1, 1, 1),
                        -1, 1
                    )
                    pred = F.mse_loss(eps_pred, noise)

                missing = (1.0 - mask)
                vloss += (
                    pred
                    + F.l1_loss(x0_pred * missing, tgt * missing) * W_MISS
                    + F.l1_loss(x0_pred * mask, tgt * mask) * W_OBS
                    + spectral_l1(x0_pred, tgt) * LAMBDA_SPEC
                    + total_variation(x0_pred) * TV_LAMBDA
                ).item()

        vloss /= max(1, len(va_loader))
        sched.step(vloss)
        print(f"           Val  {vloss:.5f}")

        # ---- Save best
        if vloss < best:
            best = vloss
            patience_ctr = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": best_state}, ckpt_path)
            print("  Saved best model.")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("  Early stopping.")
                break

    # final eval on val (load best)
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd["model"])
        print(f"[best] loaded {ckpt_path}")

    evaluate_now(
        model, diff, va_loader,
        steps_final=STEPS_FINAL,
        guidance_scale=CFG_LATE,
        repaint_R=REPAINT_LATE[0],
        repaint_J=REPAINT_LATE[1],
        dc_lambda=DC_LAMBDA,
        max_batches=1,
        save_dir=os.path.join(RUNS_DIR, "eval_final"),
        prefix=f"{int(ratio*100)}pct_final"
    )

    return ckpt_path
