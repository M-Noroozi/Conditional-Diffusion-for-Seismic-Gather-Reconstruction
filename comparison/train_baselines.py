import os
import torch
import torch.nn.functional as F

from .config import DEVICE, AMP_CTX, LR_BL, W_MISS_BL, W_OBS_BL, TV_LAMBDA_BL, PATIENCE_BL
from .utils import tv_loss_torch

def train_baseline(model, train_loader, val_loader, epochs, ckpt_path, name="CNN"):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_BL)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    patience_counter = 0

    for ep in range(1, epochs + 1):
        # train
        model.train()
        run = 0.0
        for x, M, y in train_loader:
            x = x.to(DEVICE)
            M = M.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            with AMP_CTX:
                pred = model(x, M)
                miss = (1.0 - M)
                loss_miss = F.l1_loss(pred * miss, y * miss)
                loss_obs  = F.l1_loss(pred * M, y * M)
                loss_tv   = tv_loss_torch(pred)
                loss = (W_MISS_BL * loss_miss + W_OBS_BL * loss_obs + TV_LAMBDA_BL * loss_tv)

            loss.backward()
            opt.step()
            run += float(loss.item())

        train_loss = run / max(1, len(train_loader))

        # val
        model.eval()
        vrun = 0.0
        with torch.no_grad(), AMP_CTX:
            for x, M, y in val_loader:
                x = x.to(DEVICE)
                M = M.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x, M)
                miss = (1.0 - M)
                loss_miss = F.l1_loss(pred * miss, y * miss)
                loss_obs  = F.l1_loss(pred * M, y * M)
                loss_tv   = tv_loss_torch(pred)
                vloss = (W_MISS_BL * loss_miss + W_OBS_BL * loss_obs + TV_LAMBDA_BL * loss_tv)
                vrun += float(vloss.item())

        val_loss = vrun / max(1, len(val_loader))
        sched.step(val_loss)

        print(f"[{name}] Epoch {ep:02d}/{epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"[{name}] ✓ saved best → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_BL:
                print(f"[{name}] early stopping.")
                break
