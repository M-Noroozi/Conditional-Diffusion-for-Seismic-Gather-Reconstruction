import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from .config import PATCH_HW, STRIDE_HW, RATIO_DROP, MASK_W_MIN_MAX

def build_random_vertical_trace_mask(ntr, drop_ratio, minw=1, maxw=4, rng=None):
    """Random block-mask until reaching ~drop_ratio."""
    if rng is None:
        rng = np.random.RandomState()
    mask = np.ones(ntr, dtype=np.uint8)
    target = int(round(drop_ratio * ntr))
    dropped = 0
    guard = 0
    while dropped < target and guard < 10000:
        w = rng.randint(minw, maxw + 1)
        s = rng.randint(0, max(1, ntr - w))
        if mask[s:s + w].sum() == w:
            mask[s:s + w] = 0
            dropped += w
        guard += 1
    return mask

def extract_patches(g, patch_hw=PATCH_HW, stride_hw=STRIDE_HW):
    H, W = patch_hw
    sH, sW = stride_hw
    ns, ntr = g.shape
    out = []
    for t0 in range(0, max(1, ns - H + 1), sH):
        for x0 in range(0, max(1, ntr - W + 1), sW):
            p = g[t0:t0 + H, x0:x0 + W]
            if p.shape == (H, W):
                out.append(p.copy())
    return out

class NpyPatchDataset(Dataset):
    """
    Each patch:
      x: [2,H,W] = [masked, mask]
      y: [1,H,W] = clean
    Key improvement: ensure the masked region is not dominated by near-zero energy
    (avoid masking regions with extremely low signal energy too often).
    """
    def __init__(self, npy_files,
                 patch_hw=PATCH_HW,
                 stride_hw=STRIDE_HW,
                 drop_ratio=RATIO_DROP,
                 minw=MASK_W_MIN_MAX[0],
                 maxw=MASK_W_MIN_MAX[1],
                 seed=123):
        rng = np.random.RandomState(seed)
        self.items = []

        for fp in tqdm(npy_files, desc=f"Index NPY(drop={int(drop_ratio*100)}%)"):
            g = np.load(fp).astype(np.float32)  # (ns, ntr), in [-1,1]
            for p in extract_patches(g, patch_hw, stride_hw):
                tries = 0
                while True:
                    m1d = build_random_vertical_trace_mask(p.shape[1], drop_ratio, minw, maxw, rng)

                    total_energy = np.mean(np.abs(p)) + 1e-8
                    if (m1d == 0).sum() > 0:
                        masked_energy = np.mean(np.abs(p[:, m1d == 0]))
                    else:
                        masked_energy = 0.0

                    ratio_en = masked_energy / total_energy
                    if ratio_en >= 0.3 or tries >= 5:
                        break
                    tries += 1

                M = np.repeat(m1d[None, :], p.shape[0], axis=0).astype(np.float32)
                masked = p.copy()
                masked[:, m1d == 0] = 0.0
                x = np.stack([masked, M], axis=0)  # (2,H,W)
                y = p[None, ...]                   # (1,H,W)
                self.items.append((x.astype(np.float32), y.astype(np.float32)))

        if not self.items:
            raise RuntimeError("No patches created.")
        print(f"[DATA] total patches: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = self.items[i]
        return torch.from_numpy(x), torch.from_numpy(y)
