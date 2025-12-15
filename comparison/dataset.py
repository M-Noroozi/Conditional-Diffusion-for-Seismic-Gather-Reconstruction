import numpy as np
import torch
from torch.utils.data import Dataset

from .config import PATCH_HW, STRIDE_HW

def build_random_vertical_trace_mask(ntr, drop_ratio, minw=1, maxw=4, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    mask = np.ones(ntr, dtype=np.uint8)
    target = int(round(drop_ratio * ntr))
    dropped = 0
    guard = 0
    while dropped < target and guard < 10000:
        w = rng.randint(minw, maxw + 1)
        s = rng.randint(0, max(1, ntr - w))
        if mask[s:s+w].sum() == w:
            mask[s:s+w] = 0
            dropped += w
        guard += 1
    return mask

def compute_patch_positions(ns, ntr, patch_hw=PATCH_HW, stride_hw=STRIDE_HW):
    H, W = patch_hw
    sH, sW = stride_hw
    positions = []
    max_t = max(1, ns - H + 1)
    max_x = max(1, ntr - W + 1)
    for t0 in range(0, max_t, sH):
        for x0 in range(0, max_x, sW):
            if t0 + H <= ns and x0 + W <= ntr:
                positions.append((t0, x0))
    return positions

def extract_top_left_patch(g, patch_hw):
    H, W = patch_hw
    ns, ntr = g.shape
    if ns < H or ntr < W:
        return []
    return [g[0:H, 0:W]]

class MaskedPatchDataset(Dataset):
    """
    Returns: x(masked), M(mask), y(clean) each as torch tensors [1,H,W]
    """
    def __init__(self, npy_files, patch_hw=(256,120), drop_ratio=0.2, minw=1, maxw=4, seed=123):
        self.items = []
        rng = np.random.RandomState(seed)
        for fp in npy_files:
            g = np.load(fp).astype(np.float32)
            for p in extract_top_left_patch(g, patch_hw):
                H, W = p.shape
                m1d = build_random_vertical_trace_mask(W, drop_ratio, minw, maxw, rng)
                M = np.repeat(m1d[None, :], H, axis=0).astype(np.float32)
                masked = p.copy()
                masked[:, m1d == 0] = 0.0
                self.items.append((masked[None, ...], M[None, ...], p[None, ...]))
        print(f"[DATA] {len(self.items)} samples")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        x, M, y = self.items[i]
        return torch.from_numpy(x), torch.from_numpy(M), torch.from_numpy(y)
