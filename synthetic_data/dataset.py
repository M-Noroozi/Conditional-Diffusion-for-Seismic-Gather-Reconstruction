import os, glob
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from .segy_io import read_shot_gather

def build_random_vertical_trace_mask(ntr, drop_ratio, min_width=1, max_width=4, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    mask = np.ones(ntr, dtype=np.uint8)
    target = int(round(drop_ratio * ntr))
    dropped = 0
    guard = 0
    while dropped < target and guard < 10000:
        w = rng.randint(min_width, max_width + 1)
        s = rng.randint(0, max(1, ntr - w))
        if mask[s:s+w].sum() == w:
            mask[s:s+w] = 0
            dropped += w
        guard += 1
    return mask

def apply_mask_to_gather(g, mask1d):
    ns, ntr = g.shape
    M = mask1d[None, :].repeat(ns, axis=0)
    out = g.copy()
    out[:, mask1d == 0] = 0.0
    return out.astype(np.float32), M.astype(np.float32)

def extract_patches_from_gather(g, patch_hw=(128, 96), stride_hw=(128, 96)):
    H, W = patch_hw
    sH, sW = stride_hw
    ns, ntr = g.shape
    patches = []
    for t0 in range(0, max(1, ns - H + 1), sH):
        for x0 in range(0, max(1, ntr - W + 1), sW):
            p = g[t0:t0 + H, x0:x0 + W]
            if p.shape == (H, W):
                patches.append(p.copy())
    return patches

class BP2004PatchDataset(Dataset):
    def __init__(
        self,
        files,
        patch_hw=(128, 96),
        stride_hw=(128, 96),
        drop_ratio=0.2,
        min_width=1,
        max_width=4,
        expected_traces=None,
        seed=123,
        verbose=False,
        file_limit=None
    ):
        self.items = []
        rng = np.random.RandomState(seed)
        if file_limit is not None:
            files = files[:file_limit]

        for fp in tqdm(files, desc=f"Indexing SEGY (drop={int(drop_ratio*100)}%)"):
            try:
                g = read_shot_gather(fp, expected_traces, True, verbose)
            except Exception as e:
                print(f"[warn] skip {fp}: {e}")
                continue

            for p in extract_patches_from_gather(g, patch_hw, stride_hw):
                m = build_random_vertical_trace_mask(p.shape[1], drop_ratio, min_width, max_width, rng)
                masked, M = apply_mask_to_gather(p, m)
                self.items.append((masked.astype(np.float32), M.astype(np.float32), p.astype(np.float32)))

        if not self.items:
            raise RuntimeError("No patches created — check patch/stride or input files.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        masked, M, clean = self.items[i]
        x = np.stack([masked, M], axis=0)  # (2,H,W)
        y = clean[None, ...]               # (1,H,W)
        return torch.from_numpy(x), torch.from_numpy(y)

def build_dataset_from_dir(segy_dir, segy_glob="*.segy", **kwargs):
    files = sorted(glob.glob(os.path.join(segy_dir, segy_glob)))
    if not files:
        raise FileNotFoundError(f"No SEGY files found in {segy_dir} with pattern {segy_glob}")
    return BP2004PatchDataset(files, **kwargs)
