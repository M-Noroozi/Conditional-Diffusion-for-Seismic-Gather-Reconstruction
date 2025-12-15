import os
import math
import numpy as np
import torch
import torch.nn.functional as F

def seed_everything(seed=123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def compute_psnr(x, y, data_range=2.0):
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float("inf")
    peak = data_range / 2.0
    return 10.0 * math.log10((peak**2) / mse)
