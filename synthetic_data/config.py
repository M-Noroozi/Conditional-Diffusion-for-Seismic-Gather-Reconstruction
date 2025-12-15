import os
import torch
from contextlib import nullcontext

# -----------------------------
# Paths / dataset
# -----------------------------
SEGY_DIR  = r"---"          # TODO: set your path
SEGY_GLOB = "*.segy"

EXPECTED_TRACES = None      # set e.g., 96 or 120 if you want to clip traces
FILE_LIMIT      = None      # e.g., 20 for quick test, None for all

# Masking
RATIOS          = [0.2]
MASK_W_MIN_MAX  = (1, 4)

# Patching
PATCH_HW   = (128, 96)
STRIDE_HW  = (128, 96)

# -----------------------------
# Training
# -----------------------------
SEED       = 123
EPOCHS     = 14
BATCH_SIZE = 2
LR         = 2e-4
PATIENCE   = 5
GRAD_CLIP  = 1.0
ACCUM_STEPS = 4

# Two-phase schedule
EARLY_EPOCHS = 5
CFG_EARLY    = 1.2
REPAINT_EARLY = (0, 0)

CFG_LATE     = 1.6
REPAINT_LATE = (0, 0)

# Diffusion
T_STEPS    = 1000
OBJECTIVE  = "v"   # "v" or "eps"

# Sampling / eval
STEPS_FINAL = 250
GUIDANCE_EVAL = 1.6
DC_LAMBDA     = 0.3

# Loss weights
LAMBDA_SPEC = 0.03
TV_LAMBDA   = 0.005
P_UNCOND    = 0.10
W_MISS, W_OBS = 1.0, 0.1

# Output
RUNS_DIR = "runs_bp2004_v2_clean"

# Preview (kept for compatibility; disabled by default)
PREVIEW_ENABLE = False
PREVIEW_EVERY_STEPS = 0

# Model
BASE_CH   = 32
AX_HEADS  = 2
USE_AXIAL = True

# -----------------------------
# Device / AMP
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = (DEVICE.type == "cuda")
AMP_CTX = torch.autocast(device_type="cuda", dtype=torch.float16) if AMP else nullcontext()
SCALER = torch.cuda.amp.GradScaler(enabled=AMP)
