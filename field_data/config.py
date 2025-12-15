import os
import torch
from contextlib import nullcontext

# -----------------------------
# CONFIG
# -----------------------------
# MODE = "train"           # "train" | "export_samples" | "full_val_psnr"
# MODE = "full_val_psnr"   # PSNR
MODE = "export_samples"    # png

# Data source
USE_DIR_MODE     = False  # True: folder of SEG(Y); False: single .sgy
SEGY_DIR         = r"D:\path\to\folder\with\segys"           # used if USE_DIR_MODE=True
SEGY_GLOB        = "*.sgy"
SINGLE_FILE_PATH = r"D:\maryam\data\real_data\seismic.sgy"   # main field file

# Cache / split
CACHE_DIR        = r"D:\maryam\data\real_data\cache_npy"
EXPECTED_TRACES  = 120       # Viking = 120
GATHER_LIMIT     = None      # e.g., 100 for quick test, None for all gathers
TRAIN_SPLIT      = 0.90

# Train/eval output
OUT_ROOT         = r"D:\maryam\data\real_data\runs_field_full"
RATIO_DROP       = 0.20      # missing trace ratio

# Patching & mask
PATCH_HW         = (256, 120)   # (time, trace)
STRIDE_HW        = (128, 120)   # trace-wise non-overlap
MASK_W_MIN_MAX   = (1, 4)       # min/max width of missing blocks (in traces)

# Model capacity
BASE_CH          = 48           # 48 is good for 1080Ti; if OOM reduce to 32
AX_HEADS         = 2
USE_AXIAL        = True

# Diffusion & sampling
T_STEPS          = 1000
OBJECTIVE        = "v"          # 'v' (recommended)
STEPS_FINAL      = 120          # for export on GPU, lighter than 250
ETA_DDIM         = 0.0
GUIDE_EVAL       = 1.6          # CFG at inference
REPAINT_EVAL     = (0, 0)       # (R,J) change if you want RePaint

# Train loop
EPOCHS           = 14
EARLY_EPOCHS     = 5
GUIDE_EARLY      = 1.2
REPAINT_EARLY    = (0, 0)
GUIDE_LATE       = 1.6
REPAINT_LATE     = (0, 0)

BATCH_SIZE       = 6   # if OOM → 4
ACCUM_STEPS      = 1
LR               = 2e-4
PATIENCE         = 5
GRAD_CLIP        = 1.0
SEED             = 123

# Loss weights
LAMBDA_SPEC      = 0.03
TV_LAMBDA        = 0.005
W_MISS, W_OBS    = 1.0, 0.1

# Classifier-free dropout on conditioning
P_UNCOND         = 0.10

# Data-consistency during sampling
DC_LAMBDA        = 0.3

# Export / eval
EXPORT_SAMPLES_N = 6   # number of patches to save as PNG in MODE="export_samples"
FULLVAL_STEPS    = 60  # steps for full-val PSNR (lighter than STEPS_FINAL)

# Device/AMP
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # only 1080Ti
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP    = (DEVICE.type == 'cuda')
AMP_CTX = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP) if AMP else nullcontext()
SCALER  = torch.cuda.amp.GradScaler(enabled=AMP)
