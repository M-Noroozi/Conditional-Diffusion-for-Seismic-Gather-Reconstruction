import torch
from contextlib import nullcontext

# ============================================================
# MODE
# ============================================================
MODE = "compare"   # "compare" | "train_cnn" | "train_tdae"

# ---- data / cache / segy ----
USE_DIR_MODE     = False
SEGY_DIR         = r"D:\path\to\segys"
SEGY_GLOB        = "*.sgy"
SINGLE_FILE_PATH = r"D:\maryam\data\real_data\seismic.sgy"

CACHE_DIR        = r"E:\maryam\data\real_data\cache_npy"
EXPECTED_TRACES  = 120
GATHER_LIMIT     = None
TRAIN_SPLIT      = 0.90

# ---- diffusion checkpoint (Ours) ----
CKPT_DIFFUSION = r"E:\maryam\data\real_data\runs_field_full\ratio_20\field_inpaint_20pct.pth"

# ---- baseline checkpoints ----
CKPT_CNN   = r"E:\maryam\data\real_data\baselines1\cnn_unet.pth"
CKPT_TDAE  = r"E:\maryam\data\real_data\baselines1\tdae.pth"

# ---- outputs ----
OUT_ROOT   = r"E:\maryam\data\real_data\runs_compare_advanced5"

# ---- patch / mask ----
PATCH_HW        = (256, 120)     # (time, trace)
STRIDE_HW       = (128, 120)     # to sample different offsets
DROP_RATIO      = 0.20
MASK_W_MIN_MAX  = (1, 4)
NUM_EXAMPLES    = 6

# ---- diffusion sampling ----
T_STEPS    = 1000
STEPS_SLOW = 150
GUIDE_SLOW = 1.8

STEPS_FAST = 20
GUIDE_FAST = 1.2

DC_LAMBDA  = 0.3

# ---- training params (CNN / TDAE) ----
BATCH_SIZE_BL = 8
EPOCHS_CNN    = 40
EPOCHS_TDAE   = 40
LR_BL         = 3e-4
W_MISS_BL     = 1.0
W_OBS_BL      = 0.1
TV_LAMBDA_BL  = 1e-3
PATIENCE_BL   = 5

# ---- device / AMP ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP    = (DEVICE.type == "cuda")
AMP_CTX = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP) if AMP else nullcontext()
