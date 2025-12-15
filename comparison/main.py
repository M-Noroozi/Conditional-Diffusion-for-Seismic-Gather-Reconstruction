import os, glob
from torch.utils.data import DataLoader

from .config import (
    MODE, OUT_ROOT, CACHE_DIR, TRAIN_SPLIT,
    USE_DIR_MODE, SINGLE_FILE_PATH, SEGY_DIR, SEGY_GLOB,
    EXPECTED_TRACES, GATHER_LIMIT,
    PATCH_HW, DROP_RATIO, MASK_W_MIN_MAX,
    BATCH_SIZE_BL, EPOCHS_CNN, EPOCHS_TDAE, CKPT_CNN, CKPT_TDAE
)

from .utils import ensure_dir
from .segy_cache import cache_gathers
from .dataset import MaskedPatchDataset
from .baselines import SimpleUNet, ConvAE
from .train_baselines import train_baseline
from .compare import run_compare


def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(CACHE_DIR)

    # cache ensure
    npy_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npy")))
    if len(npy_files) == 0:
        npy_files = cache_gathers(USE_DIR_MODE, SINGLE_FILE_PATH, SEGY_DIR, SEGY_GLOB,
                                  CACHE_DIR, EXPECTED_TRACES, GATHER_LIMIT)

    n = len(npy_files)
    Ni = int(TRAIN_SPLIT * n)
    train_files = npy_files[:Ni]

    if MODE in ("train_cnn", "train_tdae"):
        # baseline split from train_files
        n_bl = len(train_files)
        Ni_bl = int(0.9 * n_bl)
        bl_train_files = train_files[:Ni_bl]
        bl_val_files = train_files[Ni_bl:] if Ni_bl < n_bl else train_files

        ds_train = MaskedPatchDataset(
            bl_train_files, patch_hw=PATCH_HW,
            drop_ratio=DROP_RATIO,
            minw=MASK_W_MIN_MAX[0], maxw=MASK_W_MIN_MAX[1],
            seed=123
        )
        ds_val = MaskedPatchDataset(
            bl_val_files, patch_hw=PATCH_HW,
            drop_ratio=DROP_RATIO,
            minw=MASK_W_MIN_MAX[0], maxw=MASK_W_MIN_MAX[1],
            seed=456
        )

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE_BL, shuffle=True, num_workers=0)
        dl_val   = DataLoader(ds_val, batch_size=BATCH_SIZE_BL, shuffle=False, num_workers=0)

        if MODE == "train_cnn":
            model = SimpleUNet(in_ch=2, base=32)
            train_baseline(model, dl_train, dl_val, EPOCHS_CNN, CKPT_CNN, name="CNN")
        else:
            model = ConvAE(in_ch=2, base=32)
            train_baseline(model, dl_train, dl_val, EPOCHS_TDAE, CKPT_TDAE, name="TDAE")
        return

    # compare
    run_compare()


if __name__ == "__main__":
    main()
