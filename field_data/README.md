# Field Data — Conditional Diffusion for Seismic Gather Reconstruction

This folder contains the **field (real) seismic data** implementation used in the paper:

**Conditional Diffusion Framework for High-Resolution Seismic Gather Reconstruction**

The code performs **missing-trace (vertical gap) reconstruction** on seismic gathers using a
**conditional diffusion model** with strong data-consistency constraints and a ResUNet backbone.

---

## Overview

The proposed method reconstructs missing vertical traces in seismic gathers while preserving:
- Structural continuity
- Long-range coherency
- Amplitude fidelity and spectral content

Key features:
- Cosine diffusion schedule with **v-prediction**
- **ResUNet** backbone with optional **axial attention** along the trace axis
- Classifier-free guidance (CFG) during sampling
- Strict **data-consistency** enforcement on observed traces
- Spectral-domain and spatial regularization losses

---

## Folder Structure

field_data/
├── main.py # Training and evaluation entry point
├── config.py # Paths, hyperparameters, and modes
├── dataset.py # Patch dataset and masking strategy
├── segy_io.py # SEGY reading and caching utilities
├── diffusion.py # Diffusion process and schedules
├── model.py # ResUNet with axial attention
├── sampler.py # DDIM sampling + CFG + data consistency
├── losses.py # Spectral and TV losses
├── utils.py # Utilities (PSNR, plotting, seeding)
└── requirements.txt

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r field_data/requirements.txt

Notes:

segyio is used by default for SEGY reading.

If unavailable, the code automatically falls back to obspy.

Configuration

All paths and hyperparameters are defined in:
field_data/config.py

Key settings:

SINGLE_FILE_PATH: path to the field .sgy file

CACHE_DIR: directory for cached .npy gathers

OUT_ROOT: output directory for runs and results

EXPECTED_TRACES: number of traces per gather (e.g., 120)

RATIO_DROP: missing-trace ratio used during training and evaluation

Data Preparation (SEGY → NPY)

Seismic gathers are cached as .npy files for efficiency.
Caching is performed automatically when needed.

Each cached gather:

Is normalized per gather

Has values in [-1, 1]

Has shape (time_samples, n_traces)

Training

Set the mode in config.py:

MODE = "train"


Run from the repository root:

python -m field_data.main


Outputs:

Best checkpoint saved to:

OUT_ROOT/ratio_XX/field_inpaint_20pct.pth


Training and validation losses printed to the console

Exporting Validation Samples

To export reconstructed validation patches as images:

MODE = "export_samples"


Run:

python -m field_data.main


Outputs:

OUT_ROOT/ratio_XX/eval_export_gpu/


Each PNG shows:

Clean target patch

Missing-trace mask

Masked input

Reconstruction (with PSNR)


Full Validation PSNR

To compute PSNR over the entire validation set (no images):

MODE = "full_val_psnr"


Run:

python -m field_data.main


Saved outputs:

full_val_psnrs.npy

full_val_psnr_stats.txt

Masking Strategy

Training uses vertical block-wise trace masks.
To avoid trivial cases, the dataset retries mask sampling when the masked region contains
very low signal energy.

Notes

Random seeds are fixed for reproducibility.

DataLoader settings are Windows-safe by default (num_workers=0).

The code is designed for single-GPU execution.

Citation

If you use this code, please cite:

Conditional Diffusion Framework for High-Resolution Seismic Gather Reconstruction

Disclaimer

This code is provided for research purposes only.
Ensure that you have the appropriate rights to use and share any field seismic data.
