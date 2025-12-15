# Conditional Diffusion Framework for High-Resolution Seismic Gather Reconstruction

This repository contains the code accompanying the paper:

**Conditional Diffusion Framework for High-Resolution Seismic Gather Reconstruction**

It provides:
- **Field-data pipeline** (SEGY → FFID gathers → cached NPY → patch training/inference)
- **Synthetic (BP2004) pipeline** (SEGY patches for controlled experiments)
- **Baseline comparison suite** (CNN-UNet, TDAE, Fast Diffusion variant, and simplified HSLR)

> **Goal:** reconstruct missing *vertical trace intervals* while preserving structural continuity and maintaining spectral fidelity.

---

## Datasets

### Synthetic Data — BP 2004 Benchmark Dataset
To provide a controlled experimental setup and evaluate the proposed conditional diffusion framework, synthetic experiments were conducted using the **BP 2004 Benchmark dataset** (Billette & Brandsberg-Dahl, 2005).  
This dataset contains realistic synthetic seismic shot gathers with complex wave phenomena, while offering a fully controlled ground truth.

- Official description (SEG Wiki):  
  https://wiki.seg.org/wiki/2004_BP_velocity_benchmark
- Reference:  
  Billette, F., & Brandsberg-Dahl, S. (2005). *The 2004 BP velocity benchmark*. SEG Annual Meeting.

> **Note:** The dataset is publicly available for research and educational use. Please follow the terms specified by SEG.

---

### Field Data — Mobil AVO Viking Graben Line 12 (SEG Open Data)
Field-data experiments were performed using real shot gathers from the **Mobil AVO Viking Graben Line 12 survey**, provided through the **SEG Open Data** initiative.  
The dataset consists of **1001 seismic shot gathers** and exhibits realistic acquisition effects, ambient noise, and amplitude variations with offset, posing significant challenges for interpolation methods.

- Dataset page (SEG Open Data):  
  https://wiki.seg.org/wiki/Mobil_AVO_Viking_Graben_Line_12
- Provider: SEG Open Data / Mobil

> **Note:** This repository does **not** include the field SEGY files.  
> Users must download the data separately from SEG Open Data and ensure compliance with the dataset’s usage terms.


## Repository structure

├── field_data/
│ ├── (modular scripts) # training / sampling / full-val PSNR for real field gathers
│ └── README.md (optional)
├── synthetic_data/
│ ├── main.py # entry point for BP2004 training
│ ├── train.py # training loop for synthetic experiments
│ ├── (other modules) # model, diffusion, dataset, utils, eval, losses, config
│ └── README.md (optional)
├── comparison/
│ ├── main.py # MODE dispatcher: compare / train_cnn / train_tdae
│ ├── compare.py # runs evaluation + figures + metrics
│ ├── train_baselines.py # baseline training loops
│ ├── baselines.py # CNN-UNet, TDAE, HSLR
│ ├── segy_cache.py # SEGY→NPY caching (FFID)
│ ├── dataset.py # patch sampling + masks
│ ├── diffusion_core.py # diffusion schedule + reconstruction
│ ├── models_diffusion.py # ResUNet + Axial attention (Ours)
│ ├── utils.py # metrics + plotting
│ └── config.py # paths + hyperparameters
├── requirements.txt
└── LICENSE


---

## Method overview

**Ours (Conditional Diffusion Inpainting)**:
- Cosine noise schedule, **v-prediction**
- **ResUNet backbone** with **axial attention along the trace axis**
- **Classifier-Free Guidance (CFG)** at inference
- **Strict data consistency** enforced during sampling on observed traces
- Losses include:
  - reconstruction loss (missing + observed weights)
  - optional spectral L1 (log amplitude spectrum)
  - mild total variation (TV) regularization

**Fast diffusion (FDM)**:
- Same diffusion model, fewer sampling steps + weaker CFG

Baselines:
- **CNN-UNet**
- **TDAE** (Twice Denoising Autoencoder; run twice at inference)
- **HSLR** (simplified Hankel sparse low-rank)

---

## Installation

### 1) Create environment
```bash
conda create -n seismic-diffusion python=3.10 -y
conda activate seismic-diffusion

Install dependencies
pip install -r requirements.txt

## Datasets

### Synthetic Data — BP 2004 Benchmark Dataset
To provide a controlled experimental setup and evaluate the proposed conditional diffusion framework, synthetic experiments were conducted using the **BP 2004 Benchmark dataset** (Billette & Brandsberg-Dahl, 2005).  
This dataset contains realistic synthetic seismic shot gathers with complex wave phenomena, while offering a fully controlled ground truth.

- Official description (SEG Wiki):  
  https://wiki.seg.org/wiki/2004_BP_velocity_benchmark
- Reference:  
  Billette, F., & Brandsberg-Dahl, S. (2005). *The 2004 BP velocity benchmark*. SEG Annual Meeting.

> **Note:** The dataset is publicly available for research and educational use. Please follow the terms specified by SEG.

---

### Field Data — Mobil AVO Viking Graben Line 12 (SEG Open Data)
Field-data experiments were performed using real shot gathers from the **Mobil AVO Viking Graben Line 12 survey**, provided through the **SEG Open Data** initiative.  
The dataset consists of **1001 seismic shot gathers** and exhibits realistic acquisition effects, ambient noise, and amplitude variations with offset, posing significant challenges for interpolation methods.

- Dataset page (SEG Open Data):  
  https://wiki.seg.org/wiki/Mobil_AVO_Viking_Graben_Line_12
- Provider: SEG Open Data / Mobil

> **Note:** This repository does **not** include the field SEGY files.  
> Users must download the data separately from SEG Open Data and ensure compliance with the dataset’s usage terms.


Data preparation
Field data (SEGY → cached NPY by FFID)

The field pipeline and the comparison pipeline both support caching gathers to .npy files.

Set in the corresponding config.py (or top-of-script variables if you kept it monolithic):

USE_DIR_MODE (folder mode or single file)

SEGY_DIR, SEGY_GLOB OR SINGLE_FILE_PATH

CACHE_DIR

EXPECTED_TRACES (e.g., 120 for Viking-type gathers)

Caching is automatic if CACHE_DIR is empty.

Synthetic data (BP2004)

Point SEGY_DIR and SEGY_GLOB to your BP2004 .segy files

Patch extraction uses (PATCH_HW, STRIDE_HW) in synthetic_data/config.py


Running: Field-data pipeline

Inside field_data/, you typically have three modes:

train → train diffusion model

export_samples → save a few reconstructed PNG samples from validation

full_val_psnr → compute PSNR across full validation patches

Example (module-style):

python -m field_data.main


Common outputs:

checkpoints (e.g., field_inpaint_20pct.pth)

exported sample figures (Clean / Mask / Masked / Reconstruction)

optional metrics arrays (e.g., full_val_psnrs.npy)

Running: Synthetic (BP2004)

From the repository root:

python -m synthetic_data.main


Key configuration (in synthetic_data/config.py):

SEGY_DIR, SEGY_GLOB

RATIOS = [0.2] (or multiple ratios)

training hyperparameters (EPOCHS, BATCH_SIZE, etc.)

Running: Baseline training and comparison

All baseline and comparison operations are under comparison/.

1) Train CNN-UNet baseline

Set in comparison/config.py:

MODE = "train_cnn"


Then run:

python -m comparison.main


This writes CKPT_CNN.

2) Train TDAE baseline

Set:

MODE = "train_tdae"


Run:

python -m comparison.main


This writes CKPT_TDAE.

3) Compare all methods

Set:

MODE = "compare"


Run:

python -m comparison.main


Outputs in OUT_ROOT:

metrics_advanced.txt (CSV-like: example, method, PSNR, MSE, SSIM)

Figures per example patch:

example_XX_adv_offsets.png (2×4 panel of reconstructions)

example_XX_adv_offsets_diff.png (difference maps)

example_XX_adv_offsets_fk.png (F–K amplitude difference)

example_XX_adv_offsets_spectrum.png (average amplitude spectrum)


Configuration notes (important)

Paths: most scripts rely on absolute paths in config.py. Update them for your system.

GPU / AMP: AMP is enabled automatically when CUDA is available.

Windows DataLoader: for field-data training, num_workers=0 is recommended.

Expected traces: if EXPECTED_TRACES mismatches, gathers may be skipped during caching.

Normalization: gather-wise z-score → clip(3σ) → scale to [-1, 1]

