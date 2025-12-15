import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def robust_v(img, q=99.0):
    return float(np.percentile(np.abs(img), q) + 1e-8)

def compute_psnr_np(x, y, data_range=2.0):
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    peak = data_range / 2.0
    return 10.0 * math.log10((peak**2) / mse)

def compute_ssim_global_np(x, y, data_range=2.0):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    K1, K2 = 0.01, 0.03
    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sig_x = ((x - mu_x) ** 2).mean()
    sig_y = ((y - mu_y) ** 2).mean()
    sig_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
    if den == 0:
        return 1.0
    return float(num / den)

def compute_fk_amplitude(patch):
    fk = np.fft.fft2(patch)
    fk = np.fft.fftshift(fk)
    amp = np.abs(fk)
    return np.log1p(amp)

def compute_avg_amp_spectrum(patch):
    spec = np.fft.rfft(patch, axis=0)
    amp = np.abs(spec)
    return amp.mean(axis=1)

def tv_loss_torch(x):
    return ((x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean() +
            (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean())

# -----------------------------
# Plotting
# -----------------------------
def save_figure_2x4(clean, mask2d, masked,
                    rec_ours, rec_cnn, rec_tdae, rec_fdm, rec_hslr,
                    psnrs, out_png):
    v = robust_v(clean, 99.5)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)

    im0 = axes[0,0].imshow(clean, cmap="seismic", aspect="auto", vmin=-v, vmax=v)
    axes[0,0].set_title("Clean (val)"); axes[0,0].set_ylabel("Time")
    fig.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(masked, cmap="seismic", aspect="auto", vmin=-v, vmax=v)
    axes[0,1].set_title("Masked"); fig.colorbar(im1, ax=axes[0,1])

    im2 = axes[0,2].imshow(mask2d, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0,2].set_title("Mask"); fig.colorbar(im2, ax=axes[0,2])

    im3 = axes[0,3].imshow(rec_ours, cmap="seismic", aspect="auto", vmin=-v, vmax=v)
    axes[0,3].set_title(f"Ours (Diff)\nPSNR={psnrs['ours']:.2f} dB")
    fig.colorbar(im3, ax=axes[0,3])

    im4 = axes[1,0].imshow(rec_cnn, cmap="seismic", aspect="auto", vmin=-v, vmax=v)
    axes[1,0].set_title(f"CNN-UNet\nPSNR={psnrs['cnn']:.2f} dB")
    axes[1,0].set_xlabel("Trace"); axes[1,0].set_ylabel("Time")
    fig.colorbar(im4, ax=axes[1,0])

    im5 = axes[1,1].imshow(rec_tdae, cmap="seismic", aspect="auto", vmin=-v, vmax=v)
    axes[1,1].set_title(f"TDAE\nPSNR={psnrs['tdae']:.2f} dB")
    axes[1,1].set_xlabel("Trace"); fig.colorbar(im5, ax=axes[1,1])

    im6 = axes[1,2].imshow(rec_fdm, cmap="seismic", aspect="auto", vmin=-v, vmax=v)
    axes[1,2].set_title(f"FDM (fast diff)\nPSNR={psnrs['fdm']:.2f} dB")
    axes[1,2].set_xlabel("Trace"); fig.colorbar(im6, ax=axes[1,2])

    im7 = axes[1,3].imshow(rec_hslr, cmap="seismic", aspect="auto", vmin=-v, vmax=v)
    axes[1,3].set_title(f"HSLR (Hankel)\nPSNR={psnrs['hslr']:.2f} dB")
    axes[1,3].set_xlabel("Trace"); fig.colorbar(im7, ax=axes[1,3])

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print("[FIG]", out_png)

def save_diff_figure(clean, rec_ours, rec_cnn, rec_tdae, rec_fdm, rec_hslr, out_png):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)

    def show(ax, img, title):
        v = robust_v(img, 99.5)
        im = ax.imshow(img, cmap="seismic", aspect="auto", vmin=-v, vmax=v)
        ax.set_title(title); ax.set_xlabel("Trace"); ax.set_ylabel("Time")
        fig.colorbar(im, ax=ax)

    show(axes[0,0], clean, "Clean (val)")

    show(axes[0,1], clean - rec_ours, "Diff: Clean - Ours")
    show(axes[1,0], clean - rec_cnn,  "Diff: Clean - CNN")
    show(axes[1,1], clean - rec_tdae, "Diff: Clean - TDAE")
    show(axes[2,0], clean - rec_fdm,  "Diff: Clean - FDM")
    show(axes[2,1], clean - rec_hslr, "Diff: Clean - HSLR")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print("[FIG-DIFF]", out_png)

def save_fk_figure(clean, methods, out_png):
    fk_clean = compute_fk_amplitude(clean)
    v = robust_v(fk_clean, 99.5)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)

    im0 = axes[0, 0].imshow(fk_clean, cmap="viridis", aspect="auto", vmin=0, vmax=v)
    axes[0, 0].set_title("Clean F-K amplitude")
    axes[0, 0].set_ylabel("Frequency index")
    fig.colorbar(im0, ax=axes[0, 0])

    method_names = [
        ("ours", "Ours"), ("cnn", "CNN-UNet"), ("tdae", "TDAE"),
        ("fdm", "FDM (fast diff)"), ("hslr", "HSLR")
    ]

    for ax, (key, label) in zip(axes.flat[1:], method_names):
        if key not in methods:
            ax.axis("off"); continue
        fk_rec = compute_fk_amplitude(methods[key])
        fk_diff = np.abs(fk_clean - fk_rec)
        im = ax.imshow(fk_diff, cmap="viridis", aspect="auto", vmin=0, vmax=v)
        ax.set_title(f"F-K diff: Clean vs {label}")
        ax.set_xlabel("Wavenumber index"); ax.set_ylabel("Frequency index")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print("[FIG-FK]", out_png)

def save_amp_spectrum(clean, methods, out_png):
    freq_idx = np.arange(compute_avg_amp_spectrum(clean).shape[0])
    plt.figure(figsize=(6, 4))

    spec_clean = compute_avg_amp_spectrum(clean)
    plt.semilogy(freq_idx, spec_clean, label="Clean", linewidth=2)

    for key, label in [("ours","Ours"),("cnn","CNN-UNet"),("tdae","TDAE"),("fdm","FDM"),("hslr","HSLR")]:
        if key not in methods:
            continue
        spec = compute_avg_amp_spectrum(methods[key])
        plt.semilogy(freq_idx, spec, label=label, linewidth=1.5)

    plt.xlabel("Frequency index")
    plt.ylabel("Average amplitude (log scale)")
    plt.title("Average amplitude spectrum (time axis)")
    plt.legend(loc="best")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print("[FIG-SPEC]", out_png)
