# ============================================================
# Seismic Diffusion Inpainting (BP2004 SEGY) — clean v2.1
# - Cosine schedule
# - (optional) eps- or v-pred objective  (default: v)
# - Data: z-score -> clip(3σ) -> scale to [-1,1]
# - No self-conditioning at inference (4th channel = 0 always)
# - TV-loss (mild) to reduce speckle
# - Two-phase train (5 warmup -> strong CFG). RePaint default off.
# ============================================================
import os, glob, math, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

# -----------------------------
# Device / AMP
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = (DEVICE.type == 'cuda')
AMP_CTX = torch.autocast(device_type='cuda', dtype=torch.float16) if AMP else nullcontext()
SCALER = torch.cuda.amp.GradScaler(enabled=AMP)

def seed_everything(seed=123):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

# -----------------------------
# SEGY I/O
# -----------------------------
_USE_OBSPY = False
try:
    import segyio
except Exception:
    _USE_OBSPY = True
    from obspy.io.segy.segy import _read_segy

def read_shot_gather(path, expected_traces=None, normalize=True, verbose=False):
    if not _USE_OBSPY:
        with segyio.open(path, "r", ignore_geometry=True) as f:
            traces = np.asarray([f.trace[i] for i in range(f.tracecount)], dtype=np.float32)
            g = traces.T  # (ns, ntr)
    else:
        st = _read_segy(path, headonly=False)
        traces = np.stack([tr.data for tr in st.traces]).astype(np.float32)
        g = traces.T

    if expected_traces is not None and g.shape[1] > expected_traces:
        g = g[:, :expected_traces]

    if normalize:
        # z-score per gather -> clip 3σ -> map to [-1,1]
        m = np.mean(g, keepdims=True); s = np.std(g, keepdims=True) + 1e-8
        g = (g - m) / s
        g = np.clip(g, -3.0, 3.0) / 3.0  # [-1,1]
        g = g.astype(np.float32)
    return g

# -----------------------------
# Masks (vertical)
# -----------------------------
def build_random_vertical_trace_mask(ntr, drop_ratio, min_width=1, max_width=4, rng=None):
    if rng is None: rng = np.random.RandomState()
    mask = np.ones(ntr, dtype=np.uint8)
    target = int(round(drop_ratio*ntr))
    dropped=0; guard=0
    while dropped<target and guard<10000:
        w = rng.randint(min_width, max_width+1)
        s = rng.randint(0, max(1, ntr-w))
        if mask[s:s+w].sum() == w:
            mask[s:s+w] = 0; dropped += w
        guard += 1
    return mask

def apply_mask_to_gather(g, mask1d):
    ns,ntr = g.shape
    M = mask1d[None,:].repeat(ns, axis=0)
    out = g.copy(); out[:, mask1d==0] = 0.0
    return out.astype(np.float32), M.astype(np.float32)

# -----------------------------
# Patches
# -----------------------------
def extract_patches_from_gather(g, patch_hw=(128,96), stride_hw=(128,96)):
    H,W = patch_hw; sH,sW = stride_hw
    ns,ntr = g.shape; patches=[]
    for t0 in range(0, max(1, ns-H+1), sH):
        for x0 in range(0, max(1, ntr-W+1), sW):
            p = g[t0:t0+H, x0:x0+W]
            if p.shape==(H,W): patches.append(p.copy())
    return patches

class BP2004PatchDataset(Dataset):
    def __init__(self, files, patch_hw=(128,96), stride_hw=(128,96),
                 drop_ratio=0.2, min_width=1, max_width=4,
                 expected_traces=None, seed=123, verbose=False, file_limit=None):
        self.items=[]; rng=np.random.RandomState(seed)
        if file_limit is not None: files = files[:file_limit]
        for fp in tqdm(files, desc=f"Indexing SEGY (drop={int(drop_ratio*100)}%)"):
            try:
                g = read_shot_gather(fp, expected_traces, True, verbose)
            except Exception as e:
                print(f"[warn] skip {fp}: {e}"); continue
            for p in extract_patches_from_gather(g, patch_hw, stride_hw):
                m = build_random_vertical_trace_mask(p.shape[1], drop_ratio, min_width, max_width, rng)
                masked, M = apply_mask_to_gather(p, m)
                self.items.append((masked.astype(np.float32), M.astype(np.float32), p.astype(np.float32)))
        if not self.items:
            raise RuntimeError("No patches created — check patch/stride or files.")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        masked,M,clean = self.items[i]     # (H,W)
        x = np.stack([masked, M], axis=0)  # (2,H,W)
        y = clean[None, ...]               # (1,H,W)
        return torch.from_numpy(x), torch.from_numpy(y)

# -----------------------------
# Diffusion core
# -----------------------------
def timestep_embedding(t, dim):
    half = dim//2
    freqs = torch.exp(-math.log(10000)*torch.arange(0,half,device=t.device)/half)
    args = t.float().unsqueeze(1)*freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], -1)
    if dim%2==1: emb = F.pad(emb,(0,1))
    return emb

def make_beta_schedule_cosine(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float32)
    f = torch.cos(((t/T + s) / (1+s)) * math.pi * 0.5)**2
    alpha_bar = (f / f[0]).clamp(min=1e-8)
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-8, max=0.999)
    return betas

class Diffusion:
    def __init__(self, T=1000, device=DEVICE):
        self.T=T; self.device=device
        self.betas = make_beta_schedule_cosine(T).to(device)
        self.alphas=1.0-self.betas
        self.acp=torch.cumprod(self.alphas,0)
        self.acp_prev=torch.cat([torch.tensor([1.0],device=device), self.acp[:-1]],0)
        self.sqrt_acp=torch.sqrt(torch.clamp(self.acp,1e-12))
        self.sqrt_om =torch.sqrt(torch.clamp(1.0-self.acp,1e-12))
        self.post_var=self.betas*(1-self.acp_prev)/torch.clamp(1-self.acp,1e-12)

    def q_sample(self,x0,t,noise=None):
        if noise is None: noise=torch.randn_like(x0)
        s=self.sqrt_acp[t].view(-1,1,1,1); o=self.sqrt_om[t].view(-1,1,1,1)
        return s*x0 + o*noise

    # v <-> (eps, x0)
    def v_from_eps_x0(self,t,eps,x0):
        s=self.sqrt_acp[t].view(-1,1,1,1); o=self.sqrt_om[t].view(-1,1,1,1); return s*eps - o*x0
    def eps_from_v_xt(self,t,v,x):
        s=self.sqrt_acp[t].view(-1,1,1,1); o=self.sqrt_om[t].view(-1,1,1,1); return o*x + s*v
    def x0_from_v_xt(self,t,v,x):
        s=self.sqrt_acp[t].view(-1,1,1,1); o=self.sqrt_om[t].view(-1,1,1,1); return s*x - o*v

    def make_sampling_schedule(self, steps):
        return torch.linspace(0,self.T-1,steps,dtype=torch.long,device=self.device).tolist()

    def project_observation_to_xt(self,y_obs,mask,t,noise_fixed):
        s=self.sqrt_acp[t].view(-1,1,1,1); o=self.sqrt_om[t].view(-1,1,1,1)
        y_t=s*y_obs + o*noise_fixed  # forward on observed
        return mask*y_t

# -----------------------------
# Model
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self,in_ch,out_ch,time_emb_dim=None):
        super().__init__()
        self.conv1=nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.conv2=nn.Conv2d(out_ch,out_ch,3,padding=1)
        ng=min(8,out_ch)
        while ng>1 and (out_ch%ng!=0): ng-=1
        self.gn1=nn.GroupNorm(max(1,ng),out_ch); self.gn2=nn.GroupNorm(max(1,ng),out_ch)
        self.act=nn.SiLU(); self.skip=nn.Conv2d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
        self.temb=nn.Linear(time_emb_dim,out_ch) if time_emb_dim is not None else None
    def forward(self,x,emb=None):
        h=self.act(self.gn1(self.conv1(x)))
        if emb is not None and self.temb is not None:
            h = h + self.temb(emb).view(emb.shape[0],-1,1,1)
        h=self.gn2(self.conv2(h))
        return self.act(h + self.skip(x))

class AxialMHSA(nn.Module):
    def __init__(self, ch, num_heads=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(ch, num_heads, batch_first=True)
    def forward(self, x):  # [B,C,H,W] -> attn on W
        B,C,H,W = x.shape
        z = x.permute(0,2,3,1).contiguous().view(B*H, W, C)
        y,_ = self.mha(z, z, z, need_weights=False)
        y = y.view(B,H,W,C).permute(0,3,1,2).contiguous()
        return x + y

class ResUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, time_dim=64, time_emb_dim=256, use_axial=True, ax_heads=2):
        super().__init__()
        self.time_dim=time_dim
        self.time_mlp=nn.Sequential(nn.Linear(time_dim,time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim,time_emb_dim))
        self.enc1 = nn.Sequential(ResBlock(in_ch, base_ch, time_emb_dim), ResBlock(base_ch, base_ch, time_emb_dim))
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(ResBlock(base_ch, base_ch*2, time_emb_dim), ResBlock(base_ch*2, base_ch*2, time_emb_dim))
        self.ax2  = AxialMHSA(base_ch*2, ax_heads) if use_axial else nn.Identity()
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(ResBlock(base_ch*2, base_ch*4, time_emb_dim), ResBlock(base_ch*4, base_ch*4, time_emb_dim))
        self.ax3  = AxialMHSA(base_ch*4, ax_heads) if use_axial else nn.Identity()
        self.down3 = nn.MaxPool2d(2)
        self.bot1 = ResBlock(base_ch*4, base_ch*8, time_emb_dim)
        self.bot2 = ResBlock(base_ch*8, base_ch*4, time_emb_dim)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(ResBlock(base_ch*8, base_ch*4, time_emb_dim), ResBlock(base_ch*4, base_ch*4, time_emb_dim))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(ResBlock(base_ch*6, base_ch*2, time_emb_dim), ResBlock(base_ch*2, base_ch*2, time_emb_dim))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(ResBlock(base_ch*3, base_ch, time_emb_dim), ResBlock(base_ch, base_ch, time_emb_dim))
        self.final_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x3, t):
        emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h1 = self.enc1[0](x3, emb); h1 = self.enc1[1](h1, emb)
        h2 = self.down1(h1); h2 = self.enc2[0](h2, emb); h2 = self.enc2[1](h2, emb); h2 = self.ax2(h2)
        h3 = self.down2(h2); h3 = self.enc3[0](h3, emb); h3 = self.enc3[1](h3, emb); h3 = self.ax3(h3)
        b  = self.down3(h3); b  = self.bot1(b, emb); b = self.bot2(b, emb)
        u3 = self.up3(b);  u3 = F.interpolate(u3, size=h3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3[0](torch.cat([u3,h3],1), emb); d3 = self.dec3[1](d3, emb)
        u2 = self.up2(d3); u2 = F.interpolate(u2, size=h2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2[0](torch.cat([u2,h2],1), emb); d2 = self.dec2[1](d2, emb)
        u1 = self.up1(d2); u1 = F.interpolate(u1, size=h1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1[0](torch.cat([u1,h1],1), emb); d1 = self.dec1[1](d1, emb)
        return self.final_conv(d1)

# -----------------------------
# Losses / Metrics
# -----------------------------
def spectral_l1(x, y):
    x=x.squeeze(1); y=y.squeeze(1)
    nfft = 1 << (x.shape[1]-1).bit_length()
    X=torch.fft.rfft(x,n=nfft,dim=1); Y=torch.fft.rfft(y,n=nfft,dim=1)
    sx=torch.log(torch.clamp(torch.abs(X),1e-8)); sy=torch.log(torch.clamp(torch.abs(Y),1e-8))
    return F.l1_loss(sx,sy)

def total_variation(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()

def compute_psnr(x,y, data_range=2.0):
    mse=F.mse_loss(x,y).item()
    if mse==0: return float('inf')
    peak=data_range/2.0
    return 10.0*math.log10((peak**2)/mse)

# -----------------------------
# Classifier-Free Guidance
# -----------------------------
def guided_v_pred(model, x, y_obs, mask, t, guidance_scale=1.6):
    vc = model(torch.cat([x,y_obs,mask],1), t)
    vu = model(torch.cat([x,torch.zeros_like(y_obs),torch.zeros_like(mask)],1), t)
    return vu + guidance_scale*(vc - vu)

# -----------------------------
# Sampler (DDIM; no SC at inference)
# -----------------------------
@torch.no_grad()
def ddim_sample(model, diffusion, y_obs, mask, shape,
                steps=150, eta=0.0, guidance_scale=1.6,
                repaint_R=0, repaint_J=0, dc_lambda=0.3):
    seq = diffusion.make_sampling_schedule(steps)
    B=shape[0]; dev=diffusion.device
    x=torch.randn(shape,device=dev); z_fixed=torch.randn_like(x)

    for j,t in enumerate(reversed(seq)):
        t_t=torch.full((B,),t,dtype=torch.long,device=dev)

        v = guided_v_pred(model,x,y_obs,mask,t_t,guidance_scale)
        eps=diffusion.eps_from_v_xt(t_t,v,x)
        x0 =(x - diffusion.sqrt_om[t]*eps)/diffusion.sqrt_acp[t]
        x0 = torch.clamp(x0,-1,1)

        # PnP data consistency (only on observed)
        x0 = x0 - dc_lambda * mask * (x0 - y_obs)
        x0 = torch.clamp(x0, -1, 1)

        if j<steps-1:
            t_prev=seq[::-1][j+1]
            alpha_t=diffusion.acp[t]; alpha_prev=diffusion.acp[t_prev]
            sigma=eta*torch.sqrt(torch.clamp((1-alpha_prev)/(1-alpha_t),1e-12)*torch.clamp(1-alpha_t/alpha_prev,1e-12))
            c=torch.sqrt(torch.clamp(1-alpha_t/alpha_prev,1e-12))
            x = torch.sqrt(alpha_prev)*x0 + c*torch.sqrt(torch.clamp(1-alpha_prev,1e-12))*eps + sigma*torch.randn_like(x)
        else:
            x = x0

        # enforce observed in x_t-domain
        x_obs_t=diffusion.project_observation_to_xt(y_obs,mask,t_t,z_fixed)
        x=(1.0-mask)*x + x_obs_t

        # (optional) RePaint
        if (repaint_R>0 and repaint_J>0) and (j%repaint_R==0) and (j>0):
            for _ in range(repaint_J):
                n=torch.randn_like(x)
                s=diffusion.sqrt_acp[t].view(-1,1,1,1); o=diffusion.sqrt_om[t].view(-1,1,1,1)
                x = s*x + o*n
                x_obs_t=diffusion.project_observation_to_xt(y_obs,mask,t_t,n)
                x=(1.0-mask)*x + x_obs_t

    return x

# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_now(model, diffusion, val_loader,
                 steps_final=200, guidance_scale=1.6,
                 repaint_R=0, repaint_J=0, dc_lambda=0.3,
                 max_batches=1, save_dir="eval_out", prefix="eval"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for bi,(inp,tgt) in enumerate(val_loader):
        if bi>=max_batches: break
        inp=inp.to(DEVICE); tgt=tgt.to(DEVICE)
        y_obs=inp[:,0:1]; mask=inp[:,1:2]
        x_rec = ddim_sample(model, diffusion, y_obs, mask, tgt.shape,
                            steps=steps_final, eta=0.0, guidance_scale=guidance_scale,
                            repaint_R=repaint_R, repaint_J=repaint_J, dc_lambda=dc_lambda).clamp(-1,1)
        psnr = compute_psnr(tgt[:1], x_rec[:1])

        def show(img, title, vclip=99.0):
            v = np.percentile(np.abs(img), vclip)+1e-8
            plt.imshow(img, cmap='seismic', aspect='auto', vmin=-v, vmax=+v)
            plt.title(title); plt.colorbar(); plt.xlabel("Trace"); plt.ylabel("Time")

        clean=tgt[0,0].cpu().numpy(); M=mask[0,0].cpu().numpy()
        masked=y_obs[0,0].cpu().numpy(); recon=x_rec[0,0].cpu().numpy()
        plt.figure(figsize=(14,4))
        plt.subplot(1,4,1); show(clean,"Clean")
        plt.subplot(1,4,2); plt.imshow(M, cmap='gray', aspect='auto', vmin=0, vmax=1); plt.colorbar(); plt.title("Mask")
        plt.subplot(1,4,3); show(masked,"Masked")
        plt.subplot(1,4,4); show(recon,f"Final Recon | PSNR={psnr:.2f} dB")
        plt.tight_layout()
        out_png=os.path.join(save_dir, f"{prefix}_b{bi}.png")
        plt.savefig(out_png, dpi=140); plt.close()
        print(f"[EVAL] batch={bi} | PSNR={psnr:.2f} dB | saved: {out_png}")

# -----------------------------
# Train (two-phase)
# -----------------------------
def train_one_ratio(
    ratio,
    segy_dir, segy_glob="*.segy",
    patch_hw=(128, 96), stride_hw=(128, 96),
    expected_traces=None,
    # diffusion
    T=1000, objective="v",  # 'v' or 'eps'
    # phases
    early_epochs=5,
    cfg_early=1.2, repaint_early=(0, 0), preview_steps_early=40,
    cfg_late=1.6, repaint_late=(0, 0), preview_steps_late=60,
    # training
    epochs=12, batch_size=2, lr=2e-4, patience=5, grad_clip=1.0,
    lambda_spec=0.03, tv_lambda=0.005, p_uncond=0.10, w_miss=1.0, w_obs=0.1,
    accum_steps=4,
    # evaluation
    steps_final=200, dc_lambda=0.3,
    # io
    save_dir="runs_bp2004_v2", file_limit=None, num_workers=0,
    preview_every_steps=0, enable_preview=False,
    seed=123, resume=True,
    # model size
    base_ch=32, ax_heads=2, use_axial=True
):
    seed_everything(seed)
    os.makedirs(save_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(segy_dir, segy_glob)))
    if not files: raise FileNotFoundError("No SEGY files found.")

    ds = BP2004PatchDataset(
        files, patch_hw, stride_hw, ratio,
        min_width=1, max_width=4,
        expected_traces=expected_traces, seed=seed, verbose=False, file_limit=file_limit
    )
    N = len(ds); Nv = max(1, int(0.1*N)); Ni = N - Nv
    idx = np.arange(N); np.random.shuffle(idx); tr_idx, va_idx = idx[:Ni], idx[Ni:]

    tr_loader = DataLoader(torch.utils.data.Subset(ds, tr_idx.tolist()),
                           batch_size=batch_size, shuffle=True, num_workers=num_workers,
                           pin_memory=(DEVICE.type=='cuda'))
    va_loader = DataLoader(torch.utils.data.Subset(ds, va_idx.tolist()),
                           batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=(DEVICE.type=='cuda'))

    in_ch = 3  # [x_t, y_obs, mask]  
    model = ResUNet(in_ch=in_ch, base_ch=base_ch, time_dim=64, time_emb_dim=256,
                    use_axial=use_axial, ax_heads=ax_heads).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    diff = Diffusion(T=T, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)

    ckpt_path = os.path.join(save_dir, f"inpaint_v2_clean_{int(ratio*100)}pct.pth")
    if resume and os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(sd['model'])
            print(f"[resume] loaded {ckpt_path} (CPU-mapped)")
        except Exception as e:
            print(f"[resume] failed: {e}")

    print(f"==== Train ratio={int(ratio*100)}% | patches: train={len(tr_idx)}, val={len(va_idx)} ====")

    best = float('inf'); patience_ctr=0; step_ctr=0

    for ep in range(1, epochs+1):
        if ep <= early_epochs:
            guidance_scale = cfg_early; R,J = repaint_early; steps_prev = preview_steps_early; phase="early"
        else:
            guidance_scale = cfg_late;  R,J = repaint_late;  steps_prev = preview_steps_late;  phase="late"
            if ep == early_epochs+1:
                print("[INFO] Phase switched to LATE — one-shot eval.")
                evaluate_now(model, diff, va_loader,
                             steps_final=steps_final, guidance_scale=guidance_scale,
                             repaint_R=R, repaint_J=J, dc_lambda=dc_lambda,
                             max_batches=1, save_dir=os.path.join(save_dir,"eval_phase_switch"),
                             prefix=f"{int(ratio*100)}pct_ep{ep}")

        model.train(); run=0.0
        opt.zero_grad(set_to_none=True)

        for i,(inp,tgt) in enumerate(tqdm(tr_loader, desc=f"train {ep}/{epochs}", leave=False, disable=True)):
            inp = inp.to(DEVICE).to(memory_format=torch.channels_last)
            tgt = tgt.to(DEVICE).to(memory_format=torch.channels_last)
            B = tgt.size(0)
            y_obs = inp[:,0:1]; mask = inp[:,1:2]

            with torch.no_grad():
                drop = (torch.rand(B, device=DEVICE) < p_uncond).float().view(-1,1,1,1)
                y_eff = y_obs * (1.0 - drop)
                m_eff = mask * (1.0 - drop)

            t = torch.randint(0, diff.T, (B,), device=DEVICE).long()
            noise = torch.randn_like(tgt)

            with AMP_CTX:
                x_t = diff.q_sample(tgt, t, noise)
                x_in = torch.cat([x_t, y_eff, m_eff], dim=1)   

                if objective == "v":
                    v_tgt = diff.v_from_eps_x0(t, noise, tgt)
                    v_pred = model(x_in, t)
                    x0_pred = diff.x0_from_v_xt(t, v_pred, x_t).clamp(-1,1)
                    pred_loss = F.mse_loss(v_pred, v_tgt)
                else:  # eps-pred
                    eps_pred = model(x_in, t)
                    x0_pred = torch.clamp((x_t - diff.sqrt_om[t].view(-1,1,1,1)*eps_pred) / diff.sqrt_acp[t].view(-1,1,1,1), -1, 1)
                    pred_loss = F.mse_loss(eps_pred, noise)

                missing = (1.0 - mask)
                l_rec = F.l1_loss(x0_pred*missing, tgt*missing)*w_miss + F.l1_loss(x0_pred*mask, tgt*mask)*w_obs
                l_spec = spectral_l1(x0_pred, tgt)*lambda_spec
                l_tv   = total_variation(x0_pred)*tv_lambda

                loss = (pred_loss + l_rec + l_spec + l_tv) / accum_steps

            SCALER.scale(loss).backward()

            if (i+1) % accum_steps == 0:
                if grad_clip:
                    SCALER.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                SCALER.step(opt); SCALER.update(); opt.zero_grad(set_to_none=True)

            run += float((pred_loss + l_rec + l_spec + l_tv).item())
            step_ctr += 1

        tr_loss = run / max(1, len(tr_loader))
        print(f"Epoch {ep:02d}/{epochs} | Train {tr_loss:.5f} | Phase={phase}")

        # ---- validation
        model.eval(); vloss=0.0
        with torch.no_grad(), AMP_CTX:
            for inp, tgt in tqdm(va_loader, desc=f"valid {ep}/{epochs}", leave=False, disable=True):
                inp=inp.to(DEVICE).to(memory_format=torch.channels_last)
                tgt=tgt.to(DEVICE).to(memory_format=torch.channels_last)
                B=tgt.size(0); y_obs=inp[:,0:1]; mask=inp[:,1:2]
                t=torch.randint(0,diff.T,(B,),device=DEVICE).long()
                noise=torch.randn_like(tgt); x_t=diff.q_sample(tgt,t,noise)
                x_in=torch.cat([x_t,y_obs,mask],1)
                if objective == "v":
                    v_tgt = diff.v_from_eps_x0(t, noise, tgt)
                    v_pred= model(x_in, t)
                    x0_pred=diff.x0_from_v_xt(t,v_pred,x_t).clamp(-1,1)
                    pred = F.mse_loss(v_pred, v_tgt)
                else:
                    eps_pred = model(x_in, t)
                    x0_pred = torch.clamp((x_t - diff.sqrt_om[t].view(-1,1,1,1)*eps_pred) / diff.sqrt_acp[t].view(-1,1,1,1), -1, 1)
                    pred = F.mse_loss(eps_pred, noise)
                missing=(1.0-mask)
                vloss += (pred + F.l1_loss(x0_pred*missing,tgt*missing)*w_miss
                          + F.l1_loss(x0_pred*mask,tgt*mask)*w_obs
                          + spectral_l1(x0_pred,tgt)*lambda_spec
                          + total_variation(x0_pred)*tv_lambda).item()
        vloss/=max(1,len(va_loader)); sched.step(vloss)
        print(f"           Val  {vloss:.5f}")

        if vloss < best:
            best = vloss; patience_ctr=0
            best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save({'model': best_state}, ckpt_path); print("  Saved best model.")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("  Early stopping."); break

    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd['model']); print(f"[best] loaded {ckpt_path}")

    evaluate_now(model, diff, va_loader,
                 steps_final=steps_final, guidance_scale=cfg_late,
                 repaint_R=repaint_late[0], repaint_J=repaint_late[1],
                 dc_lambda=dc_lambda, max_batches=1,
                 save_dir=os.path.join(save_dir,"eval_final"),
                 prefix=f"{int(ratio*100)}pct_final")
    return ckpt_path

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    SEGY_DIR  = r"D:\Geophysics\PHD\thesis\data\synthetic_data\shots0601_0800"
    SEGY_GLOB = "*.segy"
    RATIOS    = [0.2]

    PATCH_HW=(128,96); STRIDE_HW=(128,96)
    FILE_LIMIT=None        
    PREVIEW_ENABLE=False; PREVIEW_STEPS=0

    for r in RATIOS:
        train_one_ratio(
            ratio=r, segy_dir=SEGY_DIR, segy_glob=SEGY_GLOB,
            patch_hw=PATCH_HW, stride_hw=STRIDE_HW,
            T=1000, objective="v",
            early_epochs=5, cfg_early=1.2, repaint_early=(0,0), preview_steps_early=40,
            cfg_late=1.6, repaint_late=(0,0), preview_steps_late=60,
            epochs=14, batch_size=2, lr=2e-4, patience=5, grad_clip=1.0,
            lambda_spec=0.03, tv_lambda=0.005, p_uncond=0.10, w_miss=1.0, w_obs=0.1,
            accum_steps=4, steps_final=250, dc_lambda=0.3,
            save_dir="runs_bp2004_v2_clean", file_limit=FILE_LIMIT, num_workers=0,
            preview_every_steps=PREVIEW_STEPS, enable_preview=PREVIEW_ENABLE,
            seed=123, resume=True, base_ch=32, ax_heads=2, use_axial=True
        )
