import math
import torch
import torch.nn.functional as F

from .config import DEVICE

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def make_beta_schedule_cosine(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float32)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi * 0.5) ** 2
    abar = (f / f[0]).clamp(min=1e-8)
    betas = (1 - abar[1:] / abar[:-1]).clamp(min=1e-8, max=0.999)
    return betas

class Diffusion:
    def __init__(self, T=1000, device=DEVICE):
        self.T = T
        self.device = device
        self.betas = make_beta_schedule_cosine(T).to(device)
        self.alphas = 1.0 - self.betas
        self.acp = torch.cumprod(self.alphas, dim=0)
        self.acp_prev = torch.cat([torch.tensor([1.0], device=device), self.acp[:-1]], dim=0)
        self.sqrt_acp = torch.sqrt(torch.clamp(self.acp, 1e-12))
        self.sqrt_om = torch.sqrt(torch.clamp(1.0 - self.acp, 1e-12))

    def eps_from_v_xt(self, t, v, x):
        s = self.sqrt_acp[t].view(-1, 1, 1, 1)
        o = self.sqrt_om[t].view(-1, 1, 1, 1)
        return o * x + s * v

    def x0_from_v_xt(self, t, v, x):
        s = self.sqrt_acp[t].view(-1, 1, 1, 1)
        o = self.sqrt_om[t].view(-1, 1, 1, 1)
        return s * x - o * v

    def schedule(self, steps):
        return torch.linspace(0, self.T - 1, steps, dtype=torch.long, device=self.device).tolist()

    def project_obs_to_xt(self, y_obs, mask, t, fixed_noise):
        s = self.sqrt_acp[t].view(-1, 1, 1, 1)
        o = self.sqrt_om[t].view(-1, 1, 1, 1)
        y_t = s * y_obs + o * fixed_noise
        return mask * y_t

@torch.no_grad()
def guided_v_pred(model, x, y_obs, mask, t, g):
    vc = model(torch.cat([x, y_obs, mask], 1), t)
    vu = model(torch.cat([x, torch.zeros_like(y_obs), torch.zeros_like(mask)], 1), t)
    return vu + g * (vc - vu)

@torch.no_grad()
def diffusion_reconstruct(model, diffusion, y_obs, mask, clean_shape, steps, guidance_scale, dc_lambda):
    seq = diffusion.schedule(steps)
    B = clean_shape[0]
    x = torch.randn(clean_shape, device=DEVICE)
    zfix = torch.randn_like(x)
    rev_seq = list(reversed(seq))

    for j, t in enumerate(rev_seq):
        tt = torch.full((B,), t, dtype=torch.long, device=DEVICE)
        v = guided_v_pred(model, x, y_obs, mask, tt, guidance_scale)
        x0 = diffusion.x0_from_v_xt(tt, v, x).clamp(-1, 1)

        # data consistency
        x0 = x0 - dc_lambda * mask * (x0 - y_obs)
        x0 = torch.clamp(x0, -1, 1)

        if j < steps - 1:
            t_prev = rev_seq[j + 1]
            alpha_t = diffusion.acp[t]
            alpha_p = diffusion.acp[t_prev]
            c = torch.sqrt(torch.clamp(1 - alpha_t / alpha_p, 1e-12))
            eps = diffusion.eps_from_v_xt(tt, v, x)
            x = torch.sqrt(alpha_p) * x0 + c * torch.sqrt(torch.clamp(1 - alpha_p, 1e-12)) * eps
        else:
            x = x0

        x_obs_t = diffusion.project_obs_to_xt(y_obs, mask, tt, zfix)
        x = (1.0 - mask) * x + x_obs_t

    return x
