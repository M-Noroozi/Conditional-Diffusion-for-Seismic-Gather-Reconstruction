import math
import torch
import torch.nn.functional as F

from .config import T_STEPS, DEVICE

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], -1)
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
    def __init__(self, T=T_STEPS, device=DEVICE):
        self.T = T
        self.device = device
        self.betas = make_beta_schedule_cosine(T).to(device)
        self.alphas = 1.0 - self.betas
        self.acp = torch.cumprod(self.alphas, dim=0)
        self.acp_prev = torch.cat([torch.tensor([1.0], device=device), self.acp[:-1]], dim=0)
        self.sqrt_acp = torch.sqrt(torch.clamp(self.acp, 1e-12))
        self.sqrt_om = torch.sqrt(torch.clamp(1.0 - self.acp, 1e-12))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s = self.sqrt_acp[t].view(-1, 1, 1, 1)
        o = self.sqrt_om[t].view(-1, 1, 1, 1)
        return s * x0 + o * noise

    def v_from_eps_x0(self, t, eps, x0):
        s = self.sqrt_acp[t].view(-1, 1, 1, 1)
        o = self.sqrt_om[t].view(-1, 1, 1, 1)
        return s * eps - o * x0

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
