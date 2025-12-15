import torch
from .config import (
    STEPS_FINAL, ETA_DDIM, GUIDE_EVAL, REPAINT_EVAL, DC_LAMBDA
)

@torch.no_grad()
def guided_v_pred(model, x, y_obs, mask, t, g):
    vc = model(torch.cat([x, y_obs, mask], 1), t)
    vu = model(torch.cat([x, torch.zeros_like(y_obs), torch.zeros_like(mask)], 1), t)
    return vu + g * (vc - vu)

@torch.no_grad()
def ddim_sample(model, diffusion, y_obs, mask, shape,
                steps=STEPS_FINAL, eta=ETA_DDIM, guidance_scale=GUIDE_EVAL,
                repaint_R=REPAINT_EVAL[0], repaint_J=REPAINT_EVAL[1],
                dc_lambda=DC_LAMBDA):
    seq = diffusion.schedule(steps)
    B = shape[0]
    dev = diffusion.device

    x = torch.randn(shape, device=dev)
    z_fixed = torch.randn_like(x)
    R, J = repaint_R, repaint_J

    for j, t in enumerate(reversed(seq)):
        tt = torch.full((B,), t, dtype=torch.long, device=dev)

        v = guided_v_pred(model, x, y_obs, mask, tt, guidance_scale)
        eps = diffusion.eps_from_v_xt(tt, v, x)
        x0 = diffusion.x0_from_v_xt(tt, v, x).clamp(-1, 1)

        # data consistency on observed region
        x0 = x0 - dc_lambda * mask * (x0 - y_obs)
        x0 = torch.clamp(x0, -1, 1)

        if j < steps - 1:
            t_prev = seq[::-1][j + 1]
            alpha_t = diffusion.acp[t]
            alpha_p = diffusion.acp[t_prev]
            c = torch.sqrt(torch.clamp(1 - alpha_t / alpha_p, 1e-12))
            x = torch.sqrt(alpha_p) * x0 + c * torch.sqrt(torch.clamp(1 - alpha_p, 1e-12)) * eps
        else:
            x = x0

        x_obs_t = diffusion.project_obs_to_xt(y_obs, mask, tt, z_fixed)
        x = (1.0 - mask) * x + x_obs_t

        # optional RePaint
        if (R > 0 and J > 0) and (j % R == 0) and (j > 0):
            for _ in range(J):
                n = torch.randn_like(x)
                s = diffusion.sqrt_acp[t].view(-1, 1, 1, 1)
                o = diffusion.sqrt_om[t].view(-1, 1, 1, 1)
                x = s * x + o * n
                x_obs_t = diffusion.project_obs_to_xt(y_obs, mask, tt, n)
                x = (1.0 - mask) * x + x_obs_t

    return x
