import torch

@torch.no_grad()
def guided_v_pred(model, x, y_obs, mask, t, guidance_scale=1.6):
    vc = model(torch.cat([x, y_obs, mask], 1), t)
    vu = model(torch.cat([x, torch.zeros_like(y_obs), torch.zeros_like(mask)], 1), t)
    return vu + guidance_scale * (vc - vu)

@torch.no_grad()
def ddim_sample(model, diffusion, y_obs, mask, shape,
                steps=150, eta=0.0, guidance_scale=1.6,
                repaint_R=0, repaint_J=0, dc_lambda=0.3):
    seq = diffusion.make_sampling_schedule(steps)
    B = shape[0]
    dev = diffusion.device
    x = torch.randn(shape, device=dev)
    z_fixed = torch.randn_like(x)

    for j, t in enumerate(reversed(seq)):
        t_t = torch.full((B,), t, dtype=torch.long, device=dev)

        v = guided_v_pred(model, x, y_obs, mask, t_t, guidance_scale)
        eps = diffusion.eps_from_v_xt(t_t, v, x)
        x0 = (x - diffusion.sqrt_om[t] * eps) / diffusion.sqrt_acp[t]
        x0 = torch.clamp(x0, -1, 1)

        # PnP data consistency (only on observed)
        x0 = x0 - dc_lambda * mask * (x0 - y_obs)
        x0 = torch.clamp(x0, -1, 1)

        if j < steps - 1:
            t_prev = seq[::-1][j + 1]
            alpha_t = diffusion.acp[t]
            alpha_prev = diffusion.acp[t_prev]
            sigma = eta * torch.sqrt(
                torch.clamp((1 - alpha_prev) / (1 - alpha_t), 1e-12) *
                torch.clamp(1 - alpha_t / alpha_prev, 1e-12)
            )
            c = torch.sqrt(torch.clamp(1 - alpha_t / alpha_prev, 1e-12))
            x = torch.sqrt(alpha_prev) * x0 + c * torch.sqrt(torch.clamp(1 - alpha_prev, 1e-12)) * eps + sigma * torch.randn_like(x)
        else:
            x = x0

        x_obs_t = diffusion.project_observation_to_xt(y_obs, mask, t_t, z_fixed)
        x = (1.0 - mask) * x + x_obs_t

        # optional RePaint
        if (repaint_R > 0 and repaint_J > 0) and (j % repaint_R == 0) and (j > 0):
            for _ in range(repaint_J):
                n = torch.randn_like(x)
                s = diffusion.sqrt_acp[t].view(-1, 1, 1, 1)
                o = diffusion.sqrt_om[t].view(-1, 1, 1, 1)
                x = s * x + o * n
                x_obs_t = diffusion.project_observation_to_xt(y_obs, mask, t_t, n)
                x = (1.0 - mask) * x + x_obs_t

    return x
