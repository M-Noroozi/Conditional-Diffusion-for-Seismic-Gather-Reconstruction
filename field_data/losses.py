import torch
import torch.nn.functional as F

def spectral_l1(x, y):
    x = x.squeeze(1)
    y = y.squeeze(1)
    nfft = 1 << (x.shape[1] - 1).bit_length()
    X = torch.fft.rfft(x, n=nfft, dim=1)
    Y = torch.fft.rfft(y, n=nfft, dim=1)
    sx = torch.log(torch.clamp(torch.abs(X), 1e-8))
    sy = torch.log(torch.clamp(torch.abs(Y), 1e-8))
    return F.l1_loss(sx, sy)

def total_variation(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
