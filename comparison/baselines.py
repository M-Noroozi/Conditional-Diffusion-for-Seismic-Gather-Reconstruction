import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=2, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*4, base*4, 3, padding=1), nn.ReLU(True)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.bot = nn.Sequential(
            nn.Conv2d(base*4, base*8, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*8, base*4, 3, padding=1), nn.ReLU(True)
        )

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base*8, base*4, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*4, base*4, 3, padding=1), nn.ReLU(True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*6, base*2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(True)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*3, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, 1, 1)
        )

    def forward(self, x, M):
        inp = torch.cat([x, M], dim=1)
        h1 = self.enc1(inp)
        h2 = self.enc2(self.pool1(h1))
        h3 = self.enc3(self.pool2(h2))
        b  = self.bot(self.pool3(h3))

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, h3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, h2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, h1], dim=1))

        return torch.tanh(d1)

class ConvAE(nn.Module):
    def __init__(self, in_ch=2, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*4, base*4, 3, padding=1), nn.ReLU(True)
        )

        self.dec2 = nn.Sequential(nn.Conv2d(base*4, base*2, 3, padding=1), nn.ReLU(True))
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.dec1 = nn.Sequential(nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(True))
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x, M):
        inp = torch.cat([x, M], dim=1)
        h1 = self.enc1(inp)
        h2 = self.enc2(self.pool1(h1))
        h3 = self.enc3(self.pool2(h2))

        d2 = self.dec2(h3); u2 = self.up2(d2)
        d1 = self.dec1(u2); u1 = self.up1(d1)

        out = self.out_conv(u1)
        return torch.tanh(out)

def hslr_reconstruct(patch, M2d, win_t=20, rank=5):
    H, W = patch.shape
    out = patch.copy()
    m1d = (M2d[0, :] > 0.5).astype(np.uint8)

    for ix in range(W):
        if m1d[ix] == 1:
            continue
        col = patch[:, ix]
        L = 1 if H <= win_t else (H - win_t + 1)
        Hmat = np.zeros((L, win_t), dtype=np.float32)
        for i in range(L):
            Hmat[i, :] = col[i:i+win_t]

        U, S, Vt = np.linalg.svd(Hmat, full_matrices=False)
        r = min(rank, len(S))
        Hr = (U[:, :r] * S[:r]) @ Vt[:r, :]

        rec = np.zeros(H, dtype=np.float32)
        cnt = np.zeros(H, dtype=np.float32)
        for i in range(L):
            rec[i:i+win_t] += Hr[i, :]
            cnt[i:i+win_t] += 1.0
        cnt[cnt == 0] = 1.0
        out[:, ix] = rec / cnt

    return out
