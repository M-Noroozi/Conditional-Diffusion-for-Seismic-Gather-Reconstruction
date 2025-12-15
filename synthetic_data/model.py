import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import timestep_embedding

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        ng = min(8, out_ch)
        while ng > 1 and (out_ch % ng != 0):
            ng -= 1
        self.gn1 = nn.GroupNorm(max(1, ng), out_ch)
        self.gn2 = nn.GroupNorm(max(1, ng), out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.temb = nn.Linear(time_emb_dim, out_ch) if time_emb_dim is not None else None

    def forward(self, x, emb=None):
        h = self.act(self.gn1(self.conv1(x)))
        if emb is not None and self.temb is not None:
            h = h + self.temb(emb).view(emb.shape[0], -1, 1, 1)
        h = self.gn2(self.conv2(h))
        return self.act(h + self.skip(x))

class AxialMHSA(nn.Module):
    def __init__(self, ch, num_heads=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):  # [B,C,H,W] -> attn on W
        B, C, H, W = x.shape
        z = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        y, _ = self.mha(z, z, z, need_weights=False)
        y = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x + y

class ResUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, time_dim=64, time_emb_dim=256, use_axial=True, ax_heads=2):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.enc1 = nn.Sequential(
            ResBlock(in_ch, base_ch, time_emb_dim),
            ResBlock(base_ch, base_ch, time_emb_dim),
        )
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ResBlock(base_ch, base_ch * 2, time_emb_dim),
            ResBlock(base_ch * 2, base_ch * 2, time_emb_dim),
        )
        self.ax2 = AxialMHSA(base_ch * 2, ax_heads) if use_axial else nn.Identity()
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ResBlock(base_ch * 2, base_ch * 4, time_emb_dim),
            ResBlock(base_ch * 4, base_ch * 4, time_emb_dim),
        )
        self.ax3 = AxialMHSA(base_ch * 4, ax_heads) if use_axial else nn.Identity()
        self.down3 = nn.MaxPool2d(2)

        self.bot1 = ResBlock(base_ch * 4, base_ch * 8, time_emb_dim)
        self.bot2 = ResBlock(base_ch * 8, base_ch * 4, time_emb_dim)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = nn.Sequential(
            ResBlock(base_ch * 8, base_ch * 4, time_emb_dim),
            ResBlock(base_ch * 4, base_ch * 4, time_emb_dim),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            ResBlock(base_ch * 6, base_ch * 2, time_emb_dim),
            ResBlock(base_ch * 2, base_ch * 2, time_emb_dim),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            ResBlock(base_ch * 3, base_ch, time_emb_dim),
            ResBlock(base_ch, base_ch, time_emb_dim),
        )

        self.final_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x3, t):
        emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h1 = self.enc1[0](x3, emb)
        h1 = self.enc1[1](h1, emb)

        h2 = self.down1(h1)
        h2 = self.enc2[0](h2, emb)
        h2 = self.enc2[1](h2, emb)
        h2 = self.ax2(h2)

        h3 = self.down2(h2)
        h3 = self.enc3[0](h3, emb)
        h3 = self.enc3[1](h3, emb)
        h3 = self.ax3(h3)

        b = self.down3(h3)
        b = self.bot1(b, emb)
        b = self.bot2(b, emb)

        u3 = self.up3(b)
        u3 = F.interpolate(u3, size=h3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3[0](torch.cat([u3, h3], 1), emb)
        d3 = self.dec3[1](d3, emb)

        u2 = self.up2(d3)
        u2 = F.interpolate(u2, size=h2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2[0](torch.cat([u2, h2], 1), emb)
        d2 = self.dec2[1](d2, emb)

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=h1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1[0](torch.cat([u1, h1], 1), emb)
        d1 = self.dec1[1](d1, emb)

        return self.final_conv(d1)
