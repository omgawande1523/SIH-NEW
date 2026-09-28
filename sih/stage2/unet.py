"""Small conditional U-Net used both as the diffusion denoiser and as the MSE baseline."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch=1, ch=(32, 64, 128), tdim=128):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(in_ch, ch[0], 3, padding=1)
        self.down = nn.ModuleList()
        c = ch[0]
        for co in ch:
            self.down.append(nn.ModuleList([ResBlock(c, co, tdim), ResBlock(co, co, tdim)]))
            c = co
        self.mid = ResBlock(c, c, tdim)
        self.up = nn.ModuleList()
        for co in reversed(ch):
            self.up.append(nn.ModuleList([ResBlock(c + co, co, tdim), ResBlock(co, co, tdim)]))
            c = co
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))

    def forward(self, x, t=None):
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)
        temb = self.temb(timestep_embedding(t, self.tdim))
        h = self.inp(x)
        skips = []
        for i, (b1, b2) in enumerate(self.down):
            h = b2(b1(h, temb), temb)
            skips.append(h)
            if i < len(self.down) - 1:
                h = F.avg_pool2d(h, 2)
        h = self.mid(h, temb)
        for i, (b1, b2) in enumerate(self.up):
            if i > 0:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = b2(b1(torch.cat([h, skips.pop()], 1), temb), temb)
        return self.out(h)
