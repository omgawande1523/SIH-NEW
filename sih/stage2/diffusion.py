"""Stage 2: physics-informed conditional diffusion downscaling (coarse -> fine rainfall).

* Denoiser: conditional U-Net, v-prediction, cosine noise schedule, DDIM sampling.
* Condition: the coarse rainfall and 10 m winds interpolated to the fine grid, plus
  fine-scale orography and land-sea mask (the "regional topography" the model learns from).
* Physics-informed loss, applied to the model's clean-field estimate x0_hat:
    1. Water conservation: block-averaging the generated fine field must give back the
       coarse rainfall (no rain created or lost by downscaling).
    2. Moisture-convergence consistency: heavy rain generated where the synoptic-scale
       moisture flux  div(TCWV * V10)  is divergent is penalised ("downpour without
       convergence"). In ERA5, 77% of heavy-rain area sits in convergent air against
       ~54% of all area, so the constraint is physically informative.
* Baseline: the same U-Net trained with plain MSE, which shows the spectral smoothing
  (blurred, weakened peaks) that the diffusion model avoids.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from sih import config as C
from sih.stage2.unet import UNet

K = C.DOWNSCALE_FACTOR
P = C.PATCH
T = 200
LOG_SCALE = 2.65  # log1p(200 mm) / 2


def to_x(mm):
    return torch.log1p(mm.clamp(min=0)) / LOG_SCALE - 1


def to_mm(x):
    return torch.expm1(((x + 1) * LOG_SCALE).clamp(max=7)).clamp(min=0)


def cosine_alpha_bar(t):
    s = 0.008
    return torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2 / math.cos(s / (1 + s) * math.pi / 2) ** 2


# ---------------------------------------------------------------- physics
def moisture_flux_divergence(tcwv, u, v, lat):
    """div(TCWV * V) in mm per 6 h (positive = divergent, i.e. moisture leaving)."""
    dy = 0.25 * 111e3
    dx = dy * torch.cos(torch.deg2rad(lat))[None, None, :, None]
    qu, qv = tcwv * u, tcwv * v
    dqu = (F.pad(qu, (1, 1, 0, 0), mode="replicate")[..., 2:] - F.pad(qu, (1, 1, 0, 0), mode="replicate")[..., :-2]) / (2 * dx)
    dqv = (F.pad(qv, (0, 0, 1, 1), mode="replicate")[..., 2:, :] - F.pad(qv, (0, 0, 1, 1), mode="replicate")[..., :-2, :]) / (2 * dy)
    div = (dqu + dqv) * 21600.0
    # smooth to the synoptic scale (13 x 13 cells ~ 3 deg): 10 m winds are noisy at 0.25 deg
    return F.avg_pool2d(F.pad(div, (6, 6, 6, 6), mode="replicate"), 13, 1)


def physics_losses(pred_mm, coarse_mm, mfd):
    cons = (F.avg_pool2d(pred_mm, K) - coarse_mm).abs().mean(dim=(1, 2, 3)) / 5.0
    div_w = (mfd / 5.0).clamp(0, 1)
    conv = (F.relu(pred_mm - 5.0) / 20.0 * div_w).mean(dim=(1, 2, 3))
    return cons, conv


def violation_index(mm, mfd, thr=16.0):
    """Share of heavy-rain area (>thr mm/6h) sitting in moisture-divergent air."""
    heavy = mm > thr
    if heavy.sum() == 0:
        return float("nan")
    return float((heavy & (mfd > 0.0)).sum() / heavy.sum())


# ---------------------------------------------------------------- data
class FineData:
    def __init__(self, path, static_path):
        ds = xr.open_dataset(path).load()
        st = xr.open_dataset(static_path).load()
        self.lat = torch.tensor(ds.lat.values, dtype=torch.float32)
        self.lon = ds.lon.values
        self.tp = torch.tensor(np.nan_to_num(ds.tp6.values) * 1000.0)  # mm / 6 h
        self.u = torch.tensor(np.nan_to_num(ds.u10.values))
        self.v = torch.tensor(np.nan_to_num(ds.v10.values))
        self.tcwv = torch.tensor(np.nan_to_num(ds.tcwv.values))
        self.orog = torch.tensor(st.orog.values).clamp(min=0) / 2000.0
        self.lsm = torch.tensor(st.lsm.values)
        self.times = ds.time.values
        self.mfd = torch.cat([moisture_flux_divergence(self.tcwv[i:i + 32, None], self.u[i:i + 32, None],
                                                       self.v[i:i + 32, None], self.lat)[:, 0]
                              for i in range(0, len(self.tp), 32)])

    def crop(self, ti, y0, x0):
        s = (slice(y0, y0 + P), slice(x0, x0 + P))
        fine = self.tp[ti][s][None]
        cond = make_condition(F.avg_pool2d(fine[None], K)[0],
                              F.avg_pool2d(self.u[ti][s][None, None], K)[0],
                              F.avg_pool2d(self.v[ti][s][None, None], K)[0],
                              self.orog[s], self.lsm[s])
        return fine, cond, self.mfd[ti][s][None], self.lat[s[0]]

    def batch(self, n, rng, rainy=0.7):
        ny, nx = self.tp.shape[1:]
        out = []
        while len(out) < n:
            ti = rng.integers(len(self.tp))
            y0, x0 = rng.integers(0, ny - P + 1), rng.integers(0, nx - P + 1)
            fine, cond, mfd, _ = self.crop(ti, y0, x0)
            if rng.random() < rainy and fine.max() < 10:
                continue
            out.append((fine, cond, mfd))
        f, c, m = map(torch.stack, zip(*out))
        return f, c, m


def make_condition(coarse_mm, coarse_u, coarse_v, orog, lsm):
    """Coarse fields (1, h, w) + fine statics (P, P) -> condition (5, P, P)."""
    up = lambda a: F.interpolate(a[None], size=(P, P), mode="bilinear", align_corners=False)[0]
    return torch.cat([to_x(up(coarse_mm)), up(coarse_u) / 10, up(coarse_v) / 10, orog[None], lsm[None]], 0)


# ---------------------------------------------------------------- training
def train_diffusion(data: FineData, steps=3000, batch=24, lam_cons=1.0, lam_conv=1.0, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UNet(in_ch=6)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
    log(f"Stage 2 diffusion: params={sum(p.numel() for p in model.parameters())/1e6:.2f}M steps={steps}")
    t0 = time.time()
    for it in range(steps):
        fine, cond, mfd = data.batch(batch, rng)
        x0 = to_x(fine)
        t = torch.randint(0, T, (batch,))
        ab = cosine_alpha_bar(t.float() + 1)[:, None, None, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        v_target = ab.sqrt() * eps - (1 - ab).sqrt() * x0
        v = model(torch.cat([xt, cond], 1), t)
        l_mse = F.mse_loss(v, v_target)
        x0_hat = (ab.sqrt() * xt - (1 - ab).sqrt() * v).clamp(-1, 1.6)
        cons, conv = physics_losses(to_mm(x0_hat), F.avg_pool2d(fine, K), mfd)
        w = ab.flatten()  # trust the clean estimate only at low noise
        loss = l_mse + lam_cons * (w * cons).mean() + lam_conv * (w * conv).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 250 == 0 or it == steps - 1:
            log(f"  step {it:5d} loss {loss.item():.4f} (v-mse {l_mse.item():.4f} cons {cons.mean().item():.3f} "
                f"conv {conv.mean().item():.4f}) {time.time()-t0:.0f}s")
    return model


def train_mse_baseline(data: FineData, steps=1500, batch=24, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    model = UNet(in_ch=5)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for it in range(steps):
        fine, cond, _ = data.batch(batch, rng)
        loss = F.mse_loss(model(cond), to_x(fine))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == steps - 1:
            log(f"  MSE U-Net step {it:5d} loss {loss.item():.4f} {time.time()-t0:.0f}s")
    return model


# ---------------------------------------------------------------- sampling
@torch.no_grad()
def sample(model, cond, n_steps=30, generator=None):
    """DDIM sampling. cond: (B, 5, P, P) -> rainfall mm (B, 1, P, P)."""
    model.eval()
    B = cond.shape[0]
    x = torch.randn(B, 1, P, P, generator=generator)
    ts = torch.linspace(T - 1, 0, n_steps).round().long()
    for i, t in enumerate(ts):
        ab = cosine_alpha_bar(t.float() + 1)
        v = model(torch.cat([x, cond], 1), t.repeat(B))
        x0 = (ab.sqrt() * x - (1 - ab).sqrt() * v).clamp(-1, 1.6)
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        ab_next = cosine_alpha_bar(ts[i + 1].float() + 1) if i + 1 < len(ts) else torch.tensor(1.0)
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
    return to_mm(x)


@torch.no_grad()
def predict_mse(model, cond):
    model.eval()
    return to_mm(model(cond))


def bilinear(cond):
    return to_mm(cond[:, :1])
