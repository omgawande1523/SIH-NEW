"""Extreme Forecast Index (EFI) and Shift Of Tails (SOT) against a reanalysis climate.

EFI (Lalaurette 2003; Zsoter 2006) measures how far the ensemble CDF F_f sits from the
climate CDF across the whole distribution:

    EFI = 2/pi * integral_0^1 (p - F_f(p)) / sqrt(p (1 - p)) dp

where F_f(p) is the fraction of ensemble members below the climate p-quantile.
EFI = +1 means every member exceeds the climate maximum, -1 the minimum, 0 normal (on the
101-quantile grid used here the end points are left out, so the extremes are about +-0.91).

SOT(90) = -(Q_f(0.90) - Q_c(0.99)) / (Q_c(0.90) - Q_c(0.99)) says how far the upper
tail of the ensemble goes beyond the 99th climate percentile (> 0 means beyond it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from sih import config as C

P = np.asarray(C.CLIM_QUANTILES)
_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy >= 2.0 renamed trapz
# Precipitation below this (m) is treated as dry; dry climate quantiles don't count.
DRY = {"tp24": 1e-4, "tp6": 1e-4}


def feature_key(ens_path) -> str:
    """Everything the cached features depend on; a change forces a recompute."""
    import os
    size = os.path.getsize(ens_path) if os.path.exists(ens_path) else -1
    return "|".join(map(str, [C.CLIM_YEARS, C.CLIM_DOY, C.CLIM_HALF_WINDOW_DAYS, len(C.CLIM_QUANTILES),
                              list(C.STAGE1_VARS), C.LEADS_H, size, 2]))


def cached_features_ok(name) -> bool:
    path = C.CACHE / f"feats_{name}.npz"
    if not path.exists():
        return False
    with np.load(path, allow_pickle=True) as z:
        return "key" in z.files and str(z["key"]) == feature_key(C.CACHE / f"ens_{name}.nc")


def climate_quantiles(clim: xr.Dataset, valid: pd.Timestamp, var: str) -> np.ndarray:
    """Quantiles (101, lat, lon) from the +-15-day, same-hour 30-year ERA5 sample."""
    t = pd.DatetimeIndex(clim.time.values)
    d = np.abs(((t.dayofyear - valid.dayofyear + 182) % 365) - 182)
    sel = (d <= C.CLIM_HALF_WINDOW_DAYS) & (t.hour == valid.hour)
    x = clim[var].values[sel]
    x = x[np.isfinite(x).all(axis=(1, 2))]
    return np.quantile(x, P, axis=0).astype("float32")


def efi(ens: np.ndarray, q: np.ndarray, var: str) -> np.ndarray:
    """ens: (member, lat, lon); q: (101, lat, lon) -> EFI (lat, lon) in [-1, 1]."""
    Ff = (ens[None, :, :, :] < q[:, None, :, :]).mean(axis=1)  # (101, lat, lon)
    p = P[:, None, None]
    w = np.zeros_like(P)
    w[1:-1] = 1.0 / np.sqrt(P[1:-1] * (1 - P[1:-1]))
    integrand = (p - Ff) * w[:, None, None]
    if var in DRY:  # ignore the dry part of the climate distribution
        integrand = np.where(q > DRY[var], integrand, 0.0)
    out = 2 / np.pi * _trapz(integrand, P, axis=0)
    return np.clip(out, -1, 1).astype("float32")


def sot(ens: np.ndarray, q: np.ndarray, upper=True) -> np.ndarray:
    if upper:
        qf, c_hi, c_mid = np.quantile(ens, 0.9, axis=0), q[99], q[90]
    else:  # lower tail (cold waves)
        qf, c_hi, c_mid = np.quantile(ens, 0.1, axis=0), q[1], q[10]
    den = c_mid - c_hi
    den = np.where(np.abs(den) < 1e-6, np.sign(den + 1e-12) * 1e-6, den)
    return np.clip(-(qf - c_hi) / den, -2, 5).astype("float32")


def ensemble_features(ens_ds: xr.Dataset, clim: xr.Dataset, verif: xr.Dataset | None = None):
    """Per-lead, per-gridpoint features and optional observed-extreme labels.

    Returns dict with
      feats  (lead, lat, lon, F)  EFI, SOT, standardized mean, spread, P(>q99), P(<q01) per var
      efi    (lead, var, lat, lon)
      labels (lead, var, lat, lon) 1 if the verifying ERA5 exceeded its climate q99 (else 0),
             NaN where unavailable.
    """
    init = pd.Timestamp(ens_ds.attrs["init"])
    leads = ens_ds.lead.values
    vars_ = list(C.STAGE1_VARS)
    nl, ny, nx = len(leads), ens_ds.sizes["lat"], ens_ds.sizes["lon"]
    feats = np.zeros((nl, ny, nx, 6 * len(vars_)), "float32")
    efis = np.zeros((nl, len(vars_), ny, nx), "float32")
    labels = np.full((nl, len(vars_), ny, nx), np.nan, "float32")
    qcache = {}
    for i, L in enumerate(leads):
        valid = init + pd.Timedelta(hours=int(L))
        for j, v in enumerate(vars_):
            key = (valid.dayofyear, valid.hour, v)
            if key not in qcache:
                qcache[key] = climate_quantiles(clim, valid, v)
            q = qcache[key]
            ens = ens_ds[v].isel(lead=i).values
            if not np.isfinite(ens).all():  # e.g. 24 h precip at lead 0
                continue
            e = efi(ens, q, v)
            scale = np.maximum(q[90] - q[10], 1e-4)
            f = [e, sot(ens, q) / 5,
                 np.clip((ens.mean(0) - q[50]) / scale, -5, 5) / 5,
                 np.clip(ens.std(0) / scale, 0, 5) / 5,
                 (ens > q[99]).mean(0), (ens < q[1]).mean(0)]
            feats[i, :, :, 6 * j:6 * j + 6] = np.stack(f, -1)
            efis[i, j] = e
            if verif is not None:
                obs = verif[v].sel(time=valid.to_datetime64()).values
                if np.isfinite(obs).all():
                    labels[i, j] = (obs > q[99]).astype("float32")
    return {"feats": feats, "efi": efis, "labels": labels, "leads": leads, "vars": vars_,
            "lat": ens_ds.lat.values, "lon": ens_ds.lon.values, "init": str(init.date())}
