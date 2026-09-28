"""Stage 2 driver: train, evaluate (peak preservation), and downscale a forecast anomaly."""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from sih import config as C
from sih.stage2 import diffusion as D
from sih.stage2.unet import UNet

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def train(steps=3000, mse_steps=1500, log=print):
    torch.set_num_threads(4)
    data = D.FineData(C.CACHE / "fine_train.nc", C.CACHE / "fine_static.nc")
    log(f"Stage 2 data: {len(data.tp)} fields of {data.tp.shape[1]}x{data.tp.shape[2]} at 0.25 deg")
    dm = D.train_diffusion(data, steps=steps, log=log)
    torch.save(dm.state_dict(), C.MODELS / "stage2_diffusion.pt")
    mm = D.train_mse_baseline(data, steps=mse_steps, log=log)
    torch.save(mm.state_dict(), C.MODELS / "stage2_mse_unet.pt")


def load_variant(name):
    p = C.MODELS / f"stage2_diffusion_{name}.pt"
    if not p.exists():
        return None
    m = UNet(in_ch=6)
    m.load_state_dict(torch.load(p))
    return m.eval()


def load():
    dm, mm = UNet(in_ch=6), UNet(in_ch=5)
    dm.load_state_dict(torch.load(C.MODELS / "stage2_diffusion.pt"))
    mm.load_state_dict(torch.load(C.MODELS / "stage2_mse_unet.pt"))
    return dm.eval(), mm.eval()


def psd(field):
    """Radially averaged power spectrum of a 2D field."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(field - field.mean()))) ** 2
    n = field.shape[0]
    y, x = np.indices(f.shape)
    r = np.hypot(x - n // 2, y - n // 2).astype(int)
    return np.bincount(r.ravel(), f.ravel())[1:n // 2] / np.bincount(r.ravel())[1:n // 2]


def _crps(samples, y):
    """Ensemble CRPS per pixel (lower is better); equals MAE for a single field."""
    s = samples.reshape(len(samples), -1)
    y = y.reshape(-1)
    t1 = (s - y[None]).abs().mean(0)
    t2 = (s[:, None] - s[None]).abs().mean((0, 1)) / 2 if len(s) > 1 else 0
    return float((t1 - t2).mean())


def evaluate(event=C.DEMO_EVENT, n_samples=4, extra_diffusion=None, log=print):
    """Perfect-coarse test: coarsen ERA5 truth during the event, downscale, compare to truth.

    Every diffusion sample is scored and the per-sample scores are averaged; the ensemble
    of samples is also scored with CRPS. `extra_diffusion` ({name: model}) adds variants,
    e.g. a model trained without the physics loss.
    """
    dm, mm = load()
    diff_models = {"diffusion": dm, **(extra_diffusion or {})}
    truth = D.FineData(C.CACHE / f"fine_truth_{event}.nc", C.CACHE / "fine_static.nc")
    rows, examples = [], []
    ny, nx = truth.tp.shape[1:]
    for ti in range(len(truth.tp)):
        iy, ix = np.unravel_index(int(truth.tp[ti].argmax()), (ny, nx))
        y0 = int(np.clip(iy - C.PATCH // 2, 0, ny - C.PATCH))
        x0 = int(np.clip(ix - C.PATCH // 2, 0, nx - C.PATCH))
        fine, cond, mfd, _ = truth.crop(ti, y0, x0)
        if fine.max() < 20:
            continue
        t = fine[0]
        fields = {"bilinear": D.bilinear(cond[None])[:, 0], "mse_unet": D.predict_mse(mm, cond[None])[:, 0]}
        for name, model in diff_models.items():
            g = torch.Generator().manual_seed(1000 + ti)  # same noise for every variant
            fields[name] = D.sample(model, cond[None].repeat(n_samples, 1, 1, 1), generator=g)[:, 0]
        row = {"time": str(truth.times[ti])[:16], "truth_max": float(t.max()),
               "truth_violation": D.violation_index(t, mfd[0])}
        c_true = F.avg_pool2d(t[None, None], C.DOWNSCALE_FACTOR)
        t99 = float(torch.quantile(t.flatten(), 0.99))
        for k, fs in fields.items():
            per = []
            for p in fs:  # score each sample, then average
                c_pred = F.avg_pool2d(p[None, None], C.DOWNSCALE_FACTOR)
                per.append([float(p.max()) / float(t.max()),
                            float(torch.quantile(p.flatten(), 0.99)) / t99,
                            float(((p - t) ** 2).mean().sqrt()),
                            D.violation_index(p, mfd[0]),
                            float((c_pred - c_true).abs().sum() / c_true.sum().clamp(min=1e-6))])
            per = np.nanmean(np.array(per, float), axis=0)
            for name, v in zip(("peak_ratio", "p99_ratio", "rmse", "violation", "mass_error"), per):
                row[f"{k}_{name}"] = float(v)
            row[f"{k}_crps"] = _crps(fs, t)
        rows.append(row)
        examples.append((row["truth_max"], ti, y0, x0, cond, t,
                         {"bilinear": fields["bilinear"][0], "mse_unet": fields["mse_unet"][0]},
                         fields["diffusion"], mfd[0]))

    def agg(k):
        return float(np.nanmean(np.array([r[k] for r in rows], float)))

    summary = {"n_cases": len(rows), "n_diffusion_samples_per_case": n_samples,
               "truth_peak_mm6h": round(agg("truth_max"), 1),
               "truth_heavy_rain_in_divergent_air": round(agg("truth_violation"), 3)}
    for k in ("bilinear", "mse_unet", *diff_models):
        summary[k] = {
            "peak_ratio": round(agg(f"{k}_peak_ratio"), 3),
            "p99_ratio": round(agg(f"{k}_p99_ratio"), 3),
            "rmse_mm6h": round(agg(f"{k}_rmse"), 2),
            "crps_mm6h": round(agg(f"{k}_crps"), 3),
            "heavy_rain_in_divergent_air": round(agg(f"{k}_violation"), 3),
            "coarse_mass_error": round(agg(f"{k}_mass_error"), 3),
        }
    log("Stage 2 held-out evaluation (Amphan, ERA5 0.25 deg truth):\n" + json.dumps(summary, indent=1))
    (C.OUT / "stage2_metrics.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2))
    examples.sort(key=lambda e: -e[0])
    _figure(examples[:2], truth, C.OUT / "stage2_downscaling_comparison.png")
    return summary


def _figure(examples, truth, path):
    cols = ["coarse input (1.5°)", "bilinear", "MSE U-Net", "diffusion sample 1", "diffusion sample 2", "ERA5 truth (0.25°)"]
    fig, ax = plt.subplots(len(examples) + 1, len(cols), figsize=(3.1 * len(cols), 3.1 * (len(examples) + 1)))
    vmax = max(e[5].max().item() for e in examples)
    for r, (_, ti, y0, x0, cond, t, preds, samp, _) in enumerate(examples):
        coarse = F.avg_pool2d(t[None, None], C.DOWNSCALE_FACTOR)[0, 0]
        coarse = F.interpolate(coarse[None, None], scale_factor=C.DOWNSCALE_FACTOR, mode="nearest")[0, 0]
        fields = [coarse, preds["bilinear"], preds["mse_unet"], samp[0], samp[1], t]
        lsm = truth.lsm[y0:y0 + C.PATCH, x0:x0 + C.PATCH]
        for c, (f, name) in enumerate(zip(fields, cols)):
            a = ax[r, c]
            im = a.imshow(f.numpy(), origin="lower", cmap="turbo", vmin=0, vmax=vmax)
            a.contour(lsm.numpy(), [0.5], colors="w", linewidths=0.6)
            a.set_title(f"{name}\nmax {f.max():.0f} mm/6h", fontsize=9)
            a.set_xticks([]), a.set_yticks([])
        ax[r, 0].set_ylabel(str(truth.times[ti])[:13], fontsize=9)
    fig.colorbar(im, ax=ax[:-1].ravel().tolist(), shrink=0.6, label="rain (mm / 6 h)")
    gs = ax[-1, 0].get_gridspec()
    for a in ax[-1]:
        a.remove()
    a = fig.add_subplot(gs[-1, :3])
    _, _, _, _, _, t, preds, samp, _ = examples[0]
    k = np.arange(1, C.PATCH // 2) / (C.PATCH * 0.25 * 111)
    a.loglog(k, psd(t.numpy()), "k", lw=2, label="ERA5 truth")
    a.loglog(k, psd(preds["bilinear"].numpy()), label="bilinear")
    a.loglog(k, psd(preds["mse_unet"].numpy()), label="MSE U-Net")
    a.loglog(k, np.mean([psd(s.numpy()) for s in samp], 0), label="diffusion (mean of samples)")
    a.set_xlabel("wavenumber (1/km)"), a.set_ylabel("power"), a.legend(fontsize=8)
    a.set_title("Power spectrum: MSE training drops small-scale energy (spectral smoothing)", fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def downscale_forecast(ens: xr.Dataset, lead: int, centre, members, n_samples=2, seed=0):
    """Downscale forecast rain around `centre` (lat, lon) for the given ensemble members.

    Returns fine lat/lon, samples (members*n_samples, P, P) in mm/6h, and the coarse field.
    """
    dm, _ = load()
    st = xr.open_dataset(C.CACHE / "fine_static.nc").load()
    flat, flon = st.lat.values, st.lon.values
    iy, ix = np.abs(flat - centre[0]).argmin(), np.abs(flon - centre[1]).argmin()
    y0 = int(np.clip(iy - C.PATCH // 2, 0, len(flat) - C.PATCH))
    x0 = int(np.clip(ix - C.PATCH // 2, 0, len(flon) - C.PATCH))
    plat, plon = flat[y0:y0 + C.PATCH], flon[x0:x0 + C.PATCH]
    # coarse block centres the model was trained on, then bilinear to the fine patch
    clat = plat.reshape(-1, C.DOWNSCALE_FACTOR).mean(1)
    clon = plon.reshape(-1, C.DOWNSCALE_FACTOR).mean(1)
    sub = ens.sel(lead=lead).interp(lat=clat, lon=clon)
    orog = torch.tensor(st.orog.values[y0:y0 + C.PATCH, x0:x0 + C.PATCH]).clamp(min=0) / 2000
    lsm = torch.tensor(st.lsm.values[y0:y0 + C.PATCH, x0:x0 + C.PATCH])
    conds, coarse = [], []
    for m in members:
        s = sub.sel(member=m)
        c_mm = torch.tensor(np.nan_to_num(s.tp6.values) * 1000.0, dtype=torch.float32)[None]
        conds.append(D.make_condition(c_mm, torch.tensor(s.u10.values)[None].float(),
                                      torch.tensor(s.v10.values)[None].float(), orog, lsm))
        coarse.append(c_mm[0].numpy())
    cond = torch.stack(conds).repeat_interleave(n_samples, 0)
    g = torch.Generator().manual_seed(seed)
    samples = D.sample(dm, cond, generator=g)[:, 0].numpy()
    return {"lat": plat, "lon": plon, "samples": samples, "coarse": np.array(coarse),
            "coarse_lat": clat, "coarse_lon": clon, "lsm": lsm.numpy()}
