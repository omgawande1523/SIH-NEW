"""Offline synthetic data with the same files and layout as the ERA5 / ECMWF mode.

Use when there is no internet (e.g. on stage at the hackathon):  python run_demo.py --source synthetic

* A 30-year "climatology" of 24 h rain, 10 m wind and 2 m temperature on the 1.5 deg grid.
* 20-member ensembles for each event in config.EVENTS: cyclones are Holland-profile vortices
  whose track, intensity and timing spread grows with lead time (loosely modelled on Amphan:
  genesis near 10N 87E, recurving north to landfall near 21.6N 88.3E); heatwaves are warm
  domes over north-west India; the "quiet" case has no signal.
* A "verifying analysis" (the true run) for training labels.
* 0.25 deg rain / wind / moisture fields with sharp convective cores and consistent
  low-level convergence, plus synthetic orography and a land-sea mask, for Stage 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from scipy import ndimage

from sih import config as C

LAT = np.arange(0.0, 36.01, 1.5)
LON = np.arange(61.5, 100.51, 1.5)
N_MEMBERS = 20
INDIA = [(68, 23.5), (70, 21), (72.8, 19), (74, 15), (76.5, 8.2), (77.5, 8), (80.3, 13), (80.2, 15.8),
         (82.3, 17), (85, 19.5), (87, 21.5), (88.5, 21.7), (90, 22), (91.8, 22.3), (92.5, 20.5), (94.2, 16),
         (97.7, 16.5), (98.5, 10), (99.5, 5), (100.6, 5), (100.6, 36.5), (60.5, 36.5), (60.5, 25), (66.5, 25.2)]


def land_mask(lat, lon):
    g = np.stack(np.meshgrid(lon, lat), -1).reshape(-1, 2)
    m = MplPath(INDIA).contains_points(g).reshape(len(lat), len(lon)).astype("float32")
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    m[((lo - 80.7) / 1.0) ** 2 + ((la - 7.8) / 1.6) ** 2 < 1] = 1.0  # Sri Lanka
    return m


def orography(lat, lon):
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    ghats = 900 * np.exp(-((lo - (73.5 + 0.12 * (20 - la))) / 0.6) ** 2) * ((la > 8) & (la < 21))
    himalaya = 4000 / (1 + np.exp(-(la - (28.5 + 0.08 * (lo - 80))) / 0.5)) * (lo > 72)
    arakan = 1200 * np.exp(-((lo - (93.5 - 0.1 * (la - 20))) / 0.7) ** 2) * ((la > 16) & (la < 26))
    return ((ghats + himalaya + arakan) * land_mask(lat, lon)).astype("float32")


def smooth_noise(rng, shape, sigma):
    n = rng.standard_normal(shape)
    axes = tuple(range(len(shape) - 2, len(shape)))
    s = [0] * (len(shape) - 2) + [sigma, sigma]
    n = ndimage.gaussian_filter(n, s)
    return n / n.std(axis=axes, keepdims=True)


def base_t2m(lat, lon, hour):
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    land = land_mask(lat, lon)
    diurnal = np.cos(2 * np.pi * (hour + 5.5 - 14) / 24)  # IST afternoon peak
    return 301 + 0.1 * la * land + land * (4 + 5 * diurnal) - 0.004 * orography(lat, lon)


def climate_fields(rng, n, hours):
    ny, nx = len(LAT), len(LON)
    t2m = np.stack([base_t2m(LAT, LON, h) for h in hours]) + 1.8 * smooth_noise(rng, (n, ny, nx), 2)
    ws = np.abs(6 + 2.5 * smooth_noise(rng, (n, ny, nx), 1.5) + 1.2 * rng.standard_normal((n, ny, nx)))
    wet = smooth_noise(rng, (n, ny, nx), 1.5) > 0.3
    tp = rng.gamma(0.8, 9.0, (n, ny, nx)) * wet / 1000.0  # m
    return tp.astype("float32"), ws.astype("float32"), t2m.astype("float32")


def cyclone(la, lo, clat, clon, vmax, rmax=0.6):
    r = np.hypot((lo - clon) * np.cos(np.radians(clat)), la - clat) + 1e-3
    x = rmax / r
    ws = vmax * np.sqrt(x ** 1.5 * np.exp(1 - x ** 1.5))  # Holland (1980), B = 1.5
    dp = (vmax / 55) ** 2 * 60 * np.exp(-(r / (5 * rmax)) ** 1.2)  # hPa
    rain = (vmax / 40) * 180 * np.exp(-(r / (4 * rmax)) ** 1.3) / 1000.0  # m / 24 h
    ang = np.arctan2(la - clat, (lo - clon))
    u, v = -np.sin(ang) * ws - 0.25 * np.cos(ang) * ws, np.cos(ang) * ws - 0.25 * np.sin(ang) * ws
    return ws, dp * 100, rain, u, v


def amphan_like_track(lead_h, jitter=np.zeros(4)):
    """Genesis ~10N 87E, north-north-east to landfall ~21.6N 88.3E near +120 h."""
    s = lead_h / 24.0
    speed = 1.0 + jitter[2]
    lat = 10 + 2.3 * s * speed + jitter[0] * s / 3
    lon = 87 - 0.5 * np.sin(s / 2) + 0.25 * s + jitter[1] * s / 3
    land_s = (21.6 - 10) / (2.3 * speed)
    vmax = np.where(s < land_s, 12 + 40 * np.sin(np.clip(s / land_s, 0, 1) * np.pi * 0.62) * (1 + jitter[3]),
                    np.maximum(40 * (1 + jitter[3]) * np.exp(-(s - land_s) * 1.2), 8))
    return lat, lon, vmax


def event_ensemble(ev, rng, n_members, truth=False):
    leads = np.array(C.LEADS_H)
    init = pd.Timestamp(ev.init)
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    nm = 1 if truth else n_members
    shape = (nm, len(leads), len(LAT), len(LON))
    hours = [(init + pd.Timedelta(hours=int(L))).hour for L in leads]
    tp, ws, t2m = [a.reshape(shape) for a in climate_fields(rng, nm * len(leads), hours * nm)]
    tp6 = tp / 4
    mslp = 100800 + 300 * smooth_noise(rng, shape, 2) - 250 * (la / 36)
    u = ws * 0.8
    v = ws * 0.3
    for m in range(nm):
        jit = np.zeros(4) if truth else rng.normal(0, [0.35, 0.35, 0.12, 0.15])
        for i, L in enumerate(leads):
            if ev.kind == "cyclone":
                clat, clon, vmax = amphan_like_track(L, jit)
                if ev.name != "amphan":  # other cyclones: shifted copies of the same life cycle
                    off = {"fani": (-2, -2), "vayu": (0, -18), "tauktae": (-1, -16), "yaas": (0, 1)}.get(ev.name, (0, 0))
                    clat, clon = clat + off[0], clon + off[1]
                w, dp, r, cu, cv = cyclone(la, lo, clat, clon, float(vmax) * (1 + 0.05 * L / 24 * (not truth) * rng.normal()))
                ws[m, i] = np.maximum(ws[m, i], w)
                mslp[m, i] -= dp
                tp[m, i] += r
                tp6[m, i] += r / 3
                u[m, i] += cu
                v[m, i] += cv
            elif ev.kind == "heatwave":
                amp = 7 * np.exp(-((L - 96) / 60) ** 2) * (1 + (0 if truth else rng.normal(0, 0.15)))
                t2m[m, i] += amp * np.exp(-(((la - 26) / 4) ** 2 + ((lo - 76) / 5) ** 2)) * land_mask(LAT, LON)
    ds = xr.Dataset({"tp24": (("member", "lead", "lat", "lon"), tp), "ws10": (("member", "lead", "lat", "lon"), ws),
                     "t2m": (("member", "lead", "lat", "lon"), t2m), "mslp": (("member", "lead", "lat", "lon"), mslp),
                     "tp6": (("member", "lead", "lat", "lon"), tp6), "u10": (("member", "lead", "lat", "lon"), u),
                     "v10": (("member", "lead", "lat", "lon"), v)},
                    coords={"member": np.arange(1, nm + 1), "lead": leads, "lat": LAT, "lon": LON})
    ds["tp24"][:, 0] = np.nan  # like the real archive: no 24 h accumulation at lead 0
    ds.attrs.update(init=ev.init, event=ev.name, kind=ev.kind, source="synthetic")
    return ds.astype("float32")


# ------------------------------------------------------------------ Stage 2 fine fields
FLAT = np.arange(C.STAGE2_DOMAIN.lat_min, C.STAGE2_DOMAIN.lat_max + 1e-6, 0.25)
FLON = np.arange(C.STAGE2_DOMAIN.lon_min, C.STAGE2_DOMAIN.lon_max + 1e-6, 0.25)


def fine_field(rng, vortex=None):
    """Rain cells with sharp cores + convergent low-level flow + moisture."""
    la, lo = np.meshgrid(FLAT, FLON, indexing="ij")
    orog = orography(FLAT, FLON)
    phi = np.zeros_like(la)
    rain = np.zeros_like(la)
    for _ in range(rng.integers(3, 12)):
        clat, clon = rng.uniform(FLAT[0], FLAT[-1]), rng.uniform(FLON[0], FLON[-1])
        size = rng.uniform(0.3, 1.5)
        blob = np.exp(-(((la - clat) / size) ** 2 + ((lo - clon) / (size * rng.uniform(0.6, 1.6))) ** 2))
        rain += rng.gamma(1.2, 18) * blob ** 2.5
        phi += blob * size
    if vortex is not None:
        clat, clon, vmax = vortex
        r = np.hypot((lo - clon), la - clat) + 1e-3
        ang = np.arctan2(la - clat, lo - clon)
        bands = 0.5 + 0.5 * np.cos(2 * ang - 2.5 * np.log(r))
        rain += (vmax / 45) * 110 * np.exp(-(r / 1.2) ** 1.4) * (0.4 + 0.8 * bands)
        phi += 3 * np.exp(-(r / 2.5) ** 2)
    rain *= 1 + 0.6 * np.clip(ndimage.sobel(orog, 1) / 800, 0, 1)  # upslope enhancement
    rain = np.maximum(rain + 2 * smooth_noise(rng, la.shape, 4) - 2, 0)
    gy, gx = np.gradient(phi)
    # low-level flow runs up the gradient of phi, i.e. converges into the rain cells
    u = 3 * rng.normal() + 6 * gx / (np.abs(gx).max() + 1e-6)
    v = 3 * rng.normal() + 6 * gy / (np.abs(gy).max() + 1e-6)
    u = u + 1.0 * smooth_noise(rng, la.shape, 6)
    v = v + 1.0 * smooth_noise(rng, la.shape, 6)
    tcwv = 45 + 12 * phi / (phi.max() + 1e-6) + 3 * smooth_noise(rng, la.shape, 8)
    return {"tp6": rain / 1000.0, "u10": u, "v10": v, "tcwv": tcwv}


def _fine_ds(fields, times):
    return xr.Dataset({k: (("time", "lat", "lon"), np.stack([f[k] for f in fields]).astype("float32"))
                       for k in fields[0]}, coords={"time": times, "lat": FLAT, "lon": FLON})


def generate(seed=0, log=print):
    C.use_source("synthetic")
    rng = np.random.default_rng(seed)
    if not (C.CACHE / "clim_samples.nc").exists():
        times = pd.DatetimeIndex([t for y in range(C.CLIM_YEARS[0], C.CLIM_YEARS[1] + 1)
                                  for t in pd.date_range(f"{y}-01-01", periods=366 * 4, freq="6h")
                                  if t.year == y and C.CLIM_DOY[0] <= t.dayofyear <= C.CLIM_DOY[1]])
        tp, ws, t2m = climate_fields(rng, len(times), list(times.hour))
        xr.Dataset({"tp24": (("time", "lat", "lon"), tp), "ws10": (("time", "lat", "lon"), ws),
                    "t2m": (("time", "lat", "lon"), t2m)},
                   coords={"time": times, "lat": LAT, "lon": LON}).to_netcdf(C.CACHE / "clim_samples.nc")
        log(f"  synthetic climatology: {len(times)} samples")
    for ev in C.EVENTS:
        if (C.CACHE / f"ens_{ev.name}.nc").exists():
            continue
        event_ensemble(ev, rng, N_MEMBERS).to_netcdf(C.CACHE / f"ens_{ev.name}.nc")
        tr = event_ensemble(ev, rng, 1, truth=True).isel(member=0)
        valid = pd.Timestamp(ev.init) + pd.to_timedelta(tr.lead.values, unit="h")
        tr = tr[["tp24", "ws10", "t2m"]].rename(lead="time").assign_coords(time=valid.values)
        tr.to_netcdf(C.CACHE / f"verif_{ev.name}.nc")
        log(f"  synthetic ensemble: {ev.name} ({ev.kind})")
    if not (C.CACHE / "fine_train.nc").exists():
        xr.Dataset({"orog": (("lat", "lon"), orography(FLAT, FLON)), "lsm": (("lat", "lon"), land_mask(FLAT, FLON))},
                   coords={"lat": FLAT, "lon": FLON}).to_netcdf(C.CACHE / "fine_static.nc")
        fields = [fine_field(rng, (rng.uniform(10, 22), rng.uniform(80, 92), rng.uniform(25, 55))
                             if rng.random() < 0.3 else None) for _ in range(C.STAGE2_TRAIN_SAMPLES)]
        _fine_ds(fields, pd.date_range("2010-06-01", periods=len(fields), freq="6h")).to_netcdf(C.CACHE / "fine_train.nc")
        init = pd.Timestamp([e for e in C.EVENTS if e.name == C.DEMO_EVENT][0].init)
        leads = np.arange(72, 169, 6)
        truth = [fine_field(rng, (*amphan_like_track(L)[:2], float(amphan_like_track(L)[2]))) for L in leads]
        _fine_ds(truth, init + pd.to_timedelta(leads, unit="h")).to_netcdf(C.CACHE / f"fine_truth_{C.DEMO_EVENT}.nc")
        log("  synthetic 0.25 deg training and truth fields written")


if __name__ == "__main__":
    generate()
