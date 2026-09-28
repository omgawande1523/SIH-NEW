"""End-to-end forecast products: Stage 1 tracks -> Stage 2 downscaling -> 5 km alert grid.

Writes
  outputs/products.json  tracks, 4D boxes, cyclone plume, alert core, metadata
  outputs/fields.npz     EFI / GNN probability grids (Stage 1)
  outputs/alert_grid.npz 5 km alert-probability lattice per downscaled lead (Stage 2)
"""
from __future__ import annotations

import json

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from sih import config as C
from sih.stage1 import pipeline as S1
from sih.stage1.tracker import cyclone_centres, find_lows, track
from sih.stage2 import pipeline as S2

LEVELS = list(C.ALERT_THRESHOLDS_MM6H)  # low, moderate, severe
KM_PER_DEG = 111.2
LATTICE_KM = 5.0


def stage1_products(event, thr=0.5, log=print):
    prob, r = S1.infer(event)
    ens = xr.open_dataset(C.CACHE / f"ens_{event}.nc").load()
    tracks = track(prob, r["lat"], r["lon"], r["leads"], r["init"], S1.HAZARDS, thr,
                   lows=find_lows(ens, r["leads"]))
    for t in tracks:
        if t["hazard"] in ("heavy_rain", "damaging_wind"):
            t["cyclone_plume"] = cyclone_centres(ens, t)
    log(f"Stage 1: {len(tracks)} tracked anomalies: " +
        ", ".join(f"{t['id']} {t['hazard']} {t['envelope']['lead_start_h']}-{t['envelope']['lead_end_h']}h"
                  for t in tracks))
    np.savez_compressed(C.OUT / "fields.npz", prob=prob, efi=r["efi"], lat=r["lat"], lon=r["lon"],
                        leads=r["leads"], hazards=np.array(S1.HAZARDS))
    return tracks, ens, r


def _to_lattice(lat, lon, field):
    """Bilinear resample of a 0.25 deg field onto a ~5 km lattice."""
    step = LATTICE_KM / KM_PER_DEG
    la = np.arange(lat[0], lat[-1] + 1e-6, step)
    lo = np.arange(lon[0], lon[-1] + 1e-6, step / np.cos(np.radians(lat.mean())))
    f = RegularGridInterpolator((lat, lon), field)
    g = np.stack(np.meshgrid(la, lo, indexing="ij"), -1)
    return la, lo, f(g)


def _disc(radius_km):
    r = int(np.ceil(radius_km / LATTICE_KM))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (np.hypot(y, x) * LATTICE_KM) <= radius_km + 1e-6


def weighted_quantile(x, w, q):
    """Quantile along axis 0 of x (S, ...) with sample weights w (S,) summing to 1."""
    order = np.argsort(x, axis=0)
    xs = np.take_along_axis(x, order, axis=0)
    cw = np.cumsum(w[order], axis=0)
    idx = (cw < q).sum(axis=0, keepdims=True).clip(max=len(w) - 1)
    return np.take_along_axis(xs, idx, axis=0)[0]


def alert_probabilities(lat, lon, samples, weights, radius_km=C.ALERT_RADIUS_KM):
    """P(rain anywhere within `radius_km` of each lattice point exceeds each level).

    `weights` (one per sample, summing to 1) undo the tail-focused member selection, so the
    probabilities estimate the full 50-member ensemble rather than the wettest members.
    """
    fp = _disc(radius_km)
    maxed = []
    for s in samples:
        la, lo, f = _to_lattice(lat, lon, s)
        maxed.append(ndimage.maximum_filter(f, footprint=fp, mode="nearest"))
    maxed = np.array(maxed)
    w = np.asarray(weights, float)
    probs = np.stack([np.tensordot(w, (maxed >= C.ALERT_THRESHOLDS_MM6H[k]), axes=1) for k in LEVELS])
    return la, lo, probs, weighted_quantile(maxed, w, 0.9), np.tensordot(w, maxed, axes=1)


def pick_members(ens, lead, box, n):
    """Importance sampling of ensemble members for the (expensive) downscaling step.

    The n//2 members with the heaviest 6 h rain in the anomaly box are always kept (the tail
    matters most for alerts); the rest are evenly spaced through the remaining members,
    ranked by rain. Each tail member stands for itself (weight 1/N), and each of the others
    stands for an equal share of the remaining N - n//2 members, so weighted statistics
    are unbiased estimates for the whole ensemble.
    """
    sub = ens.tp6.sel(lead=lead, lat=slice(box[0], box[2]), lon=slice(box[1], box[3]))
    order = np.argsort(-sub.max(("lat", "lon")).values)
    N = len(order)
    n = min(n, N)  # small ensembles (e.g. the synthetic one) are used whole
    k = n // 2
    rest = order[k:]
    spread = rest[np.linspace(0, len(rest) - 1, n - k).round().astype(int)]
    members = [int(ens.member.values[i]) for i in list(order[:k]) + list(spread)]
    weights = [1.0 / N] * k + [(N - k) / N / (n - k)] * (n - k)
    return members, weights


def _headline(cores):
    """Pre-declared rule for the one headline alert (fixed before looking at the levels).

    The headline is the downscaled lead whose core has the highest ensemble-weighted
    P(severe). Leads within one Monte Carlo standard error of that maximum count as tied,
    and the earliest of them wins: it is the more skilful forecast and the one that gives
    the most warning. The level is whatever that core reads; it is never promoted.
    """
    best = max(c["prob"]["severe"] for c in cores)
    top = max(cores, key=lambda c: c["prob"]["severe"])
    tied = [c for c in cores if c["prob"]["severe"] >= best - top["prob_se"]["severe"]]
    return min(tied, key=lambda c: c["lead"])


def stage2_products(tracks, ens, n_members=24, n_samples=2, max_leads=6, log=print):
    rain = [t for t in tracks if t["hazard"] == "heavy_rain"]
    if not rain:
        log("Stage 2: no heavy-rain track to downscale")
        return None
    t = min(rain, key=lambda t: t["id"])  # tracks are ranked: tropical lows first, then severity
    steps = [s for s in t["steps"] if s["lead"] > 0]
    # downscale up to `max_leads` steps, spread over the track
    idx = np.unique(np.linspace(0, len(steps) - 1, min(max_leads, len(steps))).round().astype(int))
    out, cores = {}, []
    for i in idx:
        s = steps[i]
        centre = s["centroid"]
        if t.get("cyclone_plume"):
            pl = next((p for p in t["cyclone_plume"] if p["lead"] == s["lead"] and p["mean"]), None)
            if pl:
                centre = pl["mean"][:2]
        box = [centre[0] - 6, centre[1] - 6, centre[0] + 6, centre[1] + 6]
        members, mw = pick_members(ens, s["lead"], box, n_members)
        d = S2.downscale_forecast(ens, s["lead"], centre, members, n_samples)
        sample_w = np.repeat(mw, n_samples) / n_samples  # samples are ordered member by member
        la, lo, probs, q90, mean = alert_probabilities(d["lat"], d["lon"], d["samples"], sample_w)
        iy, ix = np.unravel_index(np.argmax(probs[-1] * 1000 + q90), q90.shape)
        c = {"lead": s["lead"], "valid": s["valid"], "lat": round(float(la[iy]), 3), "lon": round(float(lo[ix]), 3),
             "rain_q90_mm6h": round(float(q90[iy, ix]), 1),
             "prob": {k: round(float(probs[j, iy, ix]), 2) for j, k in enumerate(LEVELS)}}
        c["level"] = next((k for k in reversed(LEVELS) if c["prob"][k] >= 0.5), "none")
        # Monte Carlo standard error of each weighted probability (effective sample size)
        n_eff = 1.0 / float(np.sum(sample_w ** 2))
        c["n_members"], c["n_eff"] = len(members), round(n_eff, 1)
        c["prob_se"] = {k: round(float(np.sqrt(max(p * (1 - p), 0.25 / n_eff) / n_eff)), 2)
                        for k, p in c["prob"].items()}
        out[s["lead"]] = {"lat5": la, "lon5": lo, "probs": probs, "q90": q90, "mean": mean,
                          "fine_lat": d["lat"], "fine_lon": d["lon"], "samples": d["samples"],
                          "coarse": d["coarse"], "coarse_lat": d["coarse_lat"], "coarse_lon": d["coarse_lon"],
                          "members": np.array(members), "member_weights": np.array(mw), "lsm": d["lsm"]}
        cores.append(c)
        log(f"Stage 2: lead {s['lead']:3d}h centre {centre[0]:.1f}N {centre[1]:.1f}E -> "
            f"peak q90 {c['rain_q90_mm6h']} mm/6h, P(low/mod/severe) "
            f"{c['prob']['low']:.0%}/{c['prob']['moderate']:.0%}/{c['prob']['severe']:.0%} "
            f"(+-{c['prob_se']['severe']:.0%}), level {c['level']}")
    core = _headline(cores)
    log(f"Stage 2: headline lead +{core['lead']} h (highest P(severe), ties -> earliest), level {core['level']}")
    # Earliest actionable warning: the first downscaled lead whose core reaches any level
    # (P >= 50% for at least "low"). Reported beside the headline, never in place of it.
    first = next((c for c in sorted(cores, key=lambda c: c["lead"]) if c["level"] != "none"), None)
    if first:
        log(f"Stage 2: earliest actionable warning +{first['lead']} h, level {first['level']}")
    flat = {}
    for L, v in out.items():
        for k, a in v.items():
            flat[f"{L}/{k}"] = a
    np.savez_compressed(C.OUT / "alert_grid.npz", leads=np.array(sorted(out)), **flat)
    return {"track_id": t["id"], "leads": [int(x) for x in sorted(out)], "core": core, "first_warning": first, "cores": cores}


def build(event=C.DEMO_EVENT, log=print):
    tracks, ens, r = stage1_products(event, log=log)
    s2 = stage2_products(tracks, ens, log=log)
    if s2 is None:  # never leave an alert grid from an earlier run lying around
        (C.OUT / "alert_grid.npz").unlink(missing_ok=True)
    prod = {"event": event, "init": r["init"], "source": ens.attrs.get("source", C.SOURCE),
            "hazards": S1.HAZARDS, "alert_levels_mm6h": C.ALERT_THRESHOLDS_MM6H,
            "alert_radius_km": C.ALERT_RADIUS_KM, "tracks": tracks, "downscaling": s2}
    (C.OUT / "products.json").write_text(json.dumps(prod, indent=1))
    return prod
