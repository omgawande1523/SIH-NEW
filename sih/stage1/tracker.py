"""Turn per-lead hazard probabilities into tracked anomaly objects with 4D bounding boxes.

1. Threshold the GNN probability for each hazard at each lead time.
2. Label connected regions (8-connectivity) -> objects with bbox, centroid, peak, area.
   Rain and wind regions that touch a tropical low are split: cells within
   CYCLONE_RADIUS_KM of the low form a cyclone object, and the rest (e.g. the monsoon
   westerlies feeding the storm) become separate objects. Without this the storm and the
   monsoon wind field merge into one region spanning half the domain.
3. Link objects across consecutive leads (same hazard, nearest centroid within a gate;
   cyclone objects prefer cyclone objects).
4. For cyclone-type tracks, locate each ensemble member's MSLP minimum near the object
   to give a track plume (spaghetti) and an ensemble-mean centre.
A track's 4D box is the list of (valid time, lat/lon box) plus its space-time envelope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from sih.stage1.mesh import R_EARTH_KM

GATE_KM = 700.0  # max centroid jump between linked 6-hourly objects
MIN_SEVERITY = 3.0  # drop short, weak, tiny tracks (sum of peak prob x area / 10)
CYCLONE_HAZARDS = ("heavy_rain", "damaging_wind")
CYCLONE_RADIUS_KM = 600.0  # cells this close to a low belong to the cyclone object
LOW_TOUCH_KM = 300.0  # a region must come this close to a low to be split around it
LOW_DEPTH_HPA = 3.0  # ensemble-mean MSLP below its ~20 deg running mean
LOW_MIN_WIND = 9.0  # m/s ensemble-mean 10 m wind near the low (excludes heat lows)
TYPE_CHANGE_KM = 500.0  # linking penalty between cyclone and non-cyclone objects


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def find_lows(ens, leads):
    """Tropical lows per lead from the ensemble-mean MSLP: {lead: [(lat, lon), ...]}.

    A low is a local minimum at least LOW_DEPTH_HPA below its surroundings with ensemble-mean
    10 m wind of LOW_MIN_WIND or more nearby, which keeps dry heat lows out.
    """
    if ens is None or "mslp" not in ens or "ws10" not in ens:
        return {}
    lat, lon = ens.lat.values, ens.lon.values
    out = {}
    for L in leads:
        f = ens.mslp.sel(lead=L).mean("member").values / 100.0
        w = ens.ws10.sel(lead=L).mean("member").values
        anom = f - ndimage.uniform_filter(f, 15, mode="nearest")
        wmax = ndimage.maximum_filter(w, 5, mode="nearest")
        m = (f == ndimage.minimum_filter(f, 5, mode="nearest")) & (anom <= -LOW_DEPTH_HPA) & (wmax >= LOW_MIN_WIND)
        out[int(L)] = [(float(lat[iy]), float(lon[ix])) for iy, ix in zip(*np.where(m))]
    return out


def _object(prob2d, lat, lon, iy, ix, low=None):
    w = prob2d[iy, ix]
    o = {
        "bbox": [float(lat[iy.min()]), float(lon[ix.min()]), float(lat[iy.max()]), float(lon[ix.max()])],
        "centroid": [float((lat[iy] * w).sum() / w.sum()), float((lon[ix] * w).sum() / w.sum())],
        "peak_prob": float(w.max()),
        "area_cells": int(len(iy)),
    }
    if low is not None:
        o["low"] = [round(low[0], 2), round(low[1], 2)]
    return o


def detect(prob2d, lat, lon, thr=0.5, min_cells=2, lows=()):
    mask = prob2d >= thr
    objs = []
    if len(lows):
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        dist = np.stack([haversine_km(la, lo, a, b) for a, b in lows])  # (low, lat, lon)
        nearest = dist.argmin(0)
        lab, _ = ndimage.label(mask, structure=np.ones((3, 3)))
        taken = np.zeros_like(mask)
        for k, low in enumerate(lows):
            touching = np.unique(lab[mask & (dist[k] <= LOW_TOUCH_KM)])
            region = np.isin(lab, touching[touching > 0]) & (dist[k] <= CYCLONE_RADIUS_KM) & (nearest == k)
            if not region.any():
                continue
            rl, _ = ndimage.label(region, structure=np.ones((3, 3)))
            core = rl == rl.flat[np.argmin(np.where(region, dist[k], np.inf))]  # piece nearest the low
            iy, ix = np.where(core)
            if len(iy) >= min_cells:
                objs.append(_object(prob2d, lat, lon, iy, ix, low))
                taken |= core
        mask = mask & ~taken
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    for k in range(1, n + 1):
        iy, ix = np.where(lab == k)
        if len(iy) >= min_cells:
            objs.append(_object(prob2d, lat, lon, iy, ix))
    return objs


def track(prob, lat, lon, leads, init, hazards, thr=0.5, lows=None):
    """prob: (lead, hazard, lat, lon) -> list of tracks. `lows` from find_lows()."""
    lows = lows or {}
    tracks = []
    for h, hz in enumerate(hazards):
        active = []
        for i, L in enumerate(leads):
            objs = detect(prob[i, h], lat, lon, thr, lows=lows.get(int(L), ()) if hz in CYCLONE_HAZARDS else ())
            used = set()
            for tr in active:
                last = tr["steps"][-1]
                if i == 0 or last["lead"] != leads[i - 1]:  # track already ended
                    continue
                best, bd = None, GATE_KM
                for j, o in enumerate(objs):
                    if j in used:
                        continue
                    d = haversine_km(*last["centroid"], *o["centroid"])
                    if ("low" in last) != ("low" in o):
                        d += TYPE_CHANGE_KM
                    if d < bd:
                        best, bd = j, d
                if best is not None:
                    used.add(best)
                    tr["steps"].append(_step(objs[best], L, init))
            for j, o in enumerate(objs):
                if j not in used:
                    tr = {"hazard": hz, "steps": [_step(o, L, init)]}
                    active.append(tr)
        tracks += [t for t in active if len(t["steps"]) >= 3 and _severity(t) >= MIN_SEVERITY]
    # Tracks that follow a tropical low come first, then the rest; each group by severity.
    # A broad monsoon surge can score higher on area alone, but the storm is the system
    # forecasters track and the one Stage 2 downscales.
    for t in tracks:
        t["cyclone_steps"] = sum("low" in s for s in t["steps"])
        t["system"] = "tropical low" if t["cyclone_steps"] >= 3 else "large-scale"
    for k, t in enumerate(sorted(tracks, key=lambda t: (t["system"] != "tropical low", -_severity(t)))):
        t["id"] = f"T{k+1:02d}"
        t["envelope"] = _envelope(t)
        t["severity_score"] = round(_severity(t), 2)
    return sorted(tracks, key=lambda t: t["id"])


def _step(o, L, init):
    return {**o, "lead": int(L), "valid": str(pd.Timestamp(init) + pd.Timedelta(hours=int(L)))}


def _severity(t):
    return sum(s["peak_prob"] * s["area_cells"] for s in t["steps"]) / 10


def _envelope(t):
    b = np.array([s["bbox"] for s in t["steps"]])
    return {"lat_min": float(b[:, 0].min()), "lon_min": float(b[:, 1].min()),
            "lat_max": float(b[:, 2].max()), "lon_max": float(b[:, 3].max()),
            "t_start": t["steps"][0]["valid"], "t_end": t["steps"][-1]["valid"],
            "lead_start_h": t["steps"][0]["lead"], "lead_end_h": t["steps"][-1]["lead"]}


def cyclone_centres(ens, track_, search_deg=4.0, depth_hpa=2.0):
    """Per-member MSLP-minimum positions near each tracked step (ensemble track plume)."""
    if "mslp" not in ens:
        return None
    lat, lon = ens.lat.values, ens.lon.values
    out = []
    for s in track_["steps"]:
        c = s.get("low", s["centroid"])
        box = ens.mslp.sel(lead=s["lead"], lat=slice(c[0] - search_deg, c[0] + search_deg),
                           lon=slice(c[1] - search_deg, c[1] + search_deg)).values / 100.0
        la = lat[(lat >= c[0] - search_deg) & (lat <= c[0] + search_deg)]
        lo = lon[(lon >= c[1] - search_deg) & (lon <= c[1] + search_deg)]
        members = []
        for m in range(box.shape[0]):
            f = box[m]
            iy, ix = np.unravel_index(np.nanargmin(f), f.shape)
            if np.nanmedian(f) - f[iy, ix] >= depth_hpa:  # a real closed low, not noise
                members.append([float(la[iy]), float(lo[ix]), float(f[iy, ix])])
        m = np.array(members) if members else np.zeros((0, 3))
        out.append({"lead": s["lead"], "valid": s["valid"], "members": members,
                    "mean": m.mean(0).round(2).tolist() if len(m) else None,
                    "strike_fraction": round(len(m) / box.shape[0], 2)})
    return out
