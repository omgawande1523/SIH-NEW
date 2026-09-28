"""REST alert API.

Run:  uvicorn sih.api.app:app --port 8000     (docs at http://localhost:8000/docs)

Endpoints
  GET /health                         service + product metadata
  GET /tracks                         tracked anomalies with their 4D bounding boxes
  GET /tracks/{track_id}              one track, step by step
  GET /alerts/core?lead_hours=        pinpoint core of the severe anomaly + 5 km impact circle
  GET /alerts/point?lat=&lon=         alert level for any location (5 km radius by default)
  GET /alerts/zones?lead_hours=       GeoJSON polygons of low / moderate / severe zones
  GET /dashboard                      the map dashboard (static HTML)
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from sih import config as C

LEVELS = list(C.ALERT_THRESHOLDS_MM6H)
app = FastAPI(title="SIH Extreme-Weather Anomaly Alert API", version="0.1.0",
              description="Spatio-temporal anomaly tracks (Stage 1 GNN) and 5 km alerts "
                          "from physics-informed diffusion downscaling (Stage 2).")


@lru_cache
def products():
    p = C.OUT / "products.json"
    if not p.exists():
        raise HTTPException(503, "No products yet: run `python run_demo.py` first")
    return json.loads(p.read_text())


def downscaling():
    d = products()["downscaling"]
    if not d:
        raise HTTPException(404, "no heavy-rain anomaly was tracked in this forecast, so there are no 5 km alerts")
    return d


@lru_cache
def grid():
    p = C.OUT / "alert_grid.npz"
    if not p.exists():
        raise HTTPException(503, "alert grid missing: run `python run_demo.py` again")
    z = np.load(p)
    leads = [int(x) for x in z["leads"]]
    return leads, {L: {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(f"{L}/")} for L in leads}


def circle(lat, lon, radius_km, n=48):
    pts = []
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        pts.append([round(lon + radius_km / (111.2 * math.cos(math.radians(lat))) * math.sin(a), 5),
                    round(lat + radius_km / 111.2 * math.cos(a), 5)])
    return {"type": "Polygon", "coordinates": [pts]}


def _level(probs, p_min=0.5):
    return next((k for k in reversed(LEVELS) if probs[k] >= p_min), "none")


def _pick_lead(lead_hours):
    d = downscaling()
    leads, g = grid()
    if lead_hours is None:
        lead_hours = d["core"]["lead"]
    if lead_hours not in g:
        raise HTTPException(404, f"lead_hours must be one of {leads}")
    return lead_hours, g[lead_hours]


@app.get("/health")
def health():
    p = products()
    return {"status": "ok", "event": p["event"], "init": p["init"], "source": p["source"],
            "n_tracks": len(p["tracks"]), "downscaled_leads_h": p["downscaling"]["leads"] if p["downscaling"] else []}


@app.get("/tracks")
def tracks():
    return [{"id": t["id"], "hazard": t["hazard"], "severity_score": t["severity_score"],
             "envelope": t["envelope"], "n_steps": len(t["steps"])} for t in products()["tracks"]]


@app.get("/tracks/{track_id}")
def track(track_id: str):
    for t in products()["tracks"]:
        if t["id"] == track_id:
            return t
    raise HTTPException(404, "unknown track")


def _core_feature(c, d, radius_km):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]},
        "properties": {**c, "radius_km": radius_km, "track_id": d["track_id"], "impact_area": circle(c["lat"], c["lon"], radius_km),
                       "message": f"{c['level'].upper()} alert within {radius_km:g} km of {c['lat']:.2f}N "
                                  f"{c['lon']:.2f}E, valid {c['valid']} UTC. Chance of more than "
                                  + ", ".join(f"{C.ALERT_THRESHOLDS_MM6H[k]:g} mm" for k in LEVELS)
                                  + " in 6 h: " + " / ".join(f"{c['prob'][k]:.0%}" for k in LEVELS)},
    }


@app.get("/alerts/core")
def core(lead_hours: int | None = None, radius_km: float = Query(C.ALERT_RADIUS_KM, gt=0, le=50)):
    """Headline core (rule in sih.products._headline), or the core at one downscaled lead time."""
    d = downscaling()
    c = d["core"]
    if lead_hours is not None:
        c = next((x for x in d["cores"] if x["lead"] == lead_hours), None)
        if c is None:
            raise HTTPException(404, f"lead_hours must be one of {d['leads']}")
    return _core_feature(c, d, radius_km)


@app.get("/alerts/first-warning")
def first_warning(radius_km: float = Query(C.ALERT_RADIUS_KM, gt=0, le=50)):
    """Earliest downscaled lead whose core reaches at least the low level (P >= 50%)."""
    d = downscaling()
    c = d.get("first_warning")
    if c is None:
        raise HTTPException(404, "no downscaled lead reaches an alert level")
    return _core_feature(c, d, radius_km)


@app.get("/alerts/point")
def point(lat: float, lon: float, lead_hours: int | None = None,
          radius_km: float = Query(C.ALERT_RADIUS_KM, gt=0, le=50)):
    L, g = _pick_lead(lead_hours)
    la, lo = g["lat5"], g["lon5"]
    if not (la[0] <= lat <= la[-1] and lo[0] <= lon <= lo[-1]):
        return {"lead_hours": L, "level": "none", "reason": "outside the downscaled impact window",
                "window": [float(la[0]), float(lo[0]), float(la[-1]), float(lo[-1])]}
    # probabilities are precomputed for a 5 km radius; widen by taking the max over the disc
    ry = max(0, int(round((radius_km - C.ALERT_RADIUS_KM) / 5.0)))
    iy, ix = np.abs(la - lat).argmin(), np.abs(lo - lon).argmin()
    s = (slice(max(0, iy - ry), iy + ry + 1), slice(max(0, ix - ry), ix + ry + 1))
    probs = {k: round(float(g["probs"][j][s].max()), 2) for j, k in enumerate(LEVELS)}
    d_ = downscaling()
    c = next((x for x in d_["cores"] if x["lead"] == L), d_["core"])
    d = 6371 * math.acos(min(1, math.sin(math.radians(lat)) * math.sin(math.radians(c["lat"])) +
                             math.cos(math.radians(lat)) * math.cos(math.radians(c["lat"])) *
                             math.cos(math.radians(lon - c["lon"]))))
    return {"lead_hours": L, "lat": lat, "lon": lon, "radius_km": radius_km, "level": _level(probs),
            "prob": probs, "rain_q90_mm6h": round(float(g["q90"][s].max()), 1),
            "core": {"lat": c["lat"], "lon": c["lon"]}, "distance_to_core_km": round(d, 1)}


@app.get("/alerts/zones")
def zones(lead_hours: int | None = None, p_min: float = Query(0.5, ge=0.05, le=1.0)):
    """Contours where P(level within 5 km) >= p_min, as a GeoJSON FeatureCollection.

    Each polygon keeps its holes (a ring-shaped zone is one polygon with an inner ring).
    """
    L, g = _pick_lead(lead_hours)
    feats = []
    for j, k in enumerate(LEVELS):
        for rings in contour_polygons(g["lon5"], g["lat5"], g["probs"][j], p_min):
            feats.append({"type": "Feature", "properties": {"level": k, "p_min": p_min, "lead_hours": L},
                          "geometry": {"type": "Polygon", "coordinates": rings}})
    return {"type": "FeatureCollection", "features": feats}


def contour_polygons(x, y, z, level):
    """Filled-contour polygons of z >= level as lists of rings [outer, hole, hole, ...].

    Uses matplotlib's object API (no pyplot global state), so it is safe in FastAPI's
    worker threads.
    """
    from matplotlib.figure import Figure
    from matplotlib.path import Path

    ax = Figure().add_subplot()
    cs = ax.contourf(x, y, z, levels=[level, max(1.01, float(np.nanmax(z)) + 0.01)])
    out = []
    for path in cs.get_paths():
        rings = [r for r in path.to_polygons() if len(r) >= 4]
        if not rings:
            continue
        area = [0.5 * np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1]) for r in rings]
        # the largest ring fixes the orientation of outer boundaries; the others are holes
        sign = np.sign(area[int(np.argmax(np.abs(area)))])
        outers = [i for i, a in enumerate(area) if np.sign(a) == sign]
        polys = {i: [rings[i]] for i in outers}
        for i, a in enumerate(area):
            if np.sign(a) == sign:
                continue
            host = next((o for o in outers if Path(rings[o]).contains_point(rings[i][0])), None)
            if host is not None:
                polys[host].append(rings[i])
        for rs in polys.values():
            out.append([np.round(r, 4).tolist() for r in rs])
    return out


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    p = C.OUT / "dashboard.html"
    if not p.exists():
        raise HTTPException(503, "dashboard not built yet")
    return FileResponse(p)
