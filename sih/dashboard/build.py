"""Map dashboard (Folium/Leaflet HTML) and static figures for slides."""
from __future__ import annotations

import json

import folium
import matplotlib
import numpy as np
from folium.plugins import TimestampedGeoJson

from sih import config as C
from sih.api.app import zones as api_zones

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402

LEVEL_COLOURS = {"low": "#f2c14e", "moderate": "#f07c2a", "severe": "#c0162c"}
HAZARD_COLOURS = {"heavy_rain": "#1f6fd1", "damaging_wind": "#7b3fbf", "extreme_heat": "#d6452b"}


def _rgba(field, cmap, vmin, vmax, alpha_floor=0.0):
    f = np.clip((field - vmin) / (vmax - vmin), 0, 1)
    img = colormaps[cmap](f)
    img[..., 3] = np.where(f <= alpha_floor, 0, 0.75)
    return img[::-1]  # north-up for Leaflet


def _bounds(lat, lon):
    dy, dx = abs(lat[1] - lat[0]) / 2, abs(lon[1] - lon[0]) / 2
    return [[float(lat.min() - dy), float(lon.min() - dx)], [float(lat.max() + dy), float(lon.max() + dx)]]


def build_dashboard():
    prod = json.loads((C.OUT / "products.json").read_text())
    f = np.load(C.OUT / "fields.npz")
    lat, lon, leads = f["lat"], f["lon"], f["leads"]
    core = prod["downscaling"]["core"] if prod["downscaling"] else None
    m = folium.Map(location=[core["lat"], core["lon"]] if core else [20, 82], zoom_start=5,
                   tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", name="Topographic",
                     attr="Map data: OpenStreetMap contributors, SRTM | Style: OpenTopoMap (CC-BY-SA)",
                     show=False).add_to(m)

    mid = (leads >= 72) & (leads <= 240)  # the 3-10 day window
    for h, hz in enumerate(prod["hazards"]):
        efi = np.nan_to_num(f["efi"][mid, h]).max(0)
        folium.raster_layers.ImageOverlay(
            _rgba(efi, "YlOrRd", 0.5, 1.0, 0.0), _bounds(lat, lon), mercator_project=True,
            name=f"EFI {hz.replace('_', ' ')} (max, day 3-10)", show=(hz == "heavy_rain")).add_to(m)
        pr = f["prob"][mid, h].max(0)
        folium.raster_layers.ImageOverlay(
            _rgba(pr, "PuBu", 0.2, 1.0, 0.0), _bounds(lat, lon), mercator_project=True,
            name=f"GNN P(extreme) {hz.replace('_', ' ')}", show=False).add_to(m)

    # tracks: envelope, per-step boxes (animated), centroid path, cyclone plume
    feats = []
    plume = folium.FeatureGroup(name="Ensemble member cyclone centres", show=False)
    for rank, t in enumerate(sorted(prod["tracks"], key=lambda t: -t["severity_score"])):
        col = HAZARD_COLOURS[t["hazard"]]
        e = t["envelope"]
        grp = folium.FeatureGroup(name=f"Track {t['id']} {t['hazard'].replace('_', ' ')}", show=rank < 3)
        folium.Rectangle([[e["lat_min"], e["lon_min"]], [e["lat_max"], e["lon_max"]]], color=col, weight=1,
                         dash_array="6", fill=False,
                         tooltip=f"{t['id']} {t['hazard']} 4D box {e['t_start'][:13]} to {e['t_end'][:13]}").add_to(grp)
        folium.PolyLine([s["centroid"] for s in t["steps"]], color=col, weight=3,
                        tooltip=f"{t['id']} anomaly centroid path").add_to(grp)
        if t.get("cyclone_plume"):
            pts = [p["mean"][:2] for p in t["cyclone_plume"] if p["mean"]]
            if len(pts) > 1:
                folium.PolyLine(pts, color="black", weight=2, dash_array="2,6",
                                tooltip="ensemble-mean cyclone centre (MSLP minimum)").add_to(grp)
            for p in t["cyclone_plume"]:
                for mem in p["members"][::3]:
                    folium.CircleMarker(mem[:2], radius=1.5, color="#555", fill=True, opacity=0.4,
                                        tooltip=f"member low {mem[2]:.0f} hPa, +{p['lead']}h").add_to(plume)
        grp.add_to(m)
        for s in t["steps"]:
            b = s["bbox"]
            feats.append({"type": "Feature",
                          "geometry": {"type": "Polygon", "coordinates": [[[b[1], b[0]], [b[3], b[0]], [b[3], b[2]],
                                                                           [b[1], b[2]], [b[1], b[0]]]]},
                          "properties": {"time": s["valid"].replace(" ", "T"),
                                         "style": {"color": col, "weight": 2, "fillOpacity": 0.15},
                                         "popup": f"{t['id']} {t['hazard']} +{s['lead']}h P={s['peak_prob']:.2f}"}})
    plume.add_to(m)
    if feats:
        TimestampedGeoJson({"type": "FeatureCollection", "features": feats}, period="PT6H", duration="PT5H",
                           add_last_point=False, auto_play=False, loop=False, transition_time=300).add_to(m)

    if core:
        g = np.load(C.OUT / "alert_grid.npz")
        L = core["lead"]
        la, lo, q90 = g[f"{L}/lat5"], g[f"{L}/lon5"], g[f"{L}/q90"]
        folium.raster_layers.ImageOverlay(
            _rgba(q90, "turbo", 2, 80, 0.02), _bounds(la, lo), mercator_project=True,
            name=f"Downscaled rain q90, +{L}h (mm/6h, 5 km lattice)", show=False).add_to(m)
        for j, lvl in ((2, "severe"), (1, "moderate")):
            folium.raster_layers.ImageOverlay(
                _rgba(g[f"{L}/probs"][j], "magma_r", 0.0, 1.0, 0.05), _bounds(la, lo), mercator_project=True,
                name=f"P({lvl} rain within 5 km), +{L}h", show=(lvl == "severe")).add_to(m)
        zg = folium.FeatureGroup(name="Alert zones (P>=50% within 5 km)")
        for feat in api_zones(L, 0.5)["features"]:
            lvl = feat["properties"]["level"]
            folium.GeoJson(feat, style_function=lambda _, c=LEVEL_COLOURS[lvl]: {
                "color": c, "weight": 1.5, "fillColor": c, "fillOpacity": 0.25},
                tooltip=f"{lvl} alert").add_to(zg)
        zg.add_to(m)
        cg = folium.FeatureGroup(name="Alert core at each downscaled lead")
        for c in prod["downscaling"]["cores"]:
            folium.CircleMarker([c["lat"], c["lon"]], radius=6, color=LEVEL_COLOURS.get(c["level"], "#666"),
                                fill=True, fill_opacity=0.9,
                                tooltip=f"+{c['lead']} h core: {c['level']}, rain q90 {c['rain_q90_mm6h']} mm/6h, "
                                        f"P(severe) {c['prob']['severe']:.0%}").add_to(cg)
        cg.add_to(m)
        folium.Circle([core["lat"], core["lon"]], radius=C.ALERT_RADIUS_KM * 1000,
                      color=LEVEL_COLOURS.get(core["level"], "black"), weight=3, fill=True, fill_opacity=0.3).add_to(m)
        folium.Marker([core["lat"], core["lon"]], icon=folium.Icon(color="red", icon="warning-sign"),
                      popup=folium.Popup(
                          f"<b>{core['level'].upper()} ALERT CORE</b><br>{core['lat']:.2f}N {core['lon']:.2f}E<br>"
                          f"valid {core['valid']} UTC (+{L} h)<br>rain q90 {core['rain_q90_mm6h']} mm/6h<br>"
                          f"P(low/mod/severe within 5 km): {core['prob']['low']:.0%} / {core['prob']['moderate']:.0%}"
                          f" / {core['prob']['severe']:.0%}", max_width=280)).add_to(m)

    first = prod["downscaling"].get("first_warning") if prod["downscaling"] else None
    warn = ""
    if core:
        warn += (f"Headline (+{core['lead']} h): <b>{core['level'].upper()}</b>, "
                 f"P(severe) {core['prob']['severe']:.0%}<br>")
    if first:
        warn += (f"Earliest actionable warning: <b>{first['level'].upper()}</b> at +{first['lead']} h, "
                 f"{first['lat']:.1f}N {first['lon']:.1f}E<br>")
    legend = f"""<div style="position: fixed; bottom: 28px; right: 12px; z-index: 9999; background: rgba(255,255,255,.92);
      padding: 10px 12px; border-radius: 6px; font: 12px sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,.3)">
      <b>{prod['event'].title()} · ensemble init {prod['init']} 00 UTC</b><br>
      Tracks: <span style="color:{HAZARD_COLOURS['heavy_rain']}">■</span> heavy rain
      <span style="color:{HAZARD_COLOURS['damaging_wind']}">■</span> wind
      <span style="color:{HAZARD_COLOURS['extreme_heat']}">■</span> heat · dashed = 4D envelope<br>
      Alerts within {C.ALERT_RADIUS_KM:g} km: <span style="color:{LEVEL_COLOURS['low']}">■</span> low ≥{C.ALERT_THRESHOLDS_MM6H['low']:g}
      <span style="color:{LEVEL_COLOURS['moderate']}">■</span> moderate ≥{C.ALERT_THRESHOLDS_MM6H['moderate']:g}
      <span style="color:{LEVEL_COLOURS['severe']}">■</span> severe ≥{C.ALERT_THRESHOLDS_MM6H['severe']:g} mm/6h<br>
      {warn}Time slider (bottom) animates the 6-hourly bounding boxes.</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    path = C.OUT / "dashboard.html"
    m.save(str(path))
    return path


def stage1_figure():
    prod = json.loads((C.OUT / "products.json").read_text())
    f = np.load(C.OUT / "fields.npz")
    lat, lon, leads = f["lat"], f["lon"], list(f["leads"])
    show = [L for L in (72, 96, 120, 144) if L in leads]
    fig, ax = plt.subplots(2, len(show), figsize=(4 * len(show), 7.2), sharex=True, sharey=True)
    for c, L in enumerate(show):
        i = leads.index(L)
        a = ax[0, c]
        im0 = a.pcolormesh(lon, lat, f["efi"][i, 0], cmap="YlOrRd", vmin=0, vmax=1, shading="nearest")
        a.set_title(f"EFI 24 h rain, +{L} h")
        b = ax[1, c]
        im1 = b.pcolormesh(lon, lat, f["prob"][i].max(0), cmap="PuBu", vmin=0, vmax=1, shading="nearest")
        b.set_title(f"GNN P(extreme), +{L} h")
        for t in prod["tracks"]:
            for s in t["steps"]:
                if s["lead"] == L:
                    bb = s["bbox"]
                    b.add_patch(plt.Rectangle((bb[1] - .75, bb[0] - .75), bb[3] - bb[1] + 1.5, bb[2] - bb[0] + 1.5,
                                              fill=False, ec=HAZARD_COLOURS[t["hazard"]], lw=2))
                    b.text(bb[1], bb[2] + 1, t["id"], color=HAZARD_COLOURS[t["hazard"]], fontsize=8)
            for p in t.get("cyclone_plume") or []:
                if p["lead"] == L and p["members"]:
                    mm = np.array(p["members"])
                    a.scatter(mm[:, 1], mm[:, 0], s=4, c="k", alpha=.5)
    fig.colorbar(im0, ax=ax[0].tolist(), shrink=.8, label="EFI")
    fig.colorbar(im1, ax=ax[1].tolist(), shrink=.8, label="probability")
    fig.suptitle(f"Stage 1 — {prod['event'].title()}, ECMWF 50-member ensemble init {prod['init']} "
                 "(black dots: member cyclone centres; boxes: tracked anomalies)")
    path = C.OUT / "stage1_tracking.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def stage2_forecast_figure():
    prod = json.loads((C.OUT / "products.json").read_text())
    if not prod["downscaling"]:
        return None
    g = np.load(C.OUT / "alert_grid.npz")
    L = prod["downscaling"]["core"]["lead"]
    c = prod["downscaling"]["core"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    clat, clon = g[f"{L}/coarse_lat"], g[f"{L}/coarse_lon"]
    ax[0].pcolormesh(clon, clat, g[f"{L}/coarse"][0], cmap="turbo", vmin=0, vmax=80, shading="nearest")
    ax[0].set_title(f"Coarse 1.5° member rain, +{L} h")
    im = ax[1].pcolormesh(g[f"{L}/fine_lon"], g[f"{L}/fine_lat"], g[f"{L}/samples"][0], cmap="turbo", vmin=0, vmax=80)
    ax[1].set_title("Diffusion sample (0.25°)")
    ax[2].pcolormesh(g[f"{L}/lon5"], g[f"{L}/lat5"], g[f"{L}/q90"], cmap="turbo", vmin=0, vmax=80)
    ax[2].set_title("q90 of max rain within 5 km")
    fig.colorbar(im, ax=ax[:3].tolist(), shrink=.85, label="mm / 6 h")
    imp = ax[3].pcolormesh(g[f"{L}/lon5"], g[f"{L}/lat5"], g[f"{L}/probs"][2], cmap="magma_r", vmin=0, vmax=1)
    ax[3].contour(g[f"{L}/lon5"], g[f"{L}/lat5"], g[f"{L}/probs"][2], [0.5], colors="c", linewidths=1.2)
    ax[3].plot(c["lon"], c["lat"], "c*", ms=14, mec="k")
    ax[3].set_title("P(severe within 5 km) + core pin\n(cyan line: alert threshold 50%)")
    fig.colorbar(imp, ax=ax[3], shrink=.85, label="probability")
    for a in ax:
        a.contour(g[f"{L}/fine_lon"], g[f"{L}/fine_lat"], g[f"{L}/lsm"], [0.5], colors="k", linewidths=0.7)
        a.set_xlim(g[f"{L}/fine_lon"][[0, -1]]), a.set_ylim(g[f"{L}/fine_lat"][[0, -1]])
        a.set_aspect("equal")
    path = C.OUT / "stage2_forecast_alerts.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
