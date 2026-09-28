"""Draw docs/architecture.png, a slide-ready picture of the two-stage pipeline."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

BOXES = {
    # key: (x, y, w, h, title, body, colour)
    "data": (0.2, 3.3, 2.6, 2.2, "Inputs", "NEPS-G / ECMWF ENS\n50 members, 0-10 days\n(member, lead, lat, lon)\n\nERA5 / IMDAA 30-yr\nclimate baseline", "#e8eef7"),
    "efi": (3.3, 3.3, 2.6, 2.2, "Climate CDF + EFI", "per grid point & valid date\nEFI, SOT, P(>q99)\nensemble mean / spread\nrain | wind | heat", "#fdf1dc"),
    "gnn": (6.4, 3.3, 2.9, 2.2, "Stage 1: mesh GNN", "grid -> icosahedral multi-mesh\n6 message-passing layers\nmesh -> grid\nP(observed extreme)\nper hazard, +-24 h context", "#dcefe3"),
    "trk": (9.8, 3.3, 2.7, 2.2, "Tracker", "threshold + connected\nregions, link across leads\n4D bounding boxes\nensemble cyclone plume", "#dcefe3"),
    "crop": (9.8, 0.3, 2.7, 2.2, "Crop anomaly window", "12 x 12 deg around core\ntail-focused members\ncoarse rain + 10 m wind", "#f3e3f3"),
    "dm": (6.4, 0.3, 2.9, 2.2, "Stage 2: diffusion", "conditional U-Net, DDIM\ncond: coarse rain, wind,\norography, land-sea\nphysics loss: water\nconservation + convergence", "#f3e3f3"),
    "alert": (3.3, 0.3, 2.6, 2.2, "5 km alert grid", "P(rain within 5 km >\nlow / moderate / severe)\nfrom generated samples\ncore pinpoint", "#fbe0e0"),
    "out": (0.2, 0.3, 2.6, 2.2, "Delivery", "FastAPI REST alerts\n/alerts/core  /point\n/zones  /tracks\nLeaflet map dashboard", "#fbe0e0"),
}
ARROWS = [("data", "efi"), ("efi", "gnn"), ("gnn", "trk"), ("trk", "crop"), ("crop", "dm"), ("dm", "alert"), ("alert", "out")]


def centre_edge(a, b):
    ax, ay, aw, ah = BOXES[a][:4]
    bx, by, bw, bh = BOXES[b][:4]
    if abs(ay - by) < 0.1:  # same row
        return ((ax + aw, ay + ah / 2), (bx, by + bh / 2)) if bx > ax else ((ax, ay + ah / 2), (bx + bw, by + bh / 2))
    return (ax + aw / 2, ay), (bx + bw / 2, by + bh)


def main():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 12.7), ax.set_ylim(0, 6.2), ax.axis("off")
    for k, (x, y, w, h, title, body, c) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.2, body, ha="center", va="center", fontsize=8.6, linespacing=1.35)
    for a, b in ARROWS:
        p, q = centre_edge(a, b)
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333"))
    ax.text(6.35, 5.9, "AI-driven spatio-temporal tracking of extreme weather anomalies: two-stage hybrid pipeline",
            ha="center", fontsize=12.5, weight="bold")
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
