"""Check the tracker's cyclone split on every event, with the constants fixed on Amphan.

For each event: find tropical lows in the ensemble-mean MSLP (tracker.find_lows), run the
tracker, and report
  * the tracks that follow a low, with the low's position at the start, middle and end,
  * at how many of their steps the cyclone object was cut out of a larger rain/wind region
    (the region the tracker would have drawn without the split), and the largest such cut,
  * lows that belong to no track (e.g. heat lows) and lows on the domain edge.
Nothing is retuned here. Stage 1 probabilities come from the shipped GNN, which was trained
on the six training events, so for them the probabilities are in-sample; the split itself
depends only on the ensemble MSLP and wind. Writes outputs/tracker_check.json.

Needs the full training data (`python -m sih.data.fetch_era5`) plus the ensemble MSLP for
the training events (a few MB each, downloaded here once).
"""
import json
import sys
from pathlib import Path

import torch  # noqa: F401  (import before netCDF/HDF5 libraries: avoids an MKL loader clash)
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sih import config as C  # noqa: E402
from sih.data.fetch_era5 import fetch_ensemble_mslp  # noqa: E402
from sih.stage1 import pipeline as S1  # noqa: E402
from sih.stage1.tracker import detect, find_lows, track  # noqa: E402


def _host_cells(prob2d, lat, lon, low):
    """Size of the unsplit region the low sits in (0 if none)."""
    a, b = low
    host = [o["area_cells"] for o in detect(prob2d, lat, lon)
            if o["bbox"][0] - 1.6 <= a <= o["bbox"][2] + 1.6 and o["bbox"][1] - 1.6 <= b <= o["bbox"][3] + 1.6]
    return max(host, default=0)


def main():
    out = {}
    for ev in C.EVENTS:
        ens = xr.open_dataset(C.CACHE / f"ens_{ev.name}.nc").load()
        if "mslp" not in ens:
            ens["mslp"] = xr.open_dataset(fetch_ensemble_mslp(ev)).load().mslp
        prob, r = S1.infer(ev.name)
        lat, lon, leads = r["lat"], r["lon"], list(r["leads"])
        lows = find_lows(ens, leads)
        tracks = track(prob, lat, lon, leads, r["init"], S1.HAZARDS, lows=lows)
        cyc = [t for t in tracks if t["system"] == "tropical low"]
        used = {(s["lead"], tuple(s["low"])) for t in cyc for s in t["steps"] if "low" in s}
        all_lows = [(L, (round(a, 2), round(b, 2))) for L, v in lows.items() for a, b in v]
        rows = []
        for t in cyc:
            h = S1.HAZARDS.index(t["hazard"])
            steps = [s for s in t["steps"] if "low" in s]
            cuts = [(_host_cells(prob[leads.index(s["lead"]), h], lat, lon, s["low"]), s["area_cells"]) for s in steps]
            cut = [c for c in cuts if c[0] > c[1]]
            rows.append({"id": t["id"], "hazard": t["hazard"],
                         "leads": [t["envelope"]["lead_start_h"], t["envelope"]["lead_end_h"]],
                         "low_path": [[steps[k]["lead"], *steps[k]["low"]] for k in (0, len(steps) // 2, -1)],
                         "steps_cut_from_larger_region": [len(cut), len(steps)],
                         "largest_cut_cells": list(max(cut, key=lambda c: c[0] / c[1])) if cut else None})
        out[ev.name] = {"kind": ev.kind, "init": ev.init, "cyclone_tracks": rows,
                        "lows_found": len(all_lows),
                        "lows_in_no_track": sum(k not in used for k in all_lows),
                        "lows_on_domain_edge": int(sum(min(abs(a - lat[0]), abs(a - lat[-1])) < 0.1 or
                                                   min(abs(b - lon[0]), abs(b - lon[-1])) < 0.1
                                                   for _, (a, b) in all_lows))}
        print(ev.name, json.dumps(out[ev.name]), flush=True)
    (C.OUT / "tracker_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
