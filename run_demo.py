#!/usr/bin/env python3
"""One-command demo: data -> Stage 1 GNN tracking -> Stage 2 diffusion downscaling -> alerts.

    python run_demo.py                      # real ECMWF ensemble + ERA5 (downloads ~200 MB once)
    python run_demo.py --source synthetic   # fully offline
    python run_demo.py --serve              # ...then start the alert API + dashboard on :8000
    python run_demo.py --retrain            # retrain both models instead of reusing weights
    python run_demo.py --fast               # tiny training runs, for a quick smoke test
"""
from __future__ import annotations

import argparse
import json
import time

import torch  # noqa: F401  (import before netCDF/HDF5 libraries: avoids an MKL loader clash)

from sih import config as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["era5", "synthetic"], default="era5")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    t0 = time.time()
    C.use_source(a.source)
    if a.fast:  # never overwrite the real trained weights with a smoke-test run
        C.MODELS = C.ROOT / "models" / f"{a.source}_fast"
        C.MODELS.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(f"[{time.time()-t0:6.0f}s] {s}", flush=True)

    need_training = a.retrain or a.fast or not all(
        (C.MODELS / f).exists() for f in ("stage1_gnn.pt", "stage2_diffusion.pt", "stage2_mse_unet.pt"))
    log(f"1/5 data ({a.source})")
    if a.source == "era5":
        from sih.data.fetch_era5 import fetch_all
        fetch_all(training=need_training)
    else:
        from sih.data.synthetic import generate
        generate(log=log)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    from sih.stage1 import pipeline as S1
    from sih.stage2 import pipeline as S2

    log("2/5 Stage 1: EFI + icosahedral-mesh GNN")
    if a.retrain or a.fast or not (C.MODELS / "stage1_gnn.pt").exists():
        S1.train(epochs=4 if a.fast else 30, log=log)
    else:
        log("  reusing trained GNN (use --retrain to train again)")
        S1.evaluate(log=log)

    log("3/5 Stage 2: physics-informed diffusion downscaler")
    if a.retrain or a.fast or not (C.MODELS / "stage2_diffusion.pt").exists():
        S2.train(steps=150 if a.fast else 3000, mse_steps=100 if a.fast else 1500, log=log)
    else:
        log("  reusing trained diffusion model and MSE baseline")
    nophys = S2.load_variant("nophys")  # optional ablation, see scripts/stage2_ablation.py
    S2.evaluate(extra_diffusion={"diffusion_no_physics": nophys} if nophys else None, log=log)

    log("4/5 products: tracks, downscaled impact zone, 5 km alerts")
    from sih.products import build
    prod = build(log=log)

    log("5/5 dashboard and figures")
    from sih.dashboard.build import build_dashboard, stage1_figure, stage2_forecast_figure
    paths = [build_dashboard(), stage1_figure(), stage2_forecast_figure()]
    core = prod["downscaling"]["core"] if prod["downscaling"] else None
    print("\n" + "=" * 72)
    print(f"Event {prod['event']} | ensemble init {prod['init']} | {len(prod['tracks'])} tracked anomalies")
    for t in prod["tracks"][:5]:
        e = t["envelope"]
        print(f"  {t['id']} {t['hazard']:<14} +{e['lead_start_h']}..{e['lead_end_h']} h  "
              f"box {e['lat_min']:.1f}-{e['lat_max']:.1f}N {e['lon_min']:.1f}-{e['lon_max']:.1f}E")
    if core:
        print(f"ALERT CORE: {core['lat']:.2f}N {core['lon']:.2f}E valid {core['valid']} UTC (+{core['lead']} h) "
              f"level={core['level'].upper()} P(severe within {C.ALERT_RADIUS_KM:g} km)={core['prob']['severe']:.0%}")
    first = prod["downscaling"].get("first_warning") if prod["downscaling"] else None
    if first:
        print(f"EARLIEST ACTIONABLE WARNING: {first['lat']:.2f}N {first['lon']:.2f}E valid {first['valid']} UTC "
              f"(+{first['lead']} h) level={first['level'].upper()} P(low)={first['prob']['low']:.0%}")
    print("Metrics:", json.dumps(json.loads((C.OUT / "stage2_metrics.json").read_text())["summary"]))
    print("Outputs:", *[str(p.relative_to(C.ROOT)) for p in paths if p], sep="\n  ")
    print("=" * 72)
    log("done")

    if a.serve:
        import uvicorn
        print(f"Alert API on http://localhost:{a.port}/docs  dashboard on http://localhost:{a.port}/dashboard")
        uvicorn.run("sih.api.app:app", host="0.0.0.0", port=a.port)


if __name__ == "__main__":
    main()
