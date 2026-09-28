"""Hold-one-event-out cross-validation of the Stage 1 GNN (about 5 min per fold on CPU).

For each of the 7 events: train on the other 6, score the held-out one. Reports AUC of the
GNN and of the raw EFI for each hazard, overall and by lead-time window, and writes
outputs/stage1_crossval.json.

When the held-out event's year lies inside the 1990-2019 climate sample (Fani, Vayu and the
2019 heatwave; the quiet 2018 case), that year is dropped from the climatology for every
event in the fold, so the held-out event's EFI features and its q99 labels never use the
event's own observations.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sih import config as C  # noqa: E402
from sih.stage1 import pipeline as S1  # noqa: E402

WINDOWS = {"day 0-2": (0, 48), "day 3-5": (54, 120), "day 6-10": (126, 240)}


_cached_features = S1.event_features


def fold_features(exclude_years):
    memo = {}

    def features(name, **_):
        if name not in memo:
            memo[name] = _cached_features(name, exclude_years=exclude_years)
        return memo[name]
    return features


def main(epochs=30):
    res = {}
    for ev in C.EVENTS:
        year = int(ev.init[:4])
        excl = (year,) if C.CLIM_YEARS[0] <= year <= C.CLIM_YEARS[1] else ()
        S1.event_features = fold_features(excl)  # S1.train looks features up through this name
        model = S1.train(epochs=epochs, holdout=[ev.name], save=False, log=lambda s: None)
        r = S1.event_features(ev.name)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(S1.stack_inputs(r)))).numpy()
        y = S1.stack_labels(r)
        efi = r["efi"].reshape(len(p), len(S1.HAZARDS), -1).transpose(0, 2, 1)
        leads = r["leads"]
        res[ev.name] = {"clim_years_excluded": list(excl)}
        for j, h in enumerate(S1.HAZARDS):
            row = {"all": [S1.auc(p[..., j], y[..., j]), S1.auc(efi[..., j], y[..., j])]}
            for w, (a, b) in WINDOWS.items():
                m = (leads >= a) & (leads <= b)
                row[w] = [S1.auc(p[m][..., j], y[m][..., j]), S1.auc(efi[m][..., j], y[m][..., j])]
            res[ev.name][h] = {k: {"gnn": round(v[0], 3), "efi": round(v[1], 3)} for k, v in row.items()}
        print(ev.name, json.dumps(res[ev.name]), flush=True)
    S1.event_features = _cached_features
    summary = {}
    for h in S1.HAZARDS:
        summary[h] = {}
        for w in ["all", *WINDOWS]:
            g = [res[e][h][w]["gnn"] for e in res]
            f = [res[e][h][w]["efi"] for e in res]
            summary[h][w] = {"gnn_mean": round(float(np.nanmean(g)), 3), "efi_mean": round(float(np.nanmean(f)), 3),
                             "gnn_wins": int(np.nansum(np.array(g) > np.array(f)))}
    out = {"per_event": res, "summary": summary, "n_events": len(res)}
    (C.OUT / "stage1_crossval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
