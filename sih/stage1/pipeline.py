"""Stage 1: ensemble -> EFI features -> mesh GNN -> per-hazard extreme probabilities."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import xarray as xr

from sih import config as C
from sih.stage1.efi import cached_features_ok, ensemble_features, feature_key
from sih.stage1.gnn import MeshGNN
from sih.stage1.mesh import build_graph

HAZARDS = ["heavy_rain", "damaging_wind", "extreme_heat"]  # aligned with C.STAGE1_VARS
CTX = 4  # temporal context: +-4 leads = +-24 h


def event_features(name, exclude_years=()):
    """EFI features and labels for one event (cached). `exclude_years` drops those years
    from the climate sample (used by cross-validation; computed in memory, not cached)."""
    path = C.CACHE / f"feats_{name}.npz"
    if not exclude_years and cached_features_ok(name):
        z = np.load(path, allow_pickle=True)
        r = {k: z[k] for k in z.files}
        r["init"] = str(r["init"])
        return r
    ens = xr.open_dataset(C.CACHE / f"ens_{name}.nc").load()
    clim = xr.open_dataset(C.CACHE / "clim_samples.nc").load()
    if exclude_years:
        clim = clim.isel(time=~np.isin(clim.time.dt.year.values, list(exclude_years)))
    vp = C.CACHE / f"verif_{name}.nc"
    verif = xr.open_dataset(vp).load() if vp.exists() else None
    r = ensemble_features(ens, clim, verif)
    if exclude_years:
        return r
    np.savez_compressed(path, key=feature_key(C.CACHE / f"ens_{name}.nc"), **r)
    return r


def stack_inputs(r):
    """(lead, lat, lon, F) -> (lead, n_grid, F_total) with +-24 h context and position."""
    f = r["feats"]
    nl, ny, nx, F = f.shape
    prev = f[np.clip(np.arange(nl) - CTX, 0, nl - 1)]
    nxt = f[np.clip(np.arange(nl) + CTX, 0, nl - 1)]
    lat, lon = np.meshgrid(r["lat"], r["lon"], indexing="ij")
    stat = np.stack([lat / 90, (lon - 80) / 40], -1)[None].repeat(nl, 0)
    lead = (r["leads"] / 240.0)[:, None, None, None] * np.ones((nl, ny, nx, 1))
    x = np.concatenate([prev, f, nxt, stat, lead], -1).astype("float32")
    return x.reshape(nl, ny * nx, -1)


def stack_labels(r):
    y = r["labels"]  # (lead, var, lat, lon)
    nl, nv = y.shape[:2]
    return y.reshape(nl, nv, -1).transpose(0, 2, 1).astype("float32")


def auc(score, y):
    m = np.isfinite(y)
    score, y = score[m], y[m]
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    npos = y.sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def train(epochs=30, hidden=64, layers=6, seed=0, log=print, holdout=None, save=True):
    """Train on every event except the test (or `holdout`) events. Returns the metrics,
    or with save=False the trained model (used by scripts/stage1_crossval.py)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    events = [e for e in C.EVENTS]
    data = {e.name: event_features(e.name) for e in events}
    r0 = data[events[0].name]
    graph = build_graph(r0["lat"], r0["lon"], level=C.MESH_LEVEL)
    X = {k: torch.from_numpy(stack_inputs(v)) for k, v in data.items()}
    Y = {k: torch.from_numpy(stack_labels(v)) for k, v in data.items()}
    tr = [e.name for e in events if (e.name not in holdout if holdout else e.split == "train")]
    Xtr, Ytr = torch.cat([X[k] for k in tr]), torch.cat([Y[k] for k in tr])
    model = MeshGNN(graph, Xtr.shape[-1], len(HAZARDS), hidden, layers)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pos = torch.nanmean(Ytr, dim=(0, 1))
    pos_weight = ((1 - pos) / pos.clamp(min=1e-3)).clamp(max=20) ** 0.5
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    log(f"Stage 1 GNN: mesh nodes={graph['n_mesh']} mesh edges={len(graph['mesh_edges'])} "
        f"grid={graph['n_grid']} train samples={len(Xtr)} params={sum(p.numel() for p in model.parameters())/1e3:.0f}k")
    t0 = time.time()
    B = 8
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), B):
            idx = perm[i:i + B]
            y = Ytr[idx]
            mask = torch.isfinite(y)
            logits = model(Xtr[idx])
            loss = (lossf(logits, torch.nan_to_num(y)) * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  epoch {ep:3d} loss {tot/len(Xtr):.4f} ({time.time()-t0:.0f}s)")
    if not save:
        return model.eval()
    torch.save({"state": model.state_dict(), "graph": graph, "n_in": Xtr.shape[-1],
                "hidden": hidden, "layers": layers}, C.MODELS / "stage1_gnn.pt")
    return evaluate(log=log)


def evaluate(log=print):
    """Held-out skill: GNN probability vs raw EFI as a detector of observed extremes."""
    model, _ = load_model()
    metrics = {}
    for k in [e.name for e in C.EVENTS if e.split == "test"]:
        r = event_features(k)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(stack_inputs(r)))).numpy()
        y = stack_labels(r)
        efi = r["efi"].reshape(len(p), len(HAZARDS), -1).transpose(0, 2, 1)
        for j, h in enumerate(HAZARDS):
            yy = y[..., j]
            m = np.isfinite(yy)
            metrics[f"{k}/{h}"] = {
                "auc_gnn": round(auc(p[..., j], yy), 3),
                "auc_raw_efi": round(auc(efi[..., j], yy), 3),
                "brier_gnn": round(float(np.mean((p[..., j][m] - yy[m]) ** 2)), 4),
                "base_rate": round(float(np.nanmean(yy)), 4),
            }
    log("Stage 1 held-out skill (AUC, higher is better): " + json.dumps(metrics))
    (C.OUT / "stage1_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model():
    ck = torch.load(C.MODELS / "stage1_gnn.pt", weights_only=False)
    m = MeshGNN(ck["graph"], ck["n_in"], len(HAZARDS), ck["hidden"], ck["layers"])
    m.load_state_dict(ck["state"])
    return m.eval(), ck["graph"]


def infer(name):
    """Probabilities (lead, hazard, lat, lon) for one ensemble run."""
    model, _ = load_model()
    r = event_features(name)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(stack_inputs(r)))).numpy()
    ny, nx = len(r["lat"]), len(r["lon"])
    prob = p.transpose(0, 2, 1).reshape(len(r["leads"]), len(HAZARDS), ny, nx)
    return prob, r
