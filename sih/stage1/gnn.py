"""Encode-process-decode message-passing GNN on the icosahedral multi-mesh.

Grid features (EFI, SOT, ensemble stats for several variables at t-24h, t, t+24h) are
encoded onto mesh nodes, processed by interaction-network layers on the multi-mesh, and
decoded back to the grid as per-hazard probabilities that an observed extreme
(ERA5 > climate q99) occurs. Message passing is written with plain torch scatter ops
(`index_add_`), which is what DGL's `update_all(message, reduce)` does under the hood, so
the model runs anywhere PyTorch runs; swapping in DGL graphs is a drop-in change.

All samples share one graph, so a batch is just a leading batch dimension.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(i, h, o, ln=True):
    layers = [nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o)]
    if ln:
        layers.append(nn.LayerNorm(o))
    return nn.Sequential(*layers)


def scatter_mean(src, index, n):
    """src: (B, E, H) messages, index: (E,) receivers -> (B, n, H) mean per receiver."""
    out = src.new_zeros(src.shape[0], n, src.shape[2]).index_add_(1, index, src)
    cnt = torch.bincount(index, minlength=n).clamp(min=1).to(src.dtype)
    return out / cnt[None, :, None]


class InteractionLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.edge = mlp(3 * h, h, h)
        self.node = mlp(2 * h, h, h)

    def forward(self, x, e, edges):
        s, d = edges[:, 0], edges[:, 1]
        e = e + self.edge(torch.cat([e, x[:, s], x[:, d]], -1))
        x = x + self.node(torch.cat([x, scatter_mean(e, d, x.shape[1])], -1))
        return x, e


class Bipartite(nn.Module):
    """Messages from one node set (senders) to another (receivers)."""

    def __init__(self, h, fe=3):
        super().__init__()
        self.edge = mlp(2 * h + fe, h, h)
        self.node = mlp(2 * h, h, h)

    def forward(self, xs, xr, ef, edges):
        s, d = edges[:, 0], edges[:, 1]
        B = xs.shape[0]
        m = self.edge(torch.cat([xs[:, s], xr[:, d], ef.expand(B, -1, -1)], -1))
        return xr + self.node(torch.cat([xr, scatter_mean(m, d, xr.shape[1])], -1))


class MeshGNN(nn.Module):
    def __init__(self, graph, n_in, n_out, hidden=64, layers=6):
        super().__init__()
        t = lambda k, dt=torch.float32: torch.as_tensor(graph[k], dtype=dt)
        for k in ("mesh_edges", "g2m", "m2g"):
            self.register_buffer(k, t(k, torch.long))
        for k in ("mesh_edge_feats", "g2m_feats", "m2g_feats", "mesh_pos"):
            self.register_buffer(k, t(k))
        self.n_mesh = graph["n_mesh"]
        self.grid_in = mlp(n_in, hidden, hidden)
        self.mesh_in = mlp(self.mesh_pos.shape[1], hidden, hidden)
        self.edge_in = mlp(3, hidden, hidden)
        self.enc = Bipartite(hidden)
        self.proc = nn.ModuleList([InteractionLayer(hidden) for _ in range(layers)])
        self.dec = Bipartite(hidden)
        self.head = mlp(hidden, hidden, n_out, ln=False)

    def forward(self, x):  # x: (B, n_grid, n_in) -> logits (B, n_grid, n_out)
        B = x.shape[0]
        g = self.grid_in(x)
        m = self.mesh_in(self.mesh_pos)[None].expand(B, -1, -1)
        m = self.enc(g, m, self.g2m_feats[None], self.g2m)
        e = self.edge_in(self.mesh_edge_feats)[None].expand(B, -1, -1)
        for layer in self.proc:
            m, e = layer(m, e, self.mesh_edges)
        g = self.dec(m, g, self.m2g_feats[None], self.m2g)
        return self.head(g)
