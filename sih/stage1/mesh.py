"""Icosahedral multi-mesh and the grid <-> mesh bipartite graphs.

The sphere is tiled by recursively subdividing an icosahedron (GraphCast-style). Because
refinement keeps the coarser vertices at the same indices, edges from every refinement
level can be merged into one "multi-mesh": fine edges carry local detail, coarse edges
let messages cross thousands of kilometres in one hop. Working on the mesh avoids the
pole/row distortions of a lat-lon pixel grid.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

R_EARTH_KM = 6371.0


def icosahedron():
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return v / np.linalg.norm(v, axis=1, keepdims=True), f


def refine(v, f):
    cache, verts = {}, list(v)

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k not in cache:
            m = verts[a] + verts[b]
            verts.append(m / np.linalg.norm(m))
            cache[k] = len(verts) - 1
        return cache[k]

    nf = []
    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(verts), np.array(nf)


def xyz_to_latlon(x):
    return np.degrees(np.arcsin(x[:, 2])), np.degrees(np.arctan2(x[:, 1], x[:, 0])) % 360


def latlon_to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _faces_to_edges(f):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    return np.unique(e, axis=0)


def _edge_feats(xs, xd):
    """Displacement of sender relative to receiver in the receiver's local tangent frame."""
    lat_d, lon_d = xyz_to_latlon(xd)
    lat_s, lon_s = xyz_to_latlon(xs)
    dlon = ((lon_s - lon_d + 180) % 360 - 180) * np.cos(np.radians(lat_d))
    dlat = lat_s - lat_d
    dist = np.linalg.norm(xs - xd, axis=1)
    return np.stack([dlon / 10, dlat / 10, dist * 10], 1).astype("float32")


def build_graph(grid_lat, grid_lon, level=6, min_level=2, margin_deg=3.0):
    """Regional multi-mesh over the grid's bounding box plus the bipartite links."""
    v, f = icosahedron()
    level_faces = []
    for lev in range(level):
        v, f = refine(v, f)
        if lev + 1 >= min_level:
            level_faces.append(f)
    lat, lon = xyz_to_latlon(v)
    keep = ((lat >= grid_lat.min() - margin_deg) & (lat <= grid_lat.max() + margin_deg) &
            (lon >= grid_lon.min() - margin_deg) & (lon <= grid_lon.max() + margin_deg))
    idx = -np.ones(len(v), int)
    idx[keep] = np.arange(keep.sum())
    edges = np.unique(np.concatenate([_faces_to_edges(ff) for ff in level_faces]), axis=0)
    edges = idx[edges]
    edges = edges[(edges >= 0).all(1)]
    mesh_x = v[keep]

    glat, glon = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    grid_x = latlon_to_xyz(glat.ravel(), glon.ravel())
    # encoder: every mesh node reads its 4 nearest grid points
    _, g_nn = cKDTree(grid_x).query(mesh_x, k=4)
    g2m = np.stack([g_nn.ravel(), np.repeat(np.arange(len(mesh_x)), 4)], 1)
    # decoder: every grid point reads its 3 nearest mesh nodes
    _, m_nn = cKDTree(mesh_x).query(grid_x, k=3)
    m2g = np.stack([m_nn.ravel(), np.repeat(np.arange(len(grid_x)), 3)], 1)

    mlat, mlon = xyz_to_latlon(mesh_x)
    return {
        "mesh_latlon": np.stack([mlat, mlon], 1).astype("float32"),
        "mesh_pos": np.concatenate([mesh_x, np.stack([np.sin(np.radians(mlat))], 1)], 1).astype("float32"),
        "mesh_edges": edges.astype("int64"),
        "mesh_edge_feats": _edge_feats(mesh_x[edges[:, 0]], mesh_x[edges[:, 1]]),
        "g2m": g2m.astype("int64"),
        "g2m_feats": _edge_feats(grid_x[g2m[:, 0]], mesh_x[g2m[:, 1]]),
        "m2g": m2g.astype("int64"),
        "m2g_feats": _edge_feats(mesh_x[m2g[:, 0]], grid_x[m2g[:, 1]]),
        "n_grid": len(grid_x), "n_mesh": len(mesh_x), "level": level,
    }
