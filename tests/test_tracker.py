import numpy as np

from sih.stage1.tracker import detect, haversine_km, track

LAT, LON = np.arange(0, 36.1, 1.5), np.arange(60, 100.6, 1.5)


def _blob(clat, clon, r=2.0):
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    return (np.hypot(la - clat, lo - clon) <= r).astype(float)


def test_haversine_one_degree_of_latitude():
    assert abs(haversine_km(10, 80, 11, 80) - 111.2) < 1.0


def test_moving_blob_becomes_one_track_with_growing_envelope():
    leads = np.arange(0, 49, 6)
    prob = np.stack([_blob(10 + 0.5 * i, 85)[None] for i in range(len(leads))])  # (lead, 1, lat, lon)
    tracks = track(prob, LAT, LON, leads, "2020-05-15", ["heavy_rain"])
    assert len(tracks) == 1
    t = tracks[0]
    assert len(t["steps"]) == len(leads)
    assert t["envelope"]["lat_max"] > t["steps"][0]["bbox"][2]


def test_region_touching_a_low_is_split_from_the_background_band():
    field = np.clip(_blob(15, 86, 3) + (LAT[:, None] <= 6) * np.ones((1, len(LON))), 0, 1)
    field[:, np.searchsorted(LON, 86)] = np.maximum(field[:, np.searchsorted(LON, 86)], (LAT <= 15))
    plain = detect(field, LAT, LON)
    split = detect(field, LAT, LON, lows=[(15.0, 86.0)])
    assert len(plain) == 1  # the storm and the band are one connected region
    cyc = [o for o in split if "low" in o]
    assert len(cyc) == 1 and len(split) >= 2
    assert cyc[0]["bbox"][2] - cyc[0]["bbox"][0] <= 12  # no wider than the cyclone radius
    assert abs(cyc[0]["centroid"][0] - 15) < 3
