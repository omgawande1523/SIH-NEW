"""Pull small regional subsets of public ERA5 and ECMWF IFS ensemble data.

The WeatherBench 2 archive on Google Cloud Storage mirrors ERA5 reanalysis and the
50-member ECMWF IFS ensemble as Zarr, readable anonymously over HTTPS. We stream
only the chunks we need, crop to the Indian region, and cache NetCDF files under
data/cache so every later run is offline.

Real NEPS-G / NCUM / IMDAA files drop in through sih.data.loader instead.
"""
from __future__ import annotations

import argparse
import time

import dask
import numpy as np
import pandas as pd
import xarray as xr

from sih import config as C

dask.config.set(scheduler="threads", num_workers=24)


def _open(url):
    return xr.open_zarr(url, consolidated=True)


def _hours(td):
    td = np.asarray(td)
    return (td / np.timedelta64(1, "h")).astype(int) if td.dtype.kind == "m" else td.astype(int)


def _crop(ds, dom: C.Domain):
    ds = ds.sortby("latitude")
    return ds.sel(latitude=slice(dom.lat_min, dom.lat_max), longitude=slice(dom.lon_min, dom.lon_max))


def _save(ds, path):
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print(f"  saved {path.name} ({path.stat().st_size/1e6:.1f} MB)")


def fetch_ensemble(event: C.Event, extra_vars=()):
    """Ensemble forecast for one initialisation: dims (member, lead, lat, lon)."""
    path = C.CACHE / f"ens_{event.name}.nc"
    if path.exists():
        return path
    t0 = time.time()
    ds = _open(C.ENS)
    names = list(C.STAGE1_VARS.values()) + list(extra_vars)
    sub = ds[names].sel(time=np.datetime64(event.init + "T00:00"))
    lead_h = _hours(sub.prediction_timedelta.values)
    sub = sub.isel(prediction_timedelta=[int(np.where(lead_h == h)[0][0]) for h in C.LEADS_H])
    sub = _crop(sub, C.STAGE1_DOMAIN).load()
    inv = {v: k for k, v in C.STAGE1_VARS.items()}
    inv.update({"mean_sea_level_pressure": "mslp", "total_precipitation_6hr": "tp6",
                "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10"})
    sub = sub.rename({k: v for k, v in inv.items() if k in sub}).rename(
        number="member", prediction_timedelta="lead", latitude="lat", longitude="lon")
    sub = sub.assign_coords(lead=_hours(sub.lead.values))
    sub = sub.transpose("member", "lead", "lat", "lon").astype("float32")
    sub.attrs.update(init=event.init, event=event.name, kind=event.kind, source="ECMWF IFS ENS via WeatherBench2")
    _save(sub, path)
    print(f"  ensemble {event.name}: {time.time()-t0:.0f}s")
    return path


def fetch_ensemble_mslp(event: C.Event):
    """Ensemble MSLP alone (member, lead, lat, lon), for events whose ensemble file lacks it.

    Used by scripts/tracker_check.py to test the cyclone split on the training events.
    """
    path = C.CACHE / f"ens_mslp_{event.name}.nc"
    if path.exists():
        return path
    ds = _open(C.ENS)
    sub = ds[["mean_sea_level_pressure"]].sel(time=np.datetime64(event.init + "T00:00"))
    lead_h = _hours(sub.prediction_timedelta.values)
    sub = sub.isel(prediction_timedelta=[int(np.where(lead_h == h)[0][0]) for h in C.LEADS_H])
    sub = _crop(sub, C.STAGE1_DOMAIN).load()
    sub = sub.rename(mean_sea_level_pressure="mslp", number="member", prediction_timedelta="lead",
                     latitude="lat", longitude="lon")
    sub = sub.assign_coords(lead=_hours(sub.lead.values)).transpose("member", "lead", "lat", "lon").astype("float32")
    _save(sub, path)
    return path


def fetch_verification(event: C.Event):
    """ERA5 analysis at the forecast valid times (labels for training the GNN)."""
    path = C.CACHE / f"verif_{event.name}.nc"
    if path.exists():
        return path
    ds = _open(C.ERA5_COARSE)
    valid = pd.Timestamp(event.init) + pd.to_timedelta(C.LEADS_H, unit="h")
    sub = ds[list(C.STAGE1_VARS.values())].sel(time=valid.values)
    sub = _crop(sub, C.STAGE1_DOMAIN).load()
    sub = sub.rename({v: k for k, v in C.STAGE1_VARS.items()}).rename(latitude="lat", longitude="lon")
    sub = sub.transpose("time", "lat", "lon").astype("float32")
    _save(sub, path)
    return path


def fetch_climatology():
    """30-year ERA5 sample used to build the climate CDF for the EFI."""
    path = C.CACHE / "clim_samples.nc"
    if path.exists():
        return path
    t0 = time.time()
    ds = _open(C.ERA5_COARSE)
    t = pd.DatetimeIndex(ds.time.values)
    keep = (t.year >= C.CLIM_YEARS[0]) & (t.year <= C.CLIM_YEARS[1]) & \
           (t.dayofyear >= C.CLIM_DOY[0]) & (t.dayofyear <= C.CLIM_DOY[1])
    sub = ds[list(C.STAGE1_VARS.values())].isel(time=np.where(keep)[0])
    sub = _crop(sub, C.STAGE1_DOMAIN)
    print(f"  climatology: {sub.sizes['time']} time steps x {len(C.STAGE1_VARS)} vars, streaming...")
    sub = sub.load()
    sub = sub.rename({v: k for k, v in C.STAGE1_VARS.items()}).rename(latitude="lat", longitude="lon")
    sub = sub.transpose("time", "lat", "lon").astype("float32")
    _save(sub, path)
    print(f"  climatology: {time.time()-t0:.0f}s")
    return path


FINE_VARS = {
    "total_precipitation_6hr": "tp6",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_column_water_vapour": "tcwv",
}


def _fine_subset(times):
    ds = _open(C.ERA5_FINE)
    sub = ds[list(FINE_VARS)].sel(time=times)
    sub = _crop(sub, C.STAGE2_DOMAIN).load()
    return sub.rename(FINE_VARS).rename(latitude="lat", longitude="lon").astype("float32")


def fetch_fine_static():
    path = C.CACHE / "fine_static.nc"
    if path.exists():
        return path
    ds = _open(C.ERA5_FINE)
    sub = _crop(ds[["geopotential_at_surface", "land_sea_mask"]], C.STAGE2_DOMAIN).load()
    sub = sub.rename(geopotential_at_surface="orog", land_sea_mask="lsm", latitude="lat", longitude="lon")
    sub["orog"] = sub["orog"] / 9.80665
    _save(sub.astype("float32"), path)
    return path


def fetch_fine_training(seed=0):
    """Random 0.25 deg ERA5 fields (Apr-Oct) used to train the downscaler."""
    path = C.CACHE / "fine_train.nc"
    if path.exists():
        return path
    t0 = time.time()
    rng = np.random.default_rng(seed)
    days = pd.date_range(f"{C.STAGE2_TRAIN_YEARS[0]}-01-01", f"{C.STAGE2_TRAIN_YEARS[1]}-12-31 18:00", freq="6h")
    days = days[(days.month >= 4) & (days.month <= 11)]
    times = np.sort(rng.choice(days.values, C.STAGE2_TRAIN_SAMPLES, replace=False))
    sub = _fine_subset(times)
    _save(sub, path)
    print(f"  fine training set: {time.time()-t0:.0f}s")
    return path


def fetch_fine_truth(event: C.Event, days=(3, 7)):
    """0.25 deg ERA5 'truth' during the test event, to score the downscaler."""
    path = C.CACHE / f"fine_truth_{event.name}.nc"
    if path.exists():
        return path
    t0 = pd.Timestamp(event.init)
    times = pd.date_range(t0 + pd.Timedelta(days=days[0]), t0 + pd.Timedelta(days=days[1]), freq="6h")
    _save(_fine_subset(times.values), path)
    return path


def fetch_all(training=True):
    """Everything for training; with training=False only what the demo event needs.

    The shipped zip already contains the demo-event files, so a run with trained weights
    works offline; anything missing is downloaded here.
    """
    print("Checking public ERA5 + ECMWF ensemble subsets (downloaded once, then cached)")
    for ev in C.EVENTS:
        if not training and ev.split != "test":
            continue
        extra = ("mean_sea_level_pressure", "total_precipitation_6hr", "10m_u_component_of_wind",
                 "10m_v_component_of_wind") if ev.split == "test" else ()
        fetch_ensemble(ev, extra)
        fetch_verification(ev)
    from sih.stage1.efi import cached_features_ok
    if training or not all(cached_features_ok(ev.name) for ev in C.EVENTS if ev.split == "test"):
        fetch_climatology()
    fetch_fine_static()
    if training:
        fetch_fine_training()
    for ev in C.EVENTS:
        if ev.split == "test":
            fetch_fine_truth(ev)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    fetch_all()
