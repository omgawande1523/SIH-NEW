"""Read any NWP ensemble file (NetCDF, GRIB2, Zarr) into the pipeline's standard layout.

Standard layout: an xarray.Dataset with dims (member, lead, lat, lon), lat ascending,
lon in degrees east, lead in hours, and short variable names (tp24, tp6, ws10, t2m, mslp).
This is the single entry point to use for NCMRWF NEPS-G / NCUM or IMDAA files, e.g.

    ds = open_forecast("neps_g_2020051500.grib2", engine="cfgrib")
    ds = open_forecast("ncum_*.nc", chunks={"lead": 8})   # Dask-backed, lazily loaded

Deterministic runs (NCUM) come back with a single member so the same code path applies.
"""
from __future__ import annotations

import glob

import numpy as np
import xarray as xr

# Common spellings found in NCMRWF / ECMWF / IMDAA files -> our short names.
VAR_ALIASES = {
    "tp24": ["tp24", "total_precipitation_24hr", "APCP_24", "tp_24h"],
    "tp6": ["tp6", "total_precipitation_6hr", "APCP_6", "tp"],
    "ws10": ["ws10", "10m_wind_speed", "si10", "wind10"],
    "u10": ["u10", "10u", "10m_u_component_of_wind", "UGRD_10m"],
    "v10": ["v10", "10v", "10m_v_component_of_wind", "VGRD_10m"],
    "t2m": ["t2m", "2t", "2m_temperature", "TMP_2m"],
    "mslp": ["mslp", "msl", "prmsl", "mean_sea_level_pressure", "PRMSL"],
}
DIM_ALIASES = {
    "member": ["member", "number", "ensemble", "realization", "ens"],
    "lead": ["lead", "step", "prediction_timedelta", "forecast_period", "fcst_hr"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "longitude", "x"],
}


def _rename_map(ds, aliases):
    out = {}
    for std, names in aliases.items():
        for n in names:
            if n in ds.variables or n in ds.dims:
                if n != std:
                    out[n] = std
                break
    return out


def standardize(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.rename(_rename_map(ds, DIM_ALIASES))
    ds = ds.rename(_rename_map(ds, VAR_ALIASES))
    if "member" not in ds.dims:
        ds = ds.expand_dims(member=[0])
    if "lead" not in ds.dims:
        ds = ds.expand_dims(lead=[0])
    lead = np.asarray(ds["lead"].values)
    if lead.dtype.kind == "m":
        ds = ds.assign_coords(lead=(lead / np.timedelta64(1, "h")).astype(int))
    if "ws10" not in ds and {"u10", "v10"} <= set(ds.data_vars):
        ds["ws10"] = np.hypot(ds.u10, ds.v10)
    ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lat").sortby("lon")
    keep = [v for v in VAR_ALIASES if v in ds.data_vars]
    return ds[keep].transpose("member", "lead", "lat", "lon", ...)


def open_forecast(path_or_glob: str, engine: str | None = None, chunks=None) -> xr.Dataset:
    """Open NetCDF / GRIB2 / Zarr forecast(s); GRIB2 needs `pip install cfgrib eccodes`."""
    if path_or_glob.endswith(".zarr") or path_or_glob.startswith("http"):
        ds = xr.open_zarr(path_or_glob, chunks=chunks)
    else:
        files = sorted(glob.glob(path_or_glob)) or [path_or_glob]
        if engine is None and files[0].endswith((".grib2", ".grb2", ".grib")):
            engine = "cfgrib"
        ds = xr.open_mfdataset(files, engine=engine, chunks=chunks or {}, combine="by_coords") \
            if len(files) > 1 else xr.open_dataset(files[0], engine=engine, chunks=chunks)
    return standardize(ds)


def crop(ds, dom):
    return ds.sel(lat=slice(dom.lat_min, dom.lat_max), lon=slice(dom.lon_min, dom.lon_max))
