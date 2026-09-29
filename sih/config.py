"""Central configuration for the SIH prototype.

Everything that a team member may want to change for the demo (domain, events,
thresholds, model sizes) lives here so the rest of the code stays generic.
"""
import os
from dataclasses import dataclass
from pathlib import Path

def _mkdir(p: Path):
    # On a read-only deployment (Vercel) the API only reads outputs/, and data/ and
    # models/ are left out of the bundle, so a folder that can't be created is fine.
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
_mkdir(OUT)

# Data source: "era5" (real ECMWF ensemble + ERA5, downloaded once) or "synthetic" (offline).
# Each source keeps its own cache and trained weights.
SOURCE = CACHE = MODELS = None


def use_source(src: str):
    global SOURCE, CACHE, MODELS
    SOURCE, CACHE, MODELS = src, DATA / "cache" / src, ROOT / "models" / src
    _mkdir(CACHE)
    _mkdir(MODELS)


use_source(os.environ.get("SIH_SOURCE", "era5"))

# Public, credential-free mirrors of ERA5 and the ECMWF IFS ensemble (WeatherBench 2).
WB2 = "https://storage.googleapis.com/weatherbench2/datasets"
ERA5_COARSE = f"{WB2}/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"  # 1.5 deg
ERA5_FINE = f"{WB2}/era5/1959-2022-6h-1440x721.zarr"  # 0.25 deg
ENS = f"{WB2}/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr"  # 50 members, 1.5 deg


@dataclass
class Domain:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# Stage 1 works on the Indian region of the global ensemble.
STAGE1_DOMAIN = Domain(0.0, 36.0, 60.0, 100.5)
# Stage 2 (downscaling) trains on the Bay of Bengal / east India box at 0.25 deg.
STAGE2_DOMAIN = Domain(5.0, 30.0, 70.0, 100.0)

# Icosahedral mesh refinement level: 5 gives ~220 km node spacing (10,242 nodes globally),
# each extra level halves the spacing. 12 km NEPS-G would use level 9.
MESH_LEVEL = 5

# Forecast lead times used by the tracker (hours): 0..240 h every 6 h.
LEADS_H = list(range(0, 241, 6))

# Variables the tracker screens. Keys are our short names, values the WB2 names.
STAGE1_VARS = {
    "tp24": "total_precipitation_24hr",   # m, 24 h accumulation
    "ws10": "10m_wind_speed",             # m/s
    "t2m": "2m_temperature",              # K (heat domes / cold waves)
}


@dataclass
class Event:
    name: str
    init: str  # ensemble initialisation (00 UTC)
    kind: str
    split: str  # train | test


# Documented Indian extreme events available in the public ensemble archive.
EVENTS = [
    Event("fani", "2019-04-27", "cyclone", "train"),
    Event("heatwave2019", "2019-05-28", "heatwave", "train"),
    Event("vayu", "2019-06-09", "cyclone", "train"),
    Event("tauktae", "2021-05-12", "cyclone", "train"),
    Event("yaas", "2021-05-21", "cyclone", "train"),
    Event("quiet2018", "2018-05-02", "none", "train"),
    Event("amphan", "2020-05-15", "cyclone", "test"),
]
DEMO_EVENT = "amphan"

# Climatology for the EFI: ERA5 1990-2019 (30 years), +-15 days around each valid date.
CLIM_YEARS = (1990, 2019)
CLIM_DOY = (100, 190)  # covers the April-June valid dates of the events above
CLIM_HALF_WINDOW_DAYS = 15
CLIM_QUANTILES = [i / 100 for i in range(101)]

# Stage 2 training sample: random 6-hourly ERA5 fields from the pre-monsoon and monsoon.
STAGE2_TRAIN_YEARS = (2010, 2019)
STAGE2_TRAIN_SAMPLES = 240
DOWNSCALE_FACTOR = 6  # 1.5 deg ensemble -> 0.25 deg (same code does 12 km -> 5 km)
PATCH = 48  # fine-grid patch size (48 x 0.25 deg = 12 deg)

# Alert categories on 6-hourly rainfall (mm / 6 h), scaled from IMD daily classes:
# heavy 64.5, very heavy 115.6, extremely heavy 204.5 mm/day.
ALERT_THRESHOLDS_MM6H = {"low": 16.0, "moderate": 29.0, "severe": 51.0}
ALERT_RADIUS_KM = 5.0

