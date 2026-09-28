import sys
from pathlib import Path

import torch  # noqa: F401  (import before netCDF/HDF5 libraries: avoids an MKL loader clash)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
