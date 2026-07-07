"""openscope_ccf — CCF annotation & penetration figures for OpenScope
Predictive Processing ecephys sessions (DANDI 001637).

Quick start
-----------
>>> from openscope_ccf import build_session_sidecars, attach, load_session_index
>>> idx = load_session_index()
>>> row = idx.iloc[0]
>>> build_session_sidecars(row.aid, row.subject, row.date, row.paradigm)
>>> # later, annotate any unit-level result:
>>> annotated = attach(my_sua_df, f"{row.subject}_{row.date}", on="unit_index")
"""
from importlib.resources import files
import pandas as pd

from .ccf import decode_ccf, decode_many, FIBER_TRACTS, UNASSIGNED
from .nwbio import s3_url, open_remote, unit_electrode_rows, electrodes_frame
from .sidecar import (build_unit_sidecar, build_channel_sidecar,
                      build_session_sidecars, load_ccf, attach)
from .figures import build_probe_data, make_3d, make_laminar, load_root_mesh

__version__ = "0.1.0"


def load_session_index() -> pd.DataFrame:
    """Return the registry of CCF-labeled ecephys sessions (DANDI 001637)."""
    return pd.read_csv(files("openscope_ccf").joinpath("data/ccf_session_index.csv"))


__all__ = [
    "decode_ccf", "decode_many", "FIBER_TRACTS", "UNASSIGNED",
    "s3_url", "open_remote", "unit_electrode_rows", "electrodes_frame",
    "build_unit_sidecar", "build_channel_sidecar", "build_session_sidecars",
    "load_ccf", "attach",
    "build_probe_data", "make_3d", "make_laminar", "load_root_mesh",
    "load_session_index", "__version__",
]
