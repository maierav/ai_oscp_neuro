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
from .nwbio import (s3_url, open_remote, unit_electrode_rows, electrodes_frame,
                    session_path, resolve_asset, open_session)
from .sidecar import (build_unit_sidecar, build_channel_sidecar,
                      build_session_sidecars, load_ccf, attach)
from .figures import build_probe_data, make_3d, make_laminar, load_root_mesh

__version__ = "0.1.0"


def load_session_index(refresh_aids: bool = False) -> pd.DataFrame:
    """Return the registry of CCF-labeled ecephys sessions (DANDI 001637).

    The shipped ``aid`` column is a snapshot: DANDI asset ids change when a
    session file is re-uploaded (the path is stable, the id is not), and 001637
    is draft-only so no id is permanent. Pass ``refresh_aids=True`` to re-resolve
    every row's asset id from the live dandiset by path — self-healing against
    re-uploads at the cost of one API call per session. Rows whose path no longer
    resolves keep their stored id and are flagged in a new ``aid_stale`` column.
    """
    idx = pd.read_csv(files("openscope_ccf").joinpath("data/ccf_session_index.csv"))
    if not refresh_aids:
        return idx
    from .nwbio import resolve_asset
    cur, stale = [], []
    for _, row in idx.iterrows():
        try:
            aid = resolve_asset(str(row.subject), str(row.date))
            cur.append(aid)
            stale.append(aid != row.aid)
        except LookupError:
            cur.append(row.aid)
            stale.append(True)
    idx = idx.copy()
    idx["aid"] = cur
    idx["aid_stale"] = stale
    return idx


__all__ = [
    "decode_ccf", "decode_many", "FIBER_TRACTS", "UNASSIGNED",
    "s3_url", "open_remote", "unit_electrode_rows", "electrodes_frame",
    "session_path", "resolve_asset", "open_session",
    "build_unit_sidecar", "build_channel_sidecar", "build_session_sidecars",
    "load_ccf", "attach",
    "build_probe_data", "make_3d", "make_laminar", "load_root_mesh",
    "load_session_index", "__version__",
]
