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
from .provenance import record as provenance_record, asset_provenance, append_manifest, code_sha

__version__ = "0.1.0"


def load_session_index(refresh_aids: bool = False) -> pd.DataFrame:
    """Return the registry of CCF-labeled ecephys sessions (DANDI 001637).

    The shipped ``aid`` column is a snapshot: DANDI asset ids change when a
    session file is re-uploaded (the path is stable, the id is not), and 001637
    is draft-only so no id is permanent. Pass ``refresh_aids=True`` to re-resolve
    every row's asset id from the live dandiset by path — self-healing against
    re-uploads at the cost of one API call per session. Two flag columns are added
    (these are *opposite* states, deliberately not merged into one ``stale`` flag):

    - ``aid_changed``    : the live id differed from the shipped snapshot and was
                           updated — the row is now FRESH (a re-upload was healed).
    - ``aid_unresolved`` : the path did not resolve on the live dandiset, so the
                           shipped id was kept UNVERIFIED (do not trust it blindly).
    """
    idx = pd.read_csv(files("openscope_ccf").joinpath("data/ccf_session_index.csv"))
    if not refresh_aids:
        return idx
    from .nwbio import resolve_asset
    cur, changed, unresolved = [], [], []
    for _, row in idx.iterrows():
        try:
            aid = resolve_asset(str(row.subject), str(row.date))
            cur.append(aid)
            changed.append(aid != row.aid)   # refreshed to a new id -> now fresh
            unresolved.append(False)
        except Exception:
            # LookupError (path not found) OR a transient network/HTTP error
            # (resolve_asset calls raise_for_status): flag this one row unverified
            # rather than aborting the whole 58-session refresh on one 5xx.
            cur.append(row.aid)              # kept the shipped id, could not verify
            changed.append(False)
            unresolved.append(True)
    idx = idx.copy()
    idx["aid"] = cur
    idx["aid_changed"] = changed
    idx["aid_unresolved"] = unresolved
    return idx


__all__ = [
    "decode_ccf", "decode_many", "FIBER_TRACTS", "UNASSIGNED",
    "s3_url", "open_remote", "unit_electrode_rows", "electrodes_frame",
    "session_path", "resolve_asset", "open_session",
    "build_unit_sidecar", "build_channel_sidecar", "build_session_sidecars",
    "load_ccf", "attach",
    "build_probe_data", "make_3d", "make_laminar", "load_root_mesh",
    "provenance_record", "asset_provenance", "append_manifest", "code_sha",
    "load_session_index", "__version__",
]
