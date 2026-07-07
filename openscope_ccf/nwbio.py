"""Stream OpenScope Predictive Processing ecephys NWB files from DANDI 001637
and map units to CCF-labeled electrodes.

The DANDI mirror serves standard HDF5 NWB (not Zarr), so we open the remote file
with :mod:`remfile` for chunked HTTP random access — no full download needed.

The key correctness detail is :func:`unit_electrode_rows`. The NWB
``units.extremum_channel_index`` is a *per-probe* channel index (0..N-1), but the
``electrodes`` table stacks all probes into one table. Indexing the electrodes
table directly by ``extremum_channel_index`` silently assigns every unit to the
first probe. The fix offsets each unit's index by the start row of its probe's
electrode block (looked up via ``units.device_name`` ->
``electrodes.group_name``).
"""
from __future__ import annotations
import requests
import numpy as np
import h5py
import remfile

DANDISET = "001637"
_API = "https://api.dandiarchive.org/api/dandisets/{ds}/versions/{ver}/assets/{aid}/download/"


def s3_url(asset_id: str, dandiset: str = DANDISET, version: str = "draft") -> str:
    """Resolve a DANDI asset id to a presigned S3 download URL.

    The returned URL is signed and expires; fetch a fresh one per session.
    """
    r = requests.get(_API.format(ds=dandiset, ver=version, aid=asset_id),
                     allow_redirects=False, timeout=30)
    r.raise_for_status()
    return r.headers["Location"]


def open_remote(asset_id: str, **kw) -> h5py.File:
    """Open a DANDI NWB asset as a read-only :class:`h5py.File` over HTTP."""
    return h5py.File(remfile.File(s3_url(asset_id, **kw)), "r")


def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def unit_electrode_rows(fh: h5py.File) -> np.ndarray:
    """Return, for each unit, the row index into the stacked electrodes table.

    Applies the per-probe offset correction described in the module docstring.
    """
    u = fh["units"]
    el = fh["general/extracellular_ephys/electrodes"]
    egrp = _decode(el["group_name"][:])
    dev = _decode(u["device_name"][:])
    eci = u["extremum_channel_index"][:]
    offset = {p: int(np.where(egrp == p)[0][0]) for p in sorted(set(egrp))}
    blocklen = {p: int((egrp == p).sum()) for p in offset}
    return np.array([offset[d] + min(int(c), blocklen[d] - 1) for d, c in zip(dev, eci)])


def electrodes_frame(fh: h5py.File):
    """Return the electrodes table as a DataFrame with CCF columns."""
    import pandas as pd
    el = fh["general/extracellular_ephys/electrodes"]
    return pd.DataFrame(dict(
        electrode_row=np.arange(len(el["x"])),
        group_name=_decode(el["group_name"][:]),
        location=_decode(el["location"][:]),
        ccf_ap=el["x"][:], ccf_dv=el["y"][:], ccf_ml=el["z"][:],
    ))
