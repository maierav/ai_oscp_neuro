"""Build and attach CCF *sidecar* tables.

A sidecar is a small tidy table that carries the CCF annotation for one session,
keyed so it joins directly onto any downstream analysis:

* **unit sidecar** — one row per unit, key ``unit_index`` (0-based row in the NWB
  ``units`` table). Columns: probe, ccf_acronym, area, layer, group, tissue,
  qc_pass, firing_rate, ccf_ap/dv/ml.
* **channel sidecar** — one row per electrode/channel, key
  ``electrode_row`` (0-based row in the NWB ``electrodes`` table). Same CCF
  columns. Use this to annotate LFP/CSD channels.

Sidecars are written as Parquet (language-agnostic, fast). :func:`load_ccf`
reads one back; :func:`attach` left-joins it onto a caller's DataFrame by the
appropriate key. Because the keys are just the NWB table row indices, any SUA /
MUA / LFP / CSD result that carries ``unit_index`` or ``electrode_row`` (or
``channel``) can be annotated with one call.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .nwbio import open_remote, unit_electrode_rows, electrodes_frame, _decode
from .ccf import decode_ccf


def build_unit_sidecar(fh, subject: str, date: str, paradigm: str) -> pd.DataFrame:
    """Per-unit CCF sidecar from an open NWB file handle."""
    u = fh["units"]
    el = fh["general/extracellular_ephys/electrodes"]
    loc = _decode(el["location"][:])
    ex, ey, ez = el["x"][:], el["y"][:], el["z"][:]
    dev = _decode(u["device_name"][:])
    elrow = unit_electrode_rows(fh)
    qc = u["default_qc"][:] if "default_qc" in u else np.full(len(dev), True)
    fr = u["firing_rate"][:] if "firing_rate" in u else np.full(len(dev), np.nan)
    rows = []
    for i in range(len(dev)):
        acr = loc[elrow[i]]
        d = decode_ccf(acr)
        rows.append(dict(
            unit_index=i, subject=subject, date=date, paradigm=paradigm,
            probe=dev[i], ccf_acronym=acr, area=d["area"], layer=d["layer"],
            group=d["group"], tissue=d["tissue"],
            qc_pass=bool(qc[i]), firing_rate=float(fr[i]),
            ccf_ap=float(ex[elrow[i]]), ccf_dv=float(ey[elrow[i]]), ccf_ml=float(ez[elrow[i]]),
        ))
    return pd.DataFrame(rows)


def build_channel_sidecar(fh, subject: str, date: str, paradigm: str) -> pd.DataFrame:
    """Per-channel (electrode) CCF sidecar from an open NWB file handle."""
    ef = electrodes_frame(fh)
    dec = pd.DataFrame([decode_ccf(a) for a in ef["location"]])
    out = pd.concat([ef, dec], axis=1)
    out.insert(0, "subject", subject)
    out.insert(1, "date", date)
    out.insert(2, "paradigm", paradigm)
    return out


def build_session_sidecars(asset_id: str, subject: str, date: str, paradigm: str,
                           outdir="data/sidecars") -> "dict[str, Path]":
    """Stream a session and write both sidecars as Parquet. Returns their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fh = open_remote(asset_id)
    try:
        us = build_unit_sidecar(fh, subject, date, paradigm)
        cs = build_channel_sidecar(fh, subject, date, paradigm)
    finally:
        fh.close()
    tag = f"{subject}_{date}"
    up = outdir / f"units_{tag}.parquet"
    cp = outdir / f"channels_{tag}.parquet"
    us.to_parquet(up, index=False)
    cs.to_parquet(cp, index=False)
    return {"units": up, "channels": cp}


def load_ccf(session_tag: str, kind: str = "units", sidecar_dir="data/sidecars") -> pd.DataFrame:
    """Load a session's sidecar. ``kind`` is ``"units"`` or ``"channels"``."""
    p = Path(sidecar_dir) / f"{kind}_{session_tag}.parquet"
    return pd.read_parquet(p)


def attach(df: pd.DataFrame, session_tag: str, on: str = "unit_index",
           kind: str = "units", sidecar_dir="data/sidecars",
           cols=("ccf_acronym", "area", "layer", "group", "tissue",
                 "ccf_ap", "ccf_dv", "ccf_ml")) -> pd.DataFrame:
    """Left-join CCF annotation onto ``df``.

    Parameters
    ----------
    df : DataFrame with a key column (``unit_index`` for unit-level results, or
        ``electrode_row`` / ``channel`` for channel-level LFP/CSD results).
    on : the key column in ``df``.
    kind : which sidecar to use (``"units"`` or ``"channels"``).
    """
    side = load_ccf(session_tag, kind=kind, sidecar_dir=sidecar_dir)
    key = "unit_index" if kind == "units" else "electrode_row"
    keep = [key] + [c for c in cols if c in side.columns]
    return df.merge(side[keep], left_on=on, right_on=key, how="left")
