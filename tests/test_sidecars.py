"""Integrity tests for the shipped CCF sidecars (no network).

These pin the properties claimed in the README: the 30 shipped unit/channel
sidecar pairs have unique+contiguous keys, their row counts match ``_manifest.csv``,
and re-running ``decode_ccf`` on the stored acronym reproduces the stored
area/layer/group/tissue columns exactly. Also checks ``load_ccf`` raises a
helpful error for a session without a shipped sidecar.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openscope_ccf.ccf import decode_ccf
from openscope_ccf.sidecar import load_ccf

SD = Path(__file__).resolve().parents[1] / "openscope_ccf" / "data" / "sidecars"

UNIT_FILES = sorted(glob.glob(str(SD / "units_*.parquet")))
CHAN_FILES = sorted(glob.glob(str(SD / "channels_*.parquet")))


def test_thirty_pairs_shipped():
    assert len(UNIT_FILES) == 30
    assert len(CHAN_FILES) == 30


def _tag(path, kind):
    return Path(path).name[len(kind) + 1:-len(".parquet")]


@pytest.mark.parametrize("path", UNIT_FILES, ids=[_tag(p, "units") for p in UNIT_FILES])
def test_unit_keys_unique_and_contiguous(path):
    df = pd.read_parquet(path, columns=["unit_index"])
    k = df["unit_index"].to_numpy()
    assert len(set(k.tolist())) == len(k), "unit_index not unique"
    assert np.array_equal(np.sort(k), np.arange(k.min(), k.min() + len(k))), "unit_index not contiguous"


@pytest.mark.parametrize("path", CHAN_FILES, ids=[_tag(p, "channels") for p in CHAN_FILES])
def test_channel_keys_unique_and_contiguous(path):
    df = pd.read_parquet(path, columns=["electrode_row"])
    k = df["electrode_row"].to_numpy()
    assert len(set(k.tolist())) == len(k), "electrode_row not unique"
    assert np.array_equal(np.sort(k), np.arange(k.min(), k.min() + len(k))), "electrode_row not contiguous"


def test_row_counts_match_manifest():
    man = pd.read_csv(SD / "_manifest.csv").set_index("tag")
    for path in UNIT_FILES:
        tag = _tag(path, "units")
        assert tag in man.index, f"{tag} missing from manifest"
        n = len(pd.read_parquet(path, columns=["unit_index"]))
        assert n == int(man.loc[tag, "n_units"]), f"{tag}: {n} rows != manifest {man.loc[tag, 'n_units']}"


@pytest.mark.parametrize("path", UNIT_FILES, ids=[_tag(p, "units") for p in UNIT_FILES])
def test_stored_decoding_matches_decode_ccf(path):
    df = pd.read_parquet(path)
    for _, r in df.iterrows():
        d = decode_ccf(r["ccf_acronym"])
        for col in ("area", "layer", "group", "tissue"):
            stored = r[col]
            stored = None if (stored is None or (isinstance(stored, float) and pd.isna(stored))) else stored
            assert str(d[col]) == str(stored), f"{path} unit {r['unit_index']} {col}: {d[col]!r} != {stored!r}"


def test_load_ccf_missing_sidecar_raises_helpful():
    with pytest.raises(FileNotFoundError) as e:
        load_ccf("999999_2099-01-01-00-00-00", sidecar_dir=str(SD))
    assert "build_session_sidecars" in str(e.value)
