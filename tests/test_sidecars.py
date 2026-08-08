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


def test_build_aborts_when_provenance_cannot_be_pinned(tmp_path, monkeypatch):
    """A provenance failure must ABORT the build (no untraceable parquets), not warn."""
    import openscope_ccf.sidecar as sc
    import openscope_ccf.provenance as prov

    class _FH:  # dummy file handle; the builders are stubbed so it's never read
        def close(self):
            pass

    monkeypatch.setattr(sc, "open_remote", lambda aid, **k: _FH())
    monkeypatch.setattr(sc, "build_unit_sidecar", lambda fh, s, d, p: pd.DataFrame({"unit_index": [0, 1]}))
    monkeypatch.setattr(sc, "build_channel_sidecar", lambda fh, s, d, p: pd.DataFrame({"electrode_row": [0]}))
    # record() raises because the asset can't be pinned (e.g. replaced on the draft)
    monkeypatch.setattr(prov, "record",
                        lambda *a, **k: (_ for _ in ()).throw(prov.ProvenanceError("asset replaced")))

    with pytest.raises(prov.ProvenanceError):
        sc.build_session_sidecars("stale-aid", "830794", "2026-01-26-12-02-05", "sensorimotor",
                                  outdir=str(tmp_path))
    # no parquets and no manifest left behind
    assert list(tmp_path.glob("*.parquet")) == []
    assert not (tmp_path / "provenance.jsonl").exists()


def test_build_can_opt_out_of_provenance(tmp_path, monkeypatch):
    """record_provenance=False writes sidecars without touching DANDI/provenance."""
    import openscope_ccf.sidecar as sc

    class _FH:
        def close(self):
            pass

    monkeypatch.setattr(sc, "open_remote", lambda aid, **k: _FH())
    monkeypatch.setattr(sc, "build_unit_sidecar", lambda fh, s, d, p: pd.DataFrame({"unit_index": [0, 1]}))
    monkeypatch.setattr(sc, "build_channel_sidecar", lambda fh, s, d, p: pd.DataFrame({"electrode_row": [0]}))
    out = sc.build_session_sidecars("any-aid", "830794", "2026-01-26-12-02-05", "sensorimotor",
                                    outdir=str(tmp_path), record_provenance=False)
    assert out["units"].exists() and out["channels"].exists()
    assert not (tmp_path / "provenance.jsonl").exists()
