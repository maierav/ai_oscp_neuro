"""Tests for the provenance module (openscope_ccf.provenance).

The DANDI-touching path is monkeypatched; the manifest/append and code_sha
logic run offline.
"""
import json

import openscope_ccf.provenance as prov


def test_code_sha_is_str_or_none():
    s = prov.code_sha()
    assert s is None or (isinstance(s, str) and len(s) >= 7)


def test_append_manifest_roundtrip(tmp_path):
    m = tmp_path / "sub" / "provenance.jsonl"
    r1 = {"subject": "830794", "asset_id": "aaa", "content_sha256": "deadbeef"}
    r2 = {"subject": "830795", "asset_id": "bbb", "content_sha256": "cafef00d"}
    prov.append_manifest(r1, m)
    prov.append_manifest(r2, m)
    lines = m.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["asset_id"] == "aaa"
    assert json.loads(lines[1])["content_sha256"] == "cafef00d"


def test_record_assembles_fields(monkeypatch):
    # stub the DANDI-touching resolver so this stays offline (new signature takes asset_id)
    monkeypatch.setattr(prov, "asset_provenance",
                        lambda subject, date, dandiset="001637", version="draft", asset_id=None: dict(
                            asset_id=asset_id or "fake-aid", content_sha256="abc123",
                            dandi_etag="etag-1", content_size=42, path=f"sub-{subject}/x.nwb",
                            dandiset=dandiset, version=version))
    r = prov.record("830794", "2026-01-26-12-02-05", params={"win": [0, 0.3]}, asset_id="used-aid")
    assert r["asset_id"] == "used-aid"          # the id actually READ, not a re-resolution
    assert r["content_sha256"] == "abc123"
    assert r["subject"] == "830794" and r["date"] == "2026-01-26-12-02-05"
    assert r["params"] == {"win": [0, 0.3]}
    assert "created_at" in r and r["dandiset"] == "001637"


def test_asset_provenance_verifies_path(monkeypatch):
    # exact asset_id whose path matches subject+date -> ok; mismatch -> ProvenanceError
    def fake_meta(aid, ds, ver):
        return {"aid-830794": {"path": "sub-830794/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys.nwb",
                               "digest": {"dandi:sha2-256": "abc"}, "contentSize": 10},
                "aid-other":   {"path": "sub-830846/sub-830846_ses-ecephys-830846-2026-03-11-10-19-32_ecephys.nwb",
                               "digest": {"dandi:sha2-256": "def"}, "contentSize": 20}}[aid]
    monkeypatch.setattr(prov, "_asset_meta", fake_meta)
    good = prov.asset_provenance("830794", "2026-01-26-12-02-05", asset_id="aid-830794")
    assert good["asset_id"] == "aid-830794" and good["content_sha256"] == "abc"
    # a stale/wrong asset id for this session must raise, not silently record the wrong digest
    import pytest
    with pytest.raises(prov.ProvenanceError):
        prov.asset_provenance("830794", "2026-01-26-12-02-05", asset_id="aid-other")


def test_asset_provenance_requires_digest(monkeypatch):
    monkeypatch.setattr(prov, "_asset_meta",
                        lambda aid, ds, ver: {"path": "sub-830794/x.nwb", "digest": {}, "contentSize": 5})
    import pytest
    with pytest.raises(prov.ProvenanceError):
        prov.asset_provenance("830794", "", asset_id="aid-nodigest")
