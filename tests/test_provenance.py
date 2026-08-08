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
    # stub the DANDI-touching resolver so this stays offline
    monkeypatch.setattr(prov, "asset_provenance",
                        lambda subject, date, dandiset="001637", version="draft": dict(
                            asset_id="fake-aid", content_sha256="abc123",
                            dandi_etag="etag-1", content_size=42,
                            dandiset=dandiset, version=version))
    r = prov.record("830794", "2026-01-26-12-02-05", params={"win": [0, 0.3]})
    assert r["asset_id"] == "fake-aid"
    assert r["content_sha256"] == "abc123"
    assert r["subject"] == "830794" and r["date"] == "2026-01-26-12-02-05"
    assert r["params"] == {"win": [0, 0.3]}
    assert "created_at" in r and r["dandiset"] == "001637"
