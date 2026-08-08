"""Provenance records for reproducibility against a mutable DANDI draft.

DANDI 001637 is draft-only: asset ids are re-minted on re-upload and the draft
is mutable, so "resolved by path at run time" (the default everywhere in this
package) is convenient but not, by itself, a reproducible pin. A provenance
record captures *exactly what was read* so a result can be traced to an
immutable content state even after the draft moves:

* ``asset_id``      — the DANDI asset id resolved for this session at run time
* ``content_sha256``— the asset's SHA-256 content digest (from DANDI metadata,
  no download needed); the immutable fingerprint of the file that was read
* ``content_size``  — byte size, a cheap corroborating check
* ``dandiset`` / ``version`` — dataset coordinates (e.g. 001637 / draft)
* ``code_sha``      — git commit of this repo at run time (or ``None`` if not a
  git checkout)
* ``params``        — the analysis parameters that affect the numbers
* ``created_at``    — UTC timestamp

:func:`record` builds one dict; :func:`asset_provenance` fills the DANDI fields
from the API by ``(subject, date)``; :func:`append_manifest` appends a record to
a JSONL manifest so a batch accumulates a full provenance log.
"""
from __future__ import annotations
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests

from .nwbio import DANDISET, resolve_asset

_ASSET_META = "https://api.dandiarchive.org/api/dandisets/{ds}/versions/{ver}/assets/{aid}/"


def code_sha(repo_root=None) -> "str | None":
    """Current git commit SHA of the repo, or ``None`` if not a git checkout."""
    root = str(repo_root) if repo_root else str(Path(__file__).resolve().parents[1])
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


class ProvenanceError(RuntimeError):
    """Raised when an immutable provenance record cannot be built for the asset read."""


def _asset_meta(aid: str, dandiset: str, version: str) -> dict:
    r = requests.get(_ASSET_META.format(ds=dandiset, ver=version, aid=aid), timeout=30)
    if r.status_code != 200:
        raise ProvenanceError(f"asset {aid} metadata HTTP {r.status_code} "
                              f"(asset may have been replaced on the {version} draft)")
    return r.json()


def asset_provenance(subject: str, date: str, dandiset: str = DANDISET,
                     version: str = "draft", asset_id: "str | None" = None) -> dict:
    """Return the immutable DANDI fingerprint fields for a session's asset.

    ``asset_id`` is the id that was **actually read**. Pass it so provenance
    describes the exact asset opened — NOT a fresh path re-resolution, which
    could return a different asset if the mutable draft moved between the read
    and this call (a time-of-check/time-of-use hazard). When ``asset_id`` is
    given, the asset's own ``path`` metadata is verified to contain the subject
    (and, when present, the date); a mismatch raises :class:`ProvenanceError`.
    When omitted, falls back to resolving by ``(subject, date)`` — only safe when
    no read has happened yet.

    Returns ``{asset_id, content_sha256, dandi_etag, content_size, path,
    dandiset, version}``. The SHA-256 comes from DANDI metadata — no download.
    """
    aid = asset_id or resolve_asset(subject, date, dandiset=dandiset, version=version)
    meta = _asset_meta(aid, dandiset, version)
    path = meta.get("path", "")
    if asset_id is not None:
        # Verify the asset we fingerprinted is the session we think it is.
        if f"sub-{subject}" not in path:
            raise ProvenanceError(f"asset {aid} path {path!r} does not match subject {subject}")
        if str(date) and str(date) not in path:
            raise ProvenanceError(f"asset {aid} path {path!r} does not match date {date}")
    digest = (meta.get("digest") or {})
    sha = digest.get("dandi:sha2-256")
    if sha is None and digest.get("dandi:dandi-etag") is None:
        raise ProvenanceError(f"asset {aid} has no content digest in DANDI metadata")
    return dict(
        asset_id=aid,
        content_sha256=sha,
        dandi_etag=digest.get("dandi:dandi-etag"),
        content_size=meta.get("contentSize"),
        path=path,
        dandiset=dandiset,
        version=version,
    )


def record(subject: str, date: str, params: "dict | None" = None,
           dandiset: str = DANDISET, version: str = "draft",
           repo_root=None, asset_id: "str | None" = None) -> dict:
    """Build a full provenance record for a session + analysis params.

    Pass ``asset_id`` = the id actually read so the digest describes that exact
    asset (see :func:`asset_provenance`). Raises :class:`ProvenanceError` if an
    immutable record cannot be built.
    """
    prov = asset_provenance(subject, date, dandiset=dandiset, version=version, asset_id=asset_id)
    prov.update(
        subject=str(subject), date=str(date),
        code_sha=code_sha(repo_root),
        params=params or {},
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return prov


def append_manifest(record_dict: dict, manifest_path) -> Path:
    """Append one provenance record to a JSONL manifest (one record per line)."""
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record_dict, default=str) + "\n")
    return p
