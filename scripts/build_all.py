#!/usr/bin/env python
"""Batch-build CCF sidecars and/or penetration figures for all sessions in the
registry (or a subset).

Examples
--------
    python scripts/build_all.py --sidecars
    python scripts/build_all.py --figures --paradigm sensorimotor
    python scripts/build_all.py --sidecars --figures --out data/
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openscope_ccf as o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecars", action="store_true", help="build per-session sidecar Parquets")
    ap.add_argument("--figures", action="store_true", help="build 3D + laminar figures")
    ap.add_argument("--paradigm", default=None, help="restrict to one paradigm")
    ap.add_argument("--subject", default=None, help="restrict to one subject id")
    ap.add_argument("--out", default="data", help="output root")
    args = ap.parse_args()
    if not (args.sidecars or args.figures):
        ap.error("nothing to do: pass --sidecars and/or --figures")

    # refresh_aids=True re-resolves each session's asset id from the live dandiset by path,
    # so a stale snapshot id (re-upload drift) doesn't silently pull a superseded file.
    idx = o.load_session_index(refresh_aids=True)
    if args.paradigm:
        idx = idx[idx.paradigm == args.paradigm]
    if args.subject:
        idx = idx[idx.subject.astype(str) == str(args.subject)]
    out = Path(args.out)
    sidecar_dir = out / "sidecars"
    fig_dir = out / "figures"

    mesh = None
    bg = None
    if args.figures:
        try:
            mesh = o.load_root_mesh()
            from brainglobe_atlasapi import BrainGlobeAtlas
            bg = BrainGlobeAtlas("allen_mouse_25um")
        except Exception as e:
            print(f"[warn] atlas unavailable, figures will render without brain shell: {str(e)[:80]}")

    t0 = time.time()
    failures = []
    skipped = []
    for i, row in idx.reset_index(drop=True).iterrows():
        tag = f"{row.subject}_{row.date}"
        # Never build against an unresolved (stale) asset id — the path did not
        # resolve on the live dandiset, so row.aid is a superseded snapshot id.
        if bool(getattr(row, "aid_unresolved", False)):
            print(f"[{i+1}/{len(idx)}] {tag} SKIPPED: aid_unresolved (path not on live dandiset)")
            skipped.append(tag)
            continue
        try:
            if args.sidecars:
                o.build_session_sidecars(row.aid, str(row.subject), row.date, row.paradigm,
                                         outdir=sidecar_dir)
            if args.figures:
                fig_dir.mkdir(parents=True, exist_ok=True)
                pd_ = o.build_probe_data(row.aid)
                o.make_3d(pd_, f"sub-{row.subject}_{row.paradigm}", str(fig_dir / f"3d_{tag}.png"), brain_mesh=mesh)
                o.make_laminar(pd_, f"sub-{row.subject}_{row.paradigm}", str(fig_dir / f"laminar_{tag}.png"), bg=bg)
            print(f"[{i+1}/{len(idx)}] {tag} {row.paradigm} ok ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"[{i+1}/{len(idx)}] {tag} FAILED: {repr(e)[:120]}")
            failures.append((tag, repr(e)[:200]))
    print(f"done in {time.time()-t0:.0f}s — "
          f"{len(idx)-len(failures)-len(skipped)} ok, {len(failures)} failed, {len(skipped)} skipped")
    if failures:
        print("FAILURES:")
        for tag, err in failures:
            print(f"  {tag}: {err}")
    # Exit non-zero if anything failed or was skipped, so a broken batch cannot appear green in CI.
    if failures or skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
