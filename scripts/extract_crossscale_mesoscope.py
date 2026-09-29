"""Parallel extraction of the mesoscope rows of data/crossscale_mechanism.parquet.

Runs the SAME per-session extractor (`img_mech`) defined in
notebooks/crossscale_oddball_index.ipynb, over the SAME mesoscope session list (the notebook's
`PATHS["mesoscope"]`), but one session per worker process. Streaming a mesoscope session takes
~20 min, so the notebook's serial loop over 23 sessions takes ~8 h; with 8 workers this takes ~75 min.

Usage (from the repo root):
    python scripts/extract_crossscale_mesoscope.py [--workers 8] [--out-dir .meso_parts]
    python scripts/extract_crossscale_mesoscope.py --merge      # write data/crossscale_mechanism.parquet

--merge keeps the committed ecephys + SLAP2 rows and replaces the mesoscope rows with the freshly
extracted sessions. Per-session parquet files are cached in --out-dir, so an interrupted run resumes.
"""
import argparse, os, re, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(REPO, "notebooks", "crossscale_oddball_index.ipynb")
OUTCOLS = ["area", "resp", "R_std", "R_o90", "R_c90", "R_c0", "resp_sign", "OI", "DvI", "TPI",
           "modality", "subject", "asset_id", "session_path", "uid"]
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _notebook_namespace():
    """Exec the notebook's helper + extractor cells, and read (not run) its session list."""
    import nbformat as nbf
    nb = nbf.read(NB, 4)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    strip = lambda s: "\n".join(l for l in s.split("\n") if not l.strip().startswith(("!", "%")))
    ns = {}
    for src in code[1:3]:            # streaming helpers + unified per-cell extractor
        exec(strip(src), ns)
    sess_src = code[3]               # sessions cell: take PATHS + resolve_by_path without resolving all
    exec(strip(sess_src[:sess_src.index("DS={")]), ns)
    return ns


def _one(subj, path, out_dir):
    out = os.path.join(out_dir, f"meso_{subj}_{DATE_RE.search(path).group(1)}.parquet")
    if os.path.exists(out):
        return out, "cached", 0.0
    import matplotlib
    matplotlib.use("Agg")
    ns = _notebook_namespace()
    t0 = time.time()
    aid = ns["resolve_by_path"]("001768", path)
    d = ns["img_mech"]("001768", aid, slap=False).reset_index(drop=True)
    ses = DATE_RE.search(path).group(1)
    d["subject"] = subj; d["asset_id"] = aid; d["session_path"] = path
    d["uid"] = [f"{subj}_{ses}_{i}" for i in range(len(d))]
    d.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    return out, f"{int(d.resp.sum())}/{len(d)} responsive", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(REPO, ".meso_parts"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--one", nargs=2, metavar=("SUBJECT", "PATH"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.one:                         # worker mode: one session, in its own process
        out, msg, dt = _one(a.one[0], a.one[1], a.out_dir)
        print(f"{a.one[0]} {DATE_RE.search(a.one[1]).group(1)}: {msg} ({dt:.0f}s)", flush=True)
        return
    sessions = _notebook_namespace()["PATHS"]["mesoscope"]
    if not a.merge:
        # Plain subprocesses (not multiprocessing), so it also runs where POSIX semaphores are restricted.
        import subprocess
        todo, running, failed = list(sessions), [], []
        while todo or running:
            while todo and len(running) < a.workers:
                s, p = todo.pop(0)
                running.append((s, p, subprocess.Popen(
                    [sys.executable, __file__, "--out-dir", a.out_dir, "--one", s, p])))
            time.sleep(2)
            for item in list(running):
                if item[2].poll() is not None:
                    running.remove(item)
                    if item[2].returncode != 0:
                        failed.append(item[:2])
        if failed:
            sys.exit(f"{len(failed)} session(s) failed: {failed}")
        return
    import pandas as pd
    parts = []
    for s, p in sessions:
        f = os.path.join(a.out_dir, f"meso_{s}_{DATE_RE.search(p).group(1)}.parquet")
        if not os.path.exists(f):
            sys.exit(f"missing {f}: run without --merge first")
        parts.append(pd.read_parquet(f))
    meso = pd.concat(parts, ignore_index=True)
    dst = os.path.join(REPO, "data", "crossscale_mechanism.parquet")
    old = pd.read_parquet(dst)
    keep = old[old.modality != "mesoscope"]
    new = pd.concat([keep[OUTCOLS], meso[OUTCOLS]], ignore_index=True)
    assert new.uid.is_unique, "duplicate uid"
    new.to_parquet(dst, index=False)
    print(f"wrote {dst}: {len(new)} rows",
          {m: int((new.modality == m).sum()) for m in new.modality.unique()})


if __name__ == "__main__":
    main()
