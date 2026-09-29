"""Population time courses for the cross-scale feature-oddball figure.

For every session in notebooks/crossscale_oddball_index.ipynb (PATHS), runs that notebook's own
extractors with traces=True and stores, per session, the mean baseline-subtracted response to the
standard / 90-deg oddball / 90-deg control over the RESPONSIVE cells (the population the indices use).
Each worker first checks that its scalar columns reproduce data/crossscale_mechanism.parquet
row-for-row, so the time courses are guaranteed to come from the same cells as the committed DvI.

Usage (repo root):
    python scripts/extract_crossscale_timecourses.py [--workers 8] [--out-dir .tc_parts]
    python scripts/extract_crossscale_timecourses.py --merge   # -> data/crossscale_timecourses.parquet
"""
import argparse, os, re, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from extract_crossscale_mesoscope import _notebook_namespace  # noqa: E402

DS = {"ecephys": "001637", "mesoscope": "001768", "slap2": "001424"}
SCALARS = ["resp", "R_std", "R_o90", "R_c90", "R_c0", "OI", "DvI", "TPI"]
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _key(mod, path):
    return f"{mod}_{os.path.basename(path).replace('.nwb', '')}"


def _one(mod, subj, path, out_dir):
    import numpy as np, pandas as pd, matplotlib
    matplotlib.use("Agg")
    out = os.path.join(out_dir, _key(mod, path) + ".parquet")
    if os.path.exists(out):
        return "cached"
    ns = _notebook_namespace()
    aid = ns["resolve_by_path"](DS[mod], path)
    t0 = time.time()
    d = (ns["ece_mech"](aid, traces=True) if mod == "ecephys"
         else ns["img_mech"](DS[mod], aid, slap=(mod == "slap2"), traces=True)).reset_index(drop=True)
    ref = pd.read_parquet(os.path.join(REPO, "data", "crossscale_mechanism.parquet"))
    ref = ref[(ref.modality == mod) & (ref.session_path == path)].reset_index(drop=True)
    assert len(d) == len(ref), f"{path}: {len(d)} cells vs {len(ref)} committed"
    for c in SCALARS:
        a, b = d[c].astype(float).values, ref[c].astype(float).values
        assert np.allclose(a, b, equal_nan=True, atol=1e-9), f"{path}: column {c} differs from committed"
    grid = ns["TC_GRID"]
    rows = []
    for pop, m in [("responsive", d.resp.values)]:
        for cond in ["std", "o90", "c90"]:
            X = np.vstack(d.loc[m, f"tc_{cond}"].values) if m.sum() else np.full((1, len(grid)), np.nan)
            mu = np.nanmean(X, 0)
            rows += [dict(modality=mod, subject=subj, session_path=path, population=pop, condition=cond,
                          t=float(t), value=float(v), n_cells=int(m.sum())) for t, v in zip(grid, mu)]
    pd.DataFrame(rows).to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    return f"{int(d.resp.sum())}/{len(d)} responsive, scalars match committed ({time.time()-t0:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(REPO, ".tc_parts"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--one", nargs=3, metavar=("MOD", "SUBJECT", "PATH"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.one:
        print(f"{a.one[0]} {a.one[1]} {DATE_RE.search(a.one[2]).group(1)}: {_one(*a.one, a.out_dir)}", flush=True)
        return
    paths = _notebook_namespace()["PATHS"]
    jobs = [(m, s, p) for m in ["ecephys", "slap2", "mesoscope"] for s, p in paths[m]]
    if not a.merge:
        import subprocess
        todo, running, failed = list(jobs), [], []
        while todo or running:
            while todo and len(running) < a.workers:
                j = todo.pop(0)
                running.append((j, subprocess.Popen([sys.executable, __file__, "--out-dir", a.out_dir, "--one", *j])))
            time.sleep(2)
            for item in list(running):
                if item[1].poll() is not None:
                    running.remove(item)
                    if item[1].returncode != 0:
                        failed.append(item[0])
        if failed:
            sys.exit(f"{len(failed)} session(s) failed: {failed}")
        return
    import pandas as pd
    parts = []
    for m, s, p in jobs:
        f = os.path.join(a.out_dir, _key(m, p) + ".parquet")
        if not os.path.exists(f):
            sys.exit(f"missing {f}: run without --merge first")
        parts.append(pd.read_parquet(f))
    out = pd.concat(parts, ignore_index=True)
    dst = os.path.join(REPO, "data", "crossscale_timecourses.parquet")
    out.to_parquet(dst, index=False)
    print(f"wrote {dst}: {out.session_path.nunique()} sessions, {len(out)} rows")


if __name__ == "__main__":
    main()
