"""Full-cohort traces for notebooks/crosstechnique_corrections.ipynb.

Runs that notebook's own per-session extractors (`ece_full`, `img_full`: per-cell PSTHs for the standard,
90-deg oddball and omission, plus the standard split by position in the standard train) over every
feature-oddball session listed in notebooks/crossscale_oddball_index.ipynb (PATHS), one worker process
per session. The merge step reduces the per-cell traces to MOUSE-level summaries (per-cell traces for
~23k cells are not committed):

  kind="joint_balanced"  mean trace of a response-strength x tuning balanced draw, per mouse
                         (same bins/draw size as notebooks/responsiveness_matching.ipynb), 200 draws
  kind="adaptation"      mean baseline-subtracted trace per mouse for standard early/mid/late
                         (train positions 0-2 / 3-8 / >=9) and omission, responsive cells

Usage (repo root):
    python scripts/extract_crosstechnique_traces.py [--workers 8] [--out-dir .ct_parts]
    python scripts/extract_crosstechnique_traces.py --merge   # -> data/crosstechnique_traces.parquet
"""
import argparse, os, re, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from extract_crossscale_mesoscope import _notebook_namespace  # noqa: E402  (PATHS + resolve_by_path)

DS = {"ecephys": "001637", "mesoscope": "001768", "slap2": "001424"}
CONDS = ["std", "o90", "om", "e", "m", "l"]


def _key(mod, path):
    return f"{mod}_{os.path.basename(path).replace('.nwb', '')}"


def _extractors():
    import nbformat as nbf
    nb = nbf.read(os.path.join(REPO, "notebooks", "crosstechnique_corrections.ipynb"), 4)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    strip = lambda s: "\n".join(l for l in s.split("\n") if not l.strip().startswith(("!", "%")))
    ns = {}
    for src in code[1:3]:            # streaming helpers + ece_full, img_full
        exec(strip(src), ns)
    return ns


def _one(mod, subj, path, out_dir):
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    out = os.path.join(out_dir, _key(mod, path) + ".npz")
    if os.path.exists(out):
        return "cached"
    aid = _notebook_namespace()["resolve_by_path"](DS[mod], path)
    ns = _extractors(); t0 = time.time()
    df, P, cen = (ns["ece_full"](aid) if mod == "ecephys"
                  else ns["img_full"](DS[mod], aid, slap=(mod == "slap2")))
    arrs = {f"P_{k}": np.asarray(v, np.float32) for k, v in P.items() if len(v)}
    cols = {f"c_{c}": df[c].values for c in ["resp", "base_mean", "R_std", "R_o90", "OI", "TPI"]}
    np.savez_compressed(out + ".tmp.npz", cen=cen, subject=subj, path=path, **cols, **arrs)
    os.replace(out + ".tmp.npz", out)
    return f"{int(df.resp.sum())}/{len(df)} responsive ({time.time()-t0:.0f}s)"


def _balanced_mean(T, tpi, rstd, rng, reps=200, ntpi=6, nmag=3):
    import numpy as np
    tb = np.clip(np.digitize(tpi, np.linspace(-1, 1, ntpi + 1)) - 1, 0, ntpi - 1)
    r = np.argsort(np.argsort(np.abs(rstd))) / max(len(rstd) - 1, 1)
    mb = np.clip(np.digitize(r, np.linspace(0, 1, nmag + 1)) - 1, 0, nmag - 1)
    c = tb * nmag + mb; occ, cnt = np.unique(c, return_counts=True); k = max(int(np.median(cnt)), 3)
    idx = [np.where(c == o)[0] for o in occ]
    acc = [np.nanmean(T[np.concatenate([rng.choice(ix, k, replace=len(ix) < k) for ix in idx])], 0)
           for _ in range(reps)]
    return np.nanmean(acc, 0)


def merge(out_dir):
    import numpy as np, pandas as pd
    paths = _notebook_namespace()["PATHS"]
    rows, rng = [], np.random.default_rng(0)
    for mod in ["ecephys", "mesoscope", "slap2"]:
        per_mouse = {}
        for subj, p in paths[mod]:
            f = os.path.join(out_dir, _key(mod, p) + ".npz")
            if not os.path.exists(f):
                sys.exit(f"missing {f}: run without --merge first")
            per_mouse.setdefault(subj, []).append(dict(np.load(f, allow_pickle=True)))
        for subj, parts in per_mouse.items():
            cen = parts[0]["cen"] * 1000
            resp = np.concatenate([d["c_resp"].astype(bool) for d in parts])
            bm = np.concatenate([d["c_base_mean"] for d in parts])[resp]
            tpi = np.concatenate([d["c_TPI"] for d in parts])[resp]
            rstd = np.concatenate([d["c_R_std"] for d in parts])[resp]
            get = lambda k: np.concatenate([d[f"P_{k}"] for d in parts])[resp] - bm[:, None]
            ok = np.isfinite(tpi)
            for cond in ["std", "o90"]:
                v = _balanced_mean(get(cond)[ok], tpi[ok], rstd[ok], rng)
                rows += [dict(modality=mod, subject=subj, kind="joint_balanced", condition=cond, t=float(t),
                              value=float(x), n_cells=int(ok.sum()), n_sessions=len(parts)) for t, x in zip(cen, v)]
            for cond in ["e", "m", "l", "om"]:
                if f"P_{cond}" not in parts[0]:
                    continue
                v = np.nanmean(get(cond), 0)
                rows += [dict(modality=mod, subject=subj, kind="adaptation", condition=cond, t=float(t),
                              value=float(x), n_cells=int(resp.sum()), n_sessions=len(parts)) for t, x in zip(cen, v)]
    out = pd.DataFrame(rows)
    dst = os.path.join(REPO, "data", "crosstechnique_traces.parquet")
    out.to_parquet(dst, index=False)
    print(f"wrote {dst}:", out.groupby(["modality", "kind"]).subject.nunique().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(REPO, ".ct_parts"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--one", nargs=3, metavar=("MOD", "SUBJECT", "PATH"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.one:
        print(f"{a.one[0]} {a.one[1]} {os.path.basename(a.one[2])[:60]}: {_one(*a.one, a.out_dir)}", flush=True)
        return
    if a.merge:
        return merge(a.out_dir)
    import subprocess
    paths = _notebook_namespace()["PATHS"]
    jobs = [(m, s, p) for m in ["ecephys", "slap2", "mesoscope"] for s, p in paths[m]]
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


if __name__ == "__main__":
    main()
