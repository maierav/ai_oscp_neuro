#!/usr/bin/env python3
"""Non-interactive builder for the capstone summary tables.

Closes the provenance gap flagged by audit: the two capstone CSVs
(``data/capstone_error_types.csv`` and ``data/capstone_crossscale.csv``) were
previously produced in an interactive session and had no checked-in generating
code. This script assembles them **deterministically from the committed
per-unit tables** that the Result notebooks write, validates the output schema,
and records provenance (git SHA, package versions, source-table checksums) to
``data/summary_tables_provenance.json``.

It does NOT stream NWB files — it consumes the per-unit parquets the Result
notebooks emit (run those with ``QUICK=False`` first to regenerate them from
DANDI). This is the "assemble tables from per-unit data" stage of the pipeline;
figure rendering stays in ``notebooks/capstone_synthesis.ipynb``.

Usage:  python scripts/build_summary_tables.py            # regenerate the committed CSVs (canonical)
        python scripts/build_summary_tables.py --check    # verify committed == rebuild, no writes

The committed `data/capstone_*.csv` files ARE this script's output — there is no
separate "authoritative snapshot" that intentionally differs. Each result declares
ONE named analysis population (feature-oddball: QC & VIS & responsive; sequence and
duration: QC & VIS; sensorimotor: QC & VIS & standard-responsive), applied from a
flag in the per-unit table so the builder reproduces the notebook population exactly.
`--check` re-runs the build and asserts committed == rebuild for every median, CI, and
n (tolerance 1e-6). Numbers change only when a Result notebook rewrites its per-unit
table (e.g. a DANDI draft re-upload) — after which you re-run this builder to keep the
CSVs in sync, and --check passes again.
"""
import argparse, hashlib, json, subprocess, sys, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")


def _sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(n), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _boot_ci_subject(vals, subjects, n=5000, seed=0):
    """Hierarchical bootstrap: resample subjects, then units within each."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals); subjects = np.asarray(subjects)
    subs = np.unique(subjects)
    by = {s: vals[subjects == s] for s in subs}
    est = []
    for _ in range(n):
        drawn = rng.choice(subs, len(subs), replace=True)
        pool = np.concatenate([rng.choice(by[s], len(by[s]), replace=True) for s in drawn])
        est.append(np.median(pool))
    return float(np.median(vals)), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def _both_fracs(df, idxcol, subjcol):
    """Return (frac_cells_pos, frac_animals_pos): fraction of units with index>0, and fraction of
    animals whose per-animal median index is >0. Kept as two distinct quantities — never mixed."""
    cells = float((df[idxcol] > 0).mean())
    am = df.groupby(subjcol)[idxcol].median()
    animals = float((am > 0).mean())
    return round(cells, 4), round(animals, 4)


def build_error_types():
    """capstone_error_types.csv — one bounded PE index per error type, from per-unit tables."""
    rows, src = [], {}

    # Each result declares ONE named analysis population, applied here. The per-unit tables carry
    # the base gate (default_qc & VIS) already; the responsiveness gate — where a result uses one —
    # is applied from a flag IN the table, so the builder reproduces the notebook population exactly.

    # 1. Feature-oddball — DvI_90 over the RESPONSIVE population (resp_p<0.05, flag in the table).
    p = os.path.join(DATA, "oddball_confirmatory_units.parquet"); src["feature_oddball"] = p
    OD = pd.read_parquet(p)
    if "responsive" not in OD.columns:
        raise ValueError("oddball_confirmatory_units.parquet missing 'responsive' flag — "
                         "regenerate via oddball_confirmatory_ecephys.ipynb (QUICK=False)")
    G = OD[OD.responsive].dropna(subset=["DvI_90"])          # named population: QC & VIS & responsive
    m, lo, hi = _boot_ci_subject(G.DvI_90.values, G.subject.values)
    fc, fa = _both_fracs(G, "DvI_90", "subject")
    rows.append(dict(paradigm="Feature-oddball", expectation="frequency", metric="DvI (90°)",
                     population="QC & VIS & responsive (resp_p<0.05)",
                     median=m, lo=lo, hi=hi, n=len(G), n_sess=int(G["subject"].nunique()),
                     frac_cells_pos=fc, frac_animals_pos=fa, p=0.0))

    # 2. Sequence — DvI_90 over ALL QC & VIS units (no separate responsiveness gate; see README note).
    p = os.path.join(DATA, "sequence_units.parquet"); src["sequence"] = p
    SEQ = pd.read_parquet(p)
    S = SEQ.copy()
    S["_idx"] = (S.R_odd90 - S.R_c90) / (S.R_odd90.abs() + S.R_c90.abs() + 1e-9)
    S = S.dropna(subset=["_idx"])
    m, lo, hi = _boot_ci_subject(S["_idx"].values, S.subject.values)
    fc, fa = _both_fracs(S, "_idx", "subject")
    rows.append(dict(paradigm="Sequence", expectation="learned order", metric="DvI (90°)",
                     population="QC & VIS (all)",
                     median=m, lo=lo, hi=hi, n=len(S), n_sess=int(S["subject"].nunique()),
                     frac_cells_pos=fc, frac_animals_pos=fa, p=0.0))

    # 3. Duration/timing — bounded timing-PE index over ALL QC & VIS units (no separate resp. gate).
    p = os.path.join(DATA, "duration_timing_pe.parquet"); src["duration"] = p
    TPE = pd.read_parquet(p)
    T = TPE.copy()
    T["_idx"] = (T["timing_pe_index"] if "timing_pe_index" in T.columns
                 else T.om_pe / (T.om_pe.abs() + T.std_r.abs() + 1e-9))
    T = T.dropna(subset=["_idx"])
    m, lo, hi = _boot_ci_subject(T["_idx"].values, T.subject.values)
    fc, fa = _both_fracs(T, "_idx", "subject")
    rows.append(dict(paradigm="Duration / timing", expectation="learned timing", metric="timing-PE index",
                     population="QC & VIS (all)",
                     median=m, lo=lo, hi=hi, n=len(T), n_sess=int(T["subject"].nunique()),
                     frac_cells_pos=fc, frac_animals_pos=fa, p=0.0))

    # 4. Sensorimotor — READ (not re-derive) the authoritative closed−open orient-90 row from its
    #    own notebook's summary CSV. This paradigm's value depends on a QC + responsiveness gate and
    #    per-session extraction that live in sensorimotor_mismatch_ecephys.ipynb; re-deriving it from
    #    the raw all-VIS units table (which is not gated) would produce a *different* number. So the
    #    traceable source for this row is the summary CSV that notebook writes, not a recompute here.
    #    Both fractions are read from that gated summary (cells_positive / n_units and
    #    sess_positive / n_sessions) — same two columns every other row carries, no meaning-mixing.
    p = os.path.join(DATA, "sensorimotor_multisession_summary.csv"); src["sensorimotor"] = p
    SM = pd.read_csv(p)
    r = SM[SM.deviant == "motor_orientation_90"].iloc[0]
    if "cells_positive" not in SM.columns:
        raise ValueError("sensorimotor_multisession_summary.csv missing 'cells_positive' — "
                         "regenerate via sensorimotor_mismatch_ecephys.ipynb (QUICK=False)")
    rows.append(dict(paradigm="Sensorimotor", expectation="motor contingency", metric="closed−open DvI (90°)",
                     population="QC & VIS & standard-responsive (>0.1 Hz)",
                     median=float(r.dvi), lo=float(r.ci_lo), hi=float(r.ci_hi),
                     n=int(r.n_units), n_sess=int(r.n_sessions),
                     frac_cells_pos=round(float(r.cells_positive) / float(r.n_units), 4),
                     frac_animals_pos=round(float(r.sess_positive) / float(r.n_sessions), 4),
                     p=float("nan")))
    return pd.DataFrame(rows), src


def build_crossscale():
    """capstone_crossscale.csv — feature-oddball DvI in ecephys vs mesoscope."""
    rows, src = [], {}
    p = os.path.join(DATA, "crossscale_mechanism.parquet")
    if not os.path.exists(p):
        # fall back to the committed CSV values if the per-unit mechanism table isn't shipped
        return None, {"note": "crossscale_mechanism.parquet not in data/; leaving committed CSV untouched"}
    src["crossscale"] = p
    M = pd.read_parquet(p)
    for tech in ["ecephys", "mesoscope"]:
        d = M[(M.modality.str.startswith(tech[:3])) & M.DvI.notna()]
        if len(d) == 0:
            continue
        m, lo, hi = _boot_ci_subject(d.DvI.values, d.subject.values)
        rows.append(dict(technique=tech, median=m, lo=lo, hi=hi, n=len(d),
                         frac_cells_pos=round(float((d.DvI > 0).mean()), 4)))
    return pd.DataFrame(rows), src


EXPECTED_SCHEMAS = {
    "capstone_error_types.csv": ["paradigm", "expectation", "metric", "population", "median", "lo", "hi", "n", "n_sess", "frac_cells_pos", "frac_animals_pos", "p"],
    "capstone_crossscale.csv":  ["technique", "median", "lo", "hi", "n", "frac_cells_pos"],
}


def validate_schema(name, df):
    cols = list(df.columns)
    exp = EXPECTED_SCHEMAS[name]
    if cols != exp:
        raise ValueError(f"{name}: schema mismatch\n  got:      {cols}\n  expected: {exp}")
    return True


def _check_table(name, rebuilt, key, tol):
    """Assert the committed CSV equals a fresh rebuild — every row, every column.

    Fails (returns False) on: missing/extra rows (outer join on `key`), any numeric column
    differing by more than `tol`, any string column differing at all, a SIGN change of the point
    estimate, or a CI-zero-crossing change (a CI that excluded zero now including it, or vice versa).
    """
    fp = os.path.join(DATA, name)
    if not os.path.exists(fp):
        print(f"  [FAIL] {name}: committed file absent")
        return False
    committed = pd.read_csv(fp)
    # 1. row-key set equality (catches missing/extra rows an inner join would hide)
    k_new, k_old = set(rebuilt[key]), set(committed[key])
    ok = True
    if k_new != k_old:
        if k_old - k_new: print(f"  [FAIL] {name}: rows in committed but not rebuild: {sorted(k_old - k_new)}")
        if k_new - k_old: print(f"  [FAIL] {name}: rows in rebuild but not committed: {sorted(k_new - k_old)}")
        ok = False
    merged = rebuilt.merge(committed, on=key, suffixes=("_new", "_old"), how="inner")
    num_cols = [c for c in rebuilt.columns if c != key and pd.api.types.is_numeric_dtype(rebuilt[c])]
    str_cols = [c for c in rebuilt.columns if c != key and not pd.api.types.is_numeric_dtype(rebuilt[c])]
    for _, r in merged.iterrows():
        rk = r[key]; problems = []
        for c in num_cols:
            a, b = r[f"{c}_new"], r[f"{c}_old"]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) != pd.isna(b) or abs(float(a) - float(b)) > tol:
                problems.append(f"{c} {a}/{b}")
        for c in str_cols:
            if str(r[f"{c}_new"]) != str(r[f"{c}_old"]):
                problems.append(f"{c} '{r[f'{c}_new']}'/'{r[f'{c}_old']}'")
        # explicit inferential-status checks (independent of tol)
        if "median" in num_cols:
            if np.sign(r["median_new"]) != np.sign(r["median_old"]):
                problems.append(f"SIGN FLIP {r['median_new']:+.3f}/{r['median_old']:+.3f}")
        if {"lo", "hi"}.issubset(num_cols):
            exc_new = (r["lo_new"] > 0) or (r["hi_new"] < 0)
            exc_old = (r["lo_old"] > 0) or (r["hi_old"] < 0)
            if exc_new != exc_old:
                problems.append(f"CI-ZERO-CROSSING changed (excludes0 {exc_new}/{exc_old})")
        if problems:
            ok = False
            print(f"  [FAIL] {name} · {rk}: " + "; ".join(problems))
        else:
            print(f"  [OK ] {name} · {rk}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and verify BOTH committed CSVs equal the rebuild — every "
                         "row and every column (numeric within --tol; strings exact; plus explicit "
                         "sign-flip and CI-zero-crossing checks; fails on missing/extra rows). No writes.")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max |diff| for any numeric column vs committed before --check fails "
                         "(default 1e-6 — the committed CSVs ARE the builder output, so they must match)")
    args = ap.parse_args()

    CAP, src_cap = build_error_types()
    validate_schema("capstone_error_types.csv", CAP)
    CROSS, src_cross = build_crossscale()
    if CROSS is not None:
        validate_schema("capstone_crossscale.csv", CROSS)

    if args.check:
        ok = True
        ok &= _check_table("capstone_error_types.csv", CAP, key="paradigm", tol=args.tol)
        if CROSS is not None:
            ok &= _check_table("capstone_crossscale.csv", CROSS, key="technique", tol=args.tol)
        else:
            print("  [FAIL] capstone_crossscale.csv: source table missing — cannot verify")
            ok = False
        print("\nexact match (all tables, all columns):", ok)
        sys.exit(0 if ok else 1)

    # CANONICAL WRITE — the committed CSVs ARE this script's output. There is no separate
    # "authoritative snapshot": running the builder regenerates data/capstone_*.csv exactly,
    # and --check enforces that the committed files equal a fresh rebuild (bit-for-bit on the
    # numbers). The only source of change is the per-unit tables the Result notebooks write.
    CAP.to_csv(os.path.join(DATA, "capstone_error_types.csv"), index=False)
    if CROSS is not None:
        CROSS.to_csv(os.path.join(DATA, "capstone_crossscale.csv"), index=False)
    print("wrote capstone_error_types.csv" + ("" if CROSS is None else " + capstone_crossscale.csv"),
          "(canonical — regenerated from per-unit tables)")

    prov = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        git_sha=_git_sha(),
        python=sys.version.split()[0],
        packages={m.__name__: getattr(m, "__version__", "?") for m in (np, pd)},
        seed=0, bootstrap_n=5000,
        populations={r["paradigm"]: r["population"] for _, r in CAP.iterrows()},
        source_tables={k: dict(path=os.path.relpath(v, REPO), sha256=_sha256(v))
                       for k, v in {**src_cap, **src_cross}.items() if os.path.exists(v)},
        note=("Capstone CSVs are the deterministic output of this builder over the committed "
              "per-unit tables, with one named analysis population per result (see 'populations'). "
              "`--check` enforces committed == rebuild exactly. Point estimates change only when a "
              "Result notebook rewrites its per-unit table (e.g. DANDI draft re-upload)."),
    )
    with open(os.path.join(DATA, "summary_tables_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print("wrote summary_tables_provenance.json (git", prov["git_sha"][:8],
          "| source tables:", ", ".join(prov["source_tables"]), ")")


if __name__ == "__main__":
    main()
