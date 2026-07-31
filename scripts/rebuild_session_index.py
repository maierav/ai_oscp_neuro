"""Rebuild ccf_session_index.csv across ALL ecephys assets in DANDI 001637.

The shipped index covers 30 sessions / 10 subjects; the dandiset now has 62 assets
across 16 subjects. CCF alignment is verified PER SESSION -- sub-832691 has one
session with location="unknown" for all channels and another with real CCF, so
CCF status must never be inferred from a sibling session of the same animal.

Emits every session with its paradigm and CCF status (including ccf=False) so
downstream code can see the full denominator, not just the usable subset.
"""
import collections, json, sys
from concurrent.futures import ThreadPoolExecutor
import requests, numpy as np, h5py, remfile

DS = "001637"
API = "https://api.dandiarchive.org/api/dandisets/{ds}/versions/draft/assets/"

# NWB interval block name -> the repo's paradigm label
PARADIGM = {
    "Standard mismatch block": "standard_oddball",
    "Sensory-motor mismatch block": "sensorimotor",
    "Sequence mismatch block": "sequence",
    "Duration mismatch block": "duration",
}

def assets():
    return requests.get(API.format(ds=DS), params={"page_size": 200}, timeout=60).json()["results"]

def url(aid):
    return requests.get(API.format(ds=DS) + aid + "/download/",
                        allow_redirects=False, timeout=60).headers["Location"]

def dec(a):
    return np.array([s.decode() if isinstance(s, bytes) else s for s in a])

def one(x):
    path = x["path"]
    fn = path.split("/")[-1]
    # sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys.nwb
    subject = fn.split("_")[0].replace("sub-", "")
    ses = fn.split("_ses-")[1].rsplit("_", 1)[0]
    date = ses.replace(f"ecephys-{subject}-", "")
    rec = {"subject": subject, "date": date, "aid": x["asset_id"],
           "gb": round(x["size"] / 1e9, 1)}
    try:
        f = h5py.File(remfile.File(url(x["asset_id"])), "r")
        el = f["/general/extracellular_ephys/electrodes"]
        loc = dec(el["location"][:])
        u = collections.Counter(loc)
        rec["ccf"] = not (len(u) == 1 and "unknown" in u)
        # ccf_xyz: coords present AND populated. Sessions without CCF registration
        # omit the x/y/z datasets entirely, so absence is itself the signal.
        if "x" in el:
            xs = el["x"][:]
            rec["ccf_xyz"] = bool(np.isfinite(xs).any() and np.nanstd(xs) > 0)
        else:
            rec["ccf_xyz"] = False
        rec["n_vis_chans"] = int(sum(v for k, v in u.items() if k.startswith("VIS")))
        rec["n_units"] = int(f["/units/id"].shape[0])
        blocks = [k.replace("_presentations", "") for k in f["/intervals"].keys()]
        rec["paradigm"] = next((v for k, v in PARADIGM.items() if k in blocks), "unknown")
        rec["blocks"] = blocks
        f.close()
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {e}"[:120]
    return rec

def write_index(rows, path):
    """Write the CCF-usable subset as the session index."""
    import csv
    cols = ["subject", "date", "paradigm", "n_units", "gb", "aid",
            "ccf", "ccf_xyz", "n_vis_chans"]
    rows = sorted((r for r in rows if r["ccf"] and r["ccf_xyz"]),
                  key=lambda r: (r["subject"], r["date"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    return len(rows)


if __name__ == "__main__":
    a = [x for x in assets() if x["path"].endswith("_ecephys.nwb")]
    print(f"sweeping {len(a)} ecephys assets", flush=True)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, a))
    out = sys.argv[1] if len(sys.argv) > 1 else "openscope_ccf/data/ccf_session_index.csv"
    json.dump(res, open("index_sweep.json", "w"), indent=1)
    ok = [r for r in res if "err" not in r]
    bad = [r for r in res if "err" in r]
    print(f"ok={len(ok)} errors={len(bad)}")
    for r in bad:
        print("  ERR", r["subject"], r["date"], r["err"])
    print("\nparadigm x ccf:")
    for (p, c), n in sorted(collections.Counter((r["paradigm"], r["ccf"]) for r in ok).items()):
        print(f"  {p:18s} ccf={c!s:5s} {n}")
    print(f"\nCCF sessions: {sum(r['ccf'] for r in ok)}/{len(ok)}  "
          f"subjects: {len(set(r['subject'] for r in ok))}")
    print("unknown-paradigm blocks:",
          {r['subject'] + '/' + r['date']: [b for b in r['blocks'] if 'mismatch' in b.lower()]
           for r in ok if r['paradigm'] == 'unknown'})
    n = write_index(ok, out)
    print(f"\nwrote {n} CCF-usable sessions -> {out}")
