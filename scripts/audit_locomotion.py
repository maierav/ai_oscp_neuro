"""Locomotion audit for the sensorimotor closed-loop/open-loop contrast.

Result 3 in ai_oscp_neuro rests on ONE session (sub-830794) because the designed
closed-loop vs open-loop contrast needs the animal RUNNING during the 8 events per
type in `Control block 4` (open_loop_prerecorded). This scores every SENSORYMOTOR
session on that specific criterion.
"""
import collections, json, sys
from concurrent.futures import ThreadPoolExecutor
import requests, numpy as np, h5py, remfile

API = "https://api.dandiarchive.org/api/dandisets/{ds}/versions/draft/assets/"
RUN_THRESH = 1.0  # cm/s, matches repo's ">1 cm/s" running definition

def assets(ds):
    return requests.get(API.format(ds=ds), params={"page_size": 200}, timeout=60).json()["results"]

def url(ds, aid):
    return requests.get(API.format(ds=ds) + aid + "/download/",
                        allow_redirects=False, timeout=60).headers["Location"]

def dec(a):
    return np.array([s.decode() if isinstance(s, bytes) else s for s in a])

def col(g, name):
    """Read an intervals column, decoding ecephys byte-strings to float."""
    v = g[name][:]
    if v.dtype.kind == "S":
        return dec(v).astype(float)
    return v.astype(float)

def one(x):
    out = {"path": x["path"]}
    try:
        f = h5py.File(remfile.File(url("001637", x["asset_id"])), "r")
        iv = list(f["/intervals"].keys())
        if "Sensory-motor mismatch block_presentations" not in iv:
            f.close(); out["skip"] = "not sensorimotor"; return out

        # running speed on the session clock
        rs = f["/processing/running/running_speed"]
        speed = rs["data"][:].astype(float)
        rt = rs["timestamps"][:].astype(float)
        out["frac_run_session"] = float((np.abs(speed) > RUN_THRESH).mean())
        out["median_speed"] = float(np.median(speed))

        def block_events(key):
            g = f["/intervals"][key]
            st = col(g, "start_time")
            tt = dec(g["TrialType"][:]) if "TrialType" in g else np.array(["?"] * len(st))
            return st, tt

        res = {}
        for label, key in (("closed", "Sensory-motor mismatch block_presentations"),
                           ("open", "Control block 4_presentations")):
            if key not in iv:
                res[label] = None; continue
            st, tt = block_events(key)
            per = {}
            for t in sorted(set(tt)):
                if t == "standard" or t == "prerecorded":
                    continue
                ev = st[tt == t]
                # speed in the 1 s before each event = the motor state it violated
                run = []
                for e in ev:
                    m = (rt >= e - 1.0) & (rt < e)
                    run.append(np.abs(speed[m]).mean() if m.any() else np.nan)
                run = np.array(run, float)
                per[t] = {"n": int(len(ev)),
                          "n_running": int(np.nansum(run > RUN_THRESH)),
                          "n_rest": int(np.nansum(run <= RUN_THRESH))}
            res[label] = per
        out["blocks"] = res
        # the gating number: running open-loop events for motor-contingent types
        op = res.get("open") or {}
        out["open_running_max"] = max((v["n_running"] for v in op.values()), default=0)
        out["open_types"] = {k: (v["n"], v["n_running"]) for k, v in op.items()}
        f.close()
    except Exception as e:
        out["err"] = f"{type(e).__name__}: {e}"[:120]
    return out

if __name__ == "__main__":
    a = [x for x in assets("001637") if x["path"].endswith("_ecephys.nwb")]
    only = sys.argv[1:]
    if only:
        a = [x for x in a if any(o in x["path"] for o in only)]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, a))
    json.dump(res, open("loco_audit.json", "w"), indent=1)
    for r in res:
        if "err" in r: print("ERR ", r["path"].split("/")[-1], r["err"]); continue
        if "skip" in r: continue
        print(f"{r['path'].split('/')[-1][:52]} run={r['frac_run_session']:.2f} "
              f"open_running_max={r['open_running_max']} {r['open_types']}")
