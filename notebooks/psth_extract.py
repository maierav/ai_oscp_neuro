# Unified per-result PSTH extractor (streaming + MUA on best channel + trial PSTHs).
# Used to build the four-level diagnostic figures; one small get_onsets adapter per result.
import numpy as np, h5py, remfile, requests, time

def s3(ds, aid):
    return requests.get(f"https://api.dandiarchive.org/api/dandisets/{ds}/versions/draft/assets/{aid}/download/",
                        allow_redirects=False, timeout=60).headers["Location"]
def col(g, c):
    v = g[c][:]; return np.array([x.decode() if isinstance(x, bytes) else x for x in v])
def resolve(subj, date, ds="001637", tmpl="sub-{s}/sub-{s}_ses-ecephys-{s}-{d}_ecephys.nwb"):
    p = tmpl.format(s=subj, d=date)
    r = requests.get(f"https://api.dandiarchive.org/api/dandisets/{ds}/versions/draft/assets/",
                     params={"path": p}, timeout=30)
    return [a for a in r.json()["results"] if a["path"] == p][0]["asset_id"]

def _unit_labels(fh, U, qc):
    """Per-unit (area, layer) via the CCF-corrected extremum-channel mapping. Returns arrays
    aligned to the QC-unit order used in extract_psth (qc==True units, in file order)."""
    import re
    try:
        el = fh["general/extracellular_ephys/electrodes"]
        loc = np.array([x.decode() if isinstance(x, bytes) else x for x in el["location"][:]])
        egrp = np.array([x.decode() if isinstance(x, bytes) else x for x in el["group_name"][:]])
    except Exception:
        return None, None
    # per-group electrode offsets (stacked electrode table) + blocklen
    offs = {}; blk = {}
    for g in np.unique(egrp):
        idxs = np.where(egrp == g)[0]; offs[g] = idxs.min(); blk[g] = len(idxs)
    dev = np.array([x.decode() if isinstance(x, bytes) else x for x in U["device_name"][:]])
    eci = U["extremum_channel_index"][:]
    def decode(s):
        s = str(s); a = re.match(r"([A-Za-z]+)", s); lay = re.search(r"(\d[ab]?|2/3)$", s)
        return (a.group(1) if a else s), (lay.group(1) if lay else None)
    areas = []; layers = []
    n = len(U["spike_times_index"][:])
    for i in range(n):
        if not qc[i]: continue
        d = dev[i]
        row = offs.get(d, 0) + min(int(eci[i]), blk.get(d, 1) - 1)
        a, l = decode(loc[row]) if row < len(loc) else (None, None)
        areas.append(a); layers.append(l)
    return np.array(areas, dtype=object), np.array(layers, dtype=object)

def extract_psth(sessions, get_onsets, example_subject, ds="001637",
                 PRE=0.1, POST=0.5, BW=0.01, resp=(0.045, 0.295), base=(-0.10, -0.005),
                 tmpl="sub-{s}/sub-{s}_ses-ecephys-{s}-{d}_ecephys.nwb", verbose=True,
                 keep_labels=False):
    """
    sessions       : list of (subject, date).
    get_onsets(fh) : returns (dev_onsets, ctl_onsets) 1-D arrays of event start-times (s).
    Returns dict {sess:[{subject,dev,ctl}], example:{...}, cen, resp}.
      example carries the example unit's trial PSTHs AND the example channel's MUA trial PSTHs.
    """
    EDGES = np.arange(-PRE, POST + BW, BW); CEN = EDGES[:-1] + BW / 2
    def upsth(sp, times):
        if len(times) == 0: return np.zeros(len(CEN))
        lo = np.searchsorted(sp, times.min() - PRE); hi = np.searchsorted(sp, times.max() + POST)
        sp2 = sp[lo:hi]
        if len(sp2) == 0: return np.zeros(len(CEN))
        rel = (sp2[None, :] - times[:, None]).ravel(); rel = rel[(rel >= -PRE) & (rel < POST)]
        return np.histogram(rel, EDGES)[0] / (len(times) * BW)
    def trials(sp, times):
        M = np.zeros((len(times), len(CEN)))
        for i, t0 in enumerate(times):
            rel = sp[(sp >= t0 - PRE) & (sp < t0 + POST)] - t0
            M[i] = np.histogram(rel, EDGES)[0] / BW
        return M
    def rate(sp, times, w):
        lo = np.searchsorted(sp, times + w[0]); hi = np.searchsorted(sp, times + w[1])
        return (hi - lo) / (w[1] - w[0])
    sess_data = []; example = None
    for subj, date in sessions:
        t0 = time.time()
        fh = h5py.File(remfile.File(s3(ds, resolve(subj, date, ds, tmpl))), "r")
        dev, ctl = get_onsets(fh)
        U = fh["units"]; st = U["spike_times"][:]; sti = U["spike_times_index"][:]
        n = len(sti); starts = np.concatenate([[0], sti[:-1]])
        qc = U["default_qc"][:].astype(bool) if "default_qc" in U else np.ones(n, bool)
        dev_a = np.asarray(dev, float); ctl_a = np.asarray(ctl, float)
        devP = []; ctlP = []; rs = []; qidx = []
        for i in range(n):
            if not qc[i]: continue
            sp = st[starts[i]:sti[i]]
            devP.append(upsth(sp, dev_a)); ctlP.append(upsth(sp, ctl_a))
            rs.append(np.nanmean(rate(sp, dev_a, resp) - rate(sp, dev_a, base)) if len(dev_a) else np.nan)
            qidx.append(i)
        devP = np.array(devP); ctlP = np.array(ctlP); rs = np.array(rs); qidx = np.array(qidx)
        sd = dict(subject=subj, dev=devP, ctl=ctlP)
        if keep_labels:
            ar, la = _unit_labels(fh, U, qc)
            sd["area"] = ar; sd["layer"] = la
        sess_data.append(sd)
        if subj == example_subject and example is None and len(rs):
            best_local = int(np.nanargmax(rs)); best = int(qidx[best_local])
            spb = st[starts[best]:sti[best]]
            # MUA: units on the same channel (device_name, extremum_channel_index) as the best unit
            try:
                dev_name = col(U, "device_name"); eci = U["extremum_channel_index"][:]
                same = np.where((dev_name == dev_name[best]) & (eci == eci[best]) & qc)[0]
                if len(same) < 2:  # widen to a +/-2 channel neighbourhood for a true multiunit
                    same = np.where((dev_name == dev_name[best]) & (np.abs(eci - eci[best]) <= 2) & qc)[0]
                mua_sp = np.sort(np.concatenate([st[starts[j]:sti[j]] for j in same]))
                ch = int(eci[best]); n_mua = len(same)
                mua_dev = trials(mua_sp, dev_a); mua_ctl = trials(mua_sp, ctl_a)
            except Exception:
                ch = None; n_mua = 0; mua_dev = mua_ctl = None
            example = dict(subject=subj, uid=best, mua_ch=ch, mua_nunits=n_mua,
                           dev_trials=trials(spb, dev_a), ctl_trials=trials(spb, ctl_a),
                           mua_dev=mua_dev, mua_ctl=mua_ctl)
        if verbose:
            print(f"  {subj}: {devP.shape[0]} QC units, dev={len(dev_a)} ctl={len(ctl_a)} ({time.time()-t0:.0f}s)")
        fh.close()
    return dict(sess=sess_data, example=example, cen=CEN, resp=resp)
