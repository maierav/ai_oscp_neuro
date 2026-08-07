# Imaging (dF/F) four-level PSTH extractor — mesoscope (001768) and SLAP2 (001424).
# Panel analogues: example PLANE/DMD mean-dF/F (A), example ROI (B), example session over ROIs (C),
# grand average across sessions (D). Traces are dF/F, resampled onto a common time grid by
# interpolation around each event onset (imaging clocks are ~7 Hz mesoscope, ~200 Hz SLAP2).
import numpy as np, h5py, remfile, requests, re, time

def s3(ds, aid):
    return requests.get(f"https://api.dandiarchive.org/api/dandisets/{ds}/versions/draft/assets/{aid}/download/",
                        allow_redirects=False, timeout=60).headers["Location"]
def dec(a): return np.array([x.decode() if isinstance(x, bytes) else x for x in a])

def _event_mat(sig, t_sig, onsets, grid, off=0.0):
    """Interpolate a 1-D dF/F signal onto (n_events x len(grid)) around each onset (+ optional off)."""
    M = np.full((len(onsets), len(grid)), np.nan)
    for i, o in enumerate(onsets):
        M[i] = np.interp(o + off + grid, t_sig, sig, left=np.nan, right=np.nan)
    return M

def extract_meso(sessions, get_onsets, example_subject, PRE=0.3, POST=0.7, DT=0.02,
                 resp=(0.03, 0.28), verbose=True):
    """
    Mesoscope sequence PSTH. sessions: list of (subject, date, aid). get_onsets(fh)->(dev,ctl).
    Returns {sess:[{subject,dev,ctl}], example:{plane mean + example ROI}, cen, resp}.
    dev/ctl are (n_soma_roi x T) trial-mean dF/F matrices per session.
    """
    grid = np.arange(-PRE, POST, DT)
    sess_data = []; example = None
    for subj, date, aid in sessions:
        t0 = time.time()
        fh = h5py.File(remfile.File(s3("001768", aid)), "r")
        dev, ctl = get_onsets(fh)
        dev = np.asarray(dev, float); ctl = np.asarray(ctl, float)
        roi_dev = []; roi_ctl = []; ex_plane = None
        planes = [k for k in fh["processing"].keys() if k.startswith("VIS")]
        best_plane = None; best_n = -1
        for pl in planes:
            pr = fh["processing"][pl]["dff_timeseries"]["dff_timeseries"]
            D = pr["data"]; ts = pr["timestamps"][:] if "timestamps" in pr else \
                np.arange(D.shape[0]) / 7.0
            rt = fh["processing"][pl]["dff_timeseries"]["roi_table"] if "roi_table" in fh["processing"][pl]["dff_timeseries"] else None
            Dv = D[:]  # (T_img x n_roi)
            issoma = None
            try:
                rtab = fh["processing"][pl]["dff_timeseries"]["roi_table"]
                issoma = rtab["is_soma"][:].astype(bool)
            except Exception:
                issoma = np.ones(Dv.shape[1], bool)
            for r in range(Dv.shape[1]):
                if not issoma[r]: continue
                sig = Dv[:, r]
                md = np.nanmean(_event_mat(sig, ts, dev, grid), 0)
                mc = np.nanmean(_event_mat(sig, ts, ctl, grid), 0)
                roi_dev.append(md); roi_ctl.append(mc)
            if subj == example_subject and issoma.sum() > best_n:
                best_n = int(issoma.sum()); best_plane = pl
                # plane mean-dF/F over somatic ROIs, and the single most-responsive soma
                somacols = np.where(issoma)[0]
                planemean = np.nanmean(Dv[:, somacols], 1)
                pm_dev = _event_mat(planemean, ts, dev, grid)  # (n_events x T)
                pm_ctl = _event_mat(planemean, ts, ctl, grid)
                # best ROI by dev-window response
                wm = (grid >= resp[0]) & (grid < resp[1])
                rr = [np.nanmean(np.nanmean(_event_mat(Dv[:, c], ts, dev, grid), 0)[wm]) for c in somacols]
                bc = somacols[int(np.nanargmax(rr))]
                roi_dev_tr = _event_mat(Dv[:, bc], ts, dev, grid)
                roi_ctl_tr = _event_mat(Dv[:, bc], ts, ctl, grid)
                ex_plane = dict(plane=pl, roi=int(bc), pm_dev=pm_dev, pm_ctl=pm_ctl,
                                roi_dev=roi_dev_tr, roi_ctl=roi_ctl_tr)
        roi_dev = np.array(roi_dev); roi_ctl = np.array(roi_ctl)
        sess_data.append(dict(subject=subj, dev=roi_dev, ctl=roi_ctl))
        if subj == example_subject and example is None and ex_plane is not None:
            example = dict(subject=subj, uid=ex_plane["roi"], mua_ch=ex_plane["plane"], mua_nunits=best_n,
                           dev_trials=ex_plane["roi_dev"], ctl_trials=ex_plane["roi_ctl"],
                           mua_dev=ex_plane["pm_dev"], mua_ctl=ex_plane["pm_ctl"])
        if verbose:
            print(f"  {subj} {date}: {roi_dev.shape[0]} soma ROIs ({time.time()-t0:.0f}s)")
        fh.close()
    return dict(sess=sess_data, example=example, cen=grid, resp=resp)
