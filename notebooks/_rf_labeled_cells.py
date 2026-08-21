# Labeled example receptive fields across three recording scales.
# Each example carries a compact, human-legible unique ID encoding its provenance:
#   NP·<subject>·<date>·<area><layer>·pr<Probe>·d<depth µm>·u<unit>     (Neuropixels)
#   MS·<subject>·<date>·<area>·<planedepth µm>·roi<id>·<x,y µm>          (Mesoscope 2p soma)
#   SL·<subject>·<date>·<DMD>·roi<id>·<x,y px>                          (SLAP2 glutamate)
# The ID is built so someone else can locate the exact source (technique, animal, session
# date, anatomy, and within-session position) at a glance, without a lookup table.
import re, h5py, remfile, requests, numpy as np, pandas as pd
from scipy.optimize import curve_fit

# ---------- streaming helpers ----------
def s3_url(dandiset, asset_id, version="draft"):
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/{version}/assets/{asset_id}/download/",
        allow_redirects=False, timeout=30)
    return r.headers["Location"]

def open_nwb(dandiset, asset_id):
    return h5py.File(remfile.File(s3_url(dandiset, asset_id)), "r")

def _col(group, name):
    v = group[name][:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v])

def _dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x

def _date_of(fh):
    """Return YYYY-MM-DD from session_start_time."""
    t = _dec(fh["session_start_time"][()])
    return t[:10]

# ---------- 2-D Gaussian fit (shared) ----------
def _fit_generic(m, XXg, YYg, edge):
    p0 = [m.max(), XXg.ravel()[m.argmax()], YYg.ravel()[m.argmax()], 12, 12, np.median(m)]
    popt, _ = curve_fit(
        lambda c, a, x0, y0, sx, sy, o:
            (a*np.exp(-((c[0]-x0)**2/(2*sx**2)+(c[1]-y0)**2/(2*sy**2)))+o).ravel(),
        (XXg, YYg), m.ravel(), p0=p0, maxfev=4000,
        bounds=([0, -edge-20, -edge-20, 3, 3, -abs(m).max()],
                [np.inf, edge+20, edge+20, 80, 80, abs(m).max()]))
    fitv = (popt[0]*np.exp(-((XXg-popt[1])**2/(2*popt[3]**2)+(YYg-popt[2])**2/(2*popt[4]**2)))+popt[5])
    r2 = 1 - np.sum((m-fitv)**2)/np.sum((m-m.mean())**2)
    return dict(x0=popt[1], y0=popt[2], fwhm=2.355*np.sqrt(popt[3]*popt[4]), r2=r2, amp=popt[0],
                centered=(abs(popt[1]) <= edge-5 and abs(popt[2]) <= edge-5))


# ---------- Neuropixels ----------
def ecephys_rf(asset="9b9e8abe-7b43-47f1-b8e1-4114f87898a1", topn=6):
    fh = open_nwb("001637", asset)
    subj = _dec(fh["general/subject/subject_id"][()]); date = _date_of(fh)
    g = fh["intervals"]["RF mapping_presentations"]
    X = _col(g, "X").astype(float); Y = _col(g, "Y").astype(float); t0 = g["start_time"][:]
    xs = np.array(sorted(set(X))); ys = np.array(sorted(set(Y)))
    xi = {v: k for k, v in enumerate(xs)}; yi = {v: k for k, v in enumerate(ys)}
    tX = np.array([xi[v] for v in X]); tY = np.array([yi[v] for v in Y])
    U = fh["units"]
    st = U["spike_times"][:]; sti = U["spike_times_index"][:]
    def spikes(i): return st[(0 if i == 0 else sti[i-1]):sti[i]]
    n = len(sti)
    qc = U["default_qc"][:] if "default_qc" in U else np.ones(n, bool)
    depth = U["depth"][:]; dev = _col(U, "device_name"); eci = U["extremum_channel_index"][:]
    E = fh["general/extracellular_ephys/electrodes"]
    eloc = _col(E, "location"); egroup = _col(E, "group_name")
    groups = sorted(set(egroup), key=lambda gn: np.where(egroup == gn)[0][0])
    offset = {gn: int(np.where(egroup == gn)[0][0]) for gn in groups}
    blocklen = {gn: int((egroup == gn).sum()) for gn in groups}
    d2g = {d: (d if d in groups else groups[0]) for d in set(dev)}
    u_region = np.array([eloc[offset[d2g[dev[i]]] + min(int(eci[i]), blocklen[d2g[dev[i]]]-1)]
                         for i in range(n)])
    is_vis = np.array([bool(re.match(r"VIS", r)) for r in u_region])
    sel = np.where(qc & is_vis)[0]
    RESP, BASE = (0.03, 0.20), (-0.15, -0.02)
    def rf_map(i):
        sp = spikes(i); grid = np.zeros((len(ys), len(xs))); cnt = np.zeros_like(grid); base = 0.0
        for k in range(len(t0)):
            s = t0[k]
            r = np.searchsorted(sp, [s+RESP[0], s+RESP[1]]); grid[tY[k], tX[k]] += (r[1]-r[0])/(RESP[1]-RESP[0])
            b = np.searchsorted(sp, [s+BASE[0], s+BASE[1]]); base += (b[1]-b[0])/(BASE[1]-BASE[0])
            cnt[tY[k], tX[k]] += 1
        return grid/np.clip(cnt, 1, None) - base/len(t0)
    XX, YY = np.meshgrid(xs, ys)
    maps = {}; rows = []
    for i in sel:
        m = rf_map(i); maps[i] = m
        if m.max() <= 0: continue
        try:
            f = _fit_generic(m, XX, YY, xs.max())
            probe = dev[i].replace("Probe", "pr")
            area = u_region[i]
            uid = f"NP\u00b7{subj}\u00b7{date}\u00b7{area}\u00b7{probe}\u00b7d{int(round(depth[i]))}\u00b7u{i}"
            f.update(idx=i, uid=uid, area=area, probe=dev[i], depth_um=float(depth[i]),
                     subject=subj, date=date)
            rows.append(f)
        except Exception:
            pass
    Q = pd.DataFrame(rows)
    pick = Q[(Q.r2 > 0.4) & (Q.centered) & (Q.amp > 2)].sort_values("r2", ascending=False).head(topn)
    fh.close()
    return dict(xs=xs, ys=ys, maps=maps, pick=pick, idcol="idx",
                title="Neuropixels\nspikes", color="#08519c")


# ---------- Mesoscope 2p ----------
def mesoscope_rf(asset="83e0c8f3-5208-417c-87c4-bc4617b0f834", planes=None, topn=6):
    fhm = open_nwb("001768", asset)
    subj = _dec(fhm["general/subject/subject_id"][()]); date = _date_of(fhm)
    gm = fhm["intervals"]["RF mapping_presentations"]
    Xm = _col(gm, "X").astype(float); Ym = _col(gm, "Y").astype(float); t0m = gm["start_time"][:]
    xsm = np.array(sorted(set(Xm))); ysm = np.array(sorted(set(Ym)))
    tXm = np.array([{v: k for k, v in enumerate(xsm)}[v] for v in Xm])
    tYm = np.array([{v: k for k, v in enumerate(ysm)}[v] for v in Ym])
    XXm, YYm = np.meshgrid(xsm, ysm)
    if planes is None:
        # default to one VISp + one VISl plane — enough for a 6-example gallery, and far
        # faster than scanning all 8 planes (pass planes=[...] to widen).
        vis = [p for p in fhm["processing"].keys() if p.startswith("VIS")]
        visp = [p for p in vis if p.startswith("VISp")][:1]
        visl = [p for p in vis if p.startswith("VISl")][:1]
        planes = visp + visl or vis[:2]
    RESP_M, BASE_M = (0.1, 0.8), (-0.3, 0.0)
    maps = {}; rows = []
    for PLANE in planes:
        op = fhm["general/optophysiology"][PLANE]
        locs = _dec(op["location"][()])
        mloc = re.search(r"Structure:\s*(\S+)\s*Depth:\s*(\d+)", locs)
        area = mloc.group(1) if mloc else PLANE.split("_")[0]
        pdepth = int(mloc.group(2)) if mloc else -1
        gsp = op["grid_spacing"][:] if "grid_spacing" in op else np.array([1.0, 1.0])
        p0 = fhm["processing"][PLANE]
        dff = p0["dff_timeseries"]["dff_timeseries"]; data = dff["data"][:]; tsv = dff["timestamps"][:]
        rt = p0["image_segmentation"]["roi_table"]
        soma = np.where(rt["is_soma"][:].astype(bool))[0]
        im = rt["image_mask"]
        def rf_map(roi):
            grid = np.zeros((len(ysm), len(xsm))); cnt = np.zeros_like(grid); tr = data[:, roi]
            for k in range(len(t0m)):
                s = t0m[k]
                rm = (tsv >= s+RESP_M[0]) & (tsv < s+RESP_M[1]); bm = (tsv >= s+BASE_M[0]) & (tsv < s+BASE_M[1])
                if rm.sum() < 1 or bm.sum() < 1: continue
                grid[tYm[k], tXm[k]] += np.nanmean(tr[rm]) - np.nanmean(tr[bm]); cnt[tYm[k], tXm[k]] += 1
            return grid/np.clip(cnt, 1, None)
        for roi in soma:
            key = (PLANE, roi)
            m = rf_map(roi); maps[key] = m
            if not np.isfinite(m).all() or m.max() <= 0: continue
            try:
                f = _fit_generic(m, XXm, YYm, xsm.max())
                mask = im[roi]; yy, xx = np.where(mask > 0)
                cx = int(round(xx.mean()*gsp[0])); cy = int(round(yy.mean()*gsp[1]))
                uid = f"MS\u00b7{subj}\u00b7{date}\u00b7{area}\u00b7{pdepth}\u00b5m\u00b7roi{roi}\u00b7({cx},{cy})\u00b5m"
                f.update(idx=key, uid=uid, area=area, plane=PLANE, plane_depth_um=pdepth,
                         roi=int(roi), cx_um=cx, cy_um=cy, subject=subj, date=date)
                rows.append(f)
            except Exception:
                pass
    Q = pd.DataFrame(rows)
    pick = Q[(Q.r2 > 0.35) & (Q.centered) & (Q.amp > 0.15)].sort_values("r2", ascending=False).head(topn)
    fhm.close()
    return dict(xs=xsm, ys=ysm, maps=maps, pick=pick, idcol="idx",
                title="Mesoscope 2p\n\u0394F/F (soma)", color="#238b45")


# ---------- SLAP2 ----------
def slap2_rf(asset="44871646-ca8d-440d-b970-5756ed7cb47e", dmd="DMD1", offset=0.115, topn=6):
    fhs = open_nwb("001424", asset)
    subj = _dec(fhs["general/subject/subject_id"][()]); date = _date_of(fhs)
    gs = fhs["intervals/gratings"]
    xg = gs["x"][:]; yg = gs["y"][:]; dia = gs["diameter"][:]; t0s = gs["start_time"][:]
    rf_idx = np.where(dia < 30)[0]
    xg_r, yg_r = xg[rf_idx], yg[rf_idx]
    xsg = np.array(sorted(set(xg_r))); ysg = np.array(sorted(set(yg_r)))
    tXg = np.array([{v: k for k, v in enumerate(xsg)}[v] for v in xg_r])
    tYg = np.array([{v: k for k, v in enumerate(ysg)}[v] for v in yg_r])
    XXg, YYg = np.meshgrid(xsg, ysg)
    dff1 = fhs[f"processing/ophys/Fluorescence_{dmd}/{dmd}_dFF"]; d = dff1["data"][:]; ts = dff1["timestamps"][:]
    ts_o = fhs["processing/ophys/Fluorescence_DMD2/DMD2_dFF"]["timestamps"][:]
    if ts[-1] < 0.6*ts_o[-1]:
        ts = np.linspace(ts_o[0], ts_o[-1], d.shape[0])
    t0_r = t0s[rf_idx] + offset
    RESP_S, BASE_S = (0.05, 0.35), (-0.25, -0.02)
    # ROI centroids from pixel_mask
    ps = fhs[f"processing/ophys/ImageSegmentation/PlaneSegmentation_{dmd}"]
    pmi = ps["pixel_mask_index"][:]; pm = ps["pixel_mask"][:]
    def centroid(roi):
        lo = 0 if roi == 0 else pmi[roi-1]; hi = pmi[roi]
        px = pm[lo:hi]
        return int(round(np.mean([p[0] for p in px]))), int(round(np.mean([p[1] for p in px])))
    def rf_map(roi):
        tr = d[:, roi]; grid = np.zeros((len(ysg), len(xsg))); cnt = np.zeros_like(grid)
        for k in range(len(t0_r)):
            s = t0_r[k]
            rm = (ts >= s+RESP_S[0]) & (ts < s+RESP_S[1]); bm = (ts >= s+BASE_S[0]) & (ts < s+BASE_S[1])
            if rm.sum() < 1 or bm.sum() < 1: continue
            rv, bv = tr[rm], tr[bm]
            if not (np.isfinite(rv).any() and np.isfinite(bv).any()): continue
            val = np.nanmean(rv) - np.nanmean(bv)
            if np.isfinite(val): grid[tYg[k], tXg[k]] += val; cnt[tYg[k], tXg[k]] += 1
        return grid/np.clip(cnt, 1, None)
    maps = {}; rows = []
    for roi in range(d.shape[1]):
        m = rf_map(roi); maps[roi] = m
        if not np.isfinite(m).all() or m.max() <= 0: continue
        try:
            f = _fit_generic(m, XXg, YYg, xsg.max())
            cx, cy = centroid(roi)
            uid = f"SL\u00b7{subj}\u00b7{date}\u00b7{dmd}\u00b7roi{roi}\u00b7({cx},{cy})px"
            f.update(idx=roi, uid=uid, area="dendrite", dmd=dmd, roi=int(roi),
                     cx_px=cx, cy_px=cy, subject=subj, date=date)
            rows.append(f)
        except Exception:
            pass
    Q = pd.DataFrame(rows)
    pick = Q[(Q.r2 > 0.35) & (Q.centered) & (Q.amp > 0.08)].sort_values("r2", ascending=False).head(topn)
    fhs.close()
    return dict(xs=xsg, ys=ysg, maps=maps, pick=pick, idcol="idx",
                title="SLAP2\nglutamate \u0394F/F", color="#d94801")


# ---------- gallery renderer ----------
def render_gallery(specs, ncol=6, savepath="../figures/rf_labeled_examples.png"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(specs), ncol, figsize=(17, 8.4))
    for ri, sp in enumerate(specs):
        xg_, yg_, maps, pick, idc = sp["xs"], sp["ys"], sp["maps"], sp["pick"], sp["idcol"]
        XXr, YYr = np.meshgrid(xg_, yg_)
        ids = list(pick[idc])[:ncol]
        for ci in range(ncol):
            ax = axes[ri, ci]
            if ci < len(ids):
                row = pick[pick[idc].apply(lambda v: v == ids[ci])].iloc[0]
                m = maps[ids[ci]]; vmax = np.abs(m).max()
                ax.imshow(m, origin="lower", extent=[xg_[0], xg_[-1], yg_[0], yg_[-1]],
                          cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
                ax.contour(XXr, YYr, m, levels=[0.5*m.max()], colors="k", linewidths=0.7)
                # unique ID as the panel title (small monospace), wrapped onto two lines at the
                # midpoint separator so the (long) mesoscope IDs fit within one panel width.
                _u = row["uid"]; _parts = _u.split("\u00b7")
                _mid = (len(_parts)+1)//2
                _wrapped = "\u00b7".join(_parts[:_mid]) + "\n" + "\u00b7".join(_parts[_mid:])
                ax.set_title(_wrapped, fontsize=5.0, fontfamily="monospace", pad=3, color="#222",
                             linespacing=1.15)
                ax.text(0.5, -0.30, f"FWHM {row['fwhm']:.0f}\u00b0 · R\u00b2={row['r2']:.2f}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=6, color="#555")
                ax.set_xticks([xg_[0], 0, xg_[-1]]); ax.set_yticks([yg_[0], 0, yg_[-1]])
                ax.tick_params(labelsize=5.5)
                ax.axhline(0, color="gray", lw=0.3); ax.axvline(0, color="gray", lw=0.3)
            else:
                ax.axis("off")
        axes[ri, 0].set_ylabel("elevation (\u00b0)", fontsize=7.5)
        bb = axes[ri, 0].get_position()
        fig.text(0.008, (bb.y0+bb.y1)/2, sp["title"], rotation=90, va="center", ha="center",
                 fontsize=9.5, fontweight="bold", color=sp["color"])
    for ci in range(ncol):
        axes[-1, ci].set_xlabel("azimuth (\u00b0)", fontsize=7.5)
    fig.suptitle("Labeled example receptive fields across three recording scales", fontsize=12, y=0.985)
    fig.text(0.5, 0.945,
             "Each panel titled with a unique provenance ID: technique·subject·date·anatomy·position. "
             "Selected by 2-D Gaussian fit quality; diverging LUT centred at zero, black = half-max.",
             ha="center", fontsize=7.5, color="#444")
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.085, wspace=0.42, hspace=0.72)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(savepath.replace(".png", f".{ext}"), dpi=200, bbox_inches="tight")
    return fig


def id_table(specs):
    """Return the provenance table for the plotted examples."""
    out = []
    for sp in specs:
        p = sp["pick"].copy()
        keep = [c for c in ["uid", "subject", "date", "area", "probe", "depth_um", "plane",
                            "plane_depth_um", "roi", "cx_um", "cy_um", "dmd", "cx_px", "cy_px",
                            "fwhm", "r2"] if c in p.columns]
        out.append(p[keep])
    return pd.concat(out, ignore_index=True)
