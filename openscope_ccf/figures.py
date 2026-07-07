"""Penetration figures: 3D atlas context + per-probe laminar cross-check.

``build_probe_data`` streams a session and computes, per probe: CCF coordinates
and region per channel, spontaneous LFP band power (1-100 Hz and gamma 30-90 Hz)
from a 30 s window, and an MUA firing-rate depth profile. ``make_3d`` renders the
probes as PCA best-fit straight tracks inside a translucent Allen CCF brain
shell; ``make_laminar`` renders the region/layer strip beside the LFP + MUA
depth profiles so the CCF alignment can be visually cross-checked against the
recordings.

The Allen brain shell (root mesh) is optional: pass ``brain_mesh=(verts, faces)``
to :func:`make_3d`, or omit it for a mesh-free scatter. Fetch the mesh with
:func:`load_root_mesh` (needs ``brainglobe-atlasapi`` and network access to the
BrainGlobe atlas mirror).
"""
from __future__ import annotations
import numpy as np
from scipy import signal

from .nwbio import open_remote, unit_electrode_rows, _decode

_TAB = None


def _tab(n):
    import matplotlib.pyplot as plt
    return plt.cm.tab10(np.linspace(0, 1, 10))[:n]


def build_probe_data(asset_id: str, welch_window_s: float = 30.0, mua_bins: int = 48) -> dict:
    """Per-probe geometry + LFP band power + MUA depth profile for one session."""
    fh = open_remote(asset_id)
    try:
        u = fh["units"]
        el = fh["general/extracellular_ephys/electrodes"]
        loc = _decode(el["location"][:])
        ex, ey, ez = el["x"][:], el["y"][:], el["z"][:]
        egrp = _decode(el["group_name"][:])
        dev = _decode(u["device_name"][:])
        sti = u["spike_times_index"][:]
        elrow = unit_electrode_rows(fh)
        u_dv = ey[elrow]
        lfp_root = fh["processing/ecephys/LFP"]
        any_key = list(lfp_root.keys())[0]
        dur = lfp_root[any_key]["timestamps"][-1]

        def nspk(i):
            lo = 0 if i == 0 else sti[i - 1]
            return sti[i] - lo
        urate = np.array([nspk(i) / dur for i in range(len(dev))])

        out = {}
        for p in sorted(set(egrp)):
            keys = [k for k in lfp_root.keys() if p in k]
            if not keys:
                continue
            es = lfp_root[keys[0]]
            ch = es["electrodes"][:]
            ts = es["timestamps"]
            fs = 1.0 / np.median(np.diff(ts[:2000]))
            n = int(welch_window_s * fs)
            data = es["data"][:n, :].astype(np.float32)
            f, Pxx = signal.welch(data, fs=fs, nperseg=int(fs), axis=0)
            bp_lf = Pxx[(f >= 1) & (f <= 100)].sum(0)
            bp_gamma = Pxx[(f >= 30) & (f <= 90)].sum(0)
            m = dev == p
            dvp, rp = u_dv[m], urate[m]
            dv_ch = ey[ch]
            edges = np.linspace(dv_ch.min() - 20, dv_ch.max() + 20, mua_bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            mua_prof, _ = np.histogram(dvp, bins=edges, weights=rp)
            out[p] = dict(ch_elidx=ch, ap=ex[ch], dv=dv_ch, ml=ez[ch], region=loc[ch],
                          bp_lf=bp_lf, bp_gamma=bp_gamma,
                          mua_dv_centers=centers, mua_dv=mua_prof, n_units=int(m.sum()))
        return out
    finally:
        fh.close()


def load_root_mesh():
    """Return (verts, faces) of the Allen mouse CCF root (whole-brain) mesh."""
    from brainglobe_atlasapi import BrainGlobeAtlas
    bg = BrainGlobeAtlas("allen_mouse_25um")
    mesh = bg.mesh_from_structure("root")
    verts = np.array(mesh.points)
    faces = None
    for c in mesh.cells:
        if c.type == "triangle":
            faces = c.data
    return verts, faces


def _region_colors(regions, bg=None):
    cmap = {}
    for r in _pd_unique(regions):
        try:
            cmap[r] = np.array(bg.structures[r]["rgb_triplet"]) / 255
        except Exception:
            cmap[r] = np.array([0.7, 0.7, 0.7])
    return cmap


def _pd_unique(x):
    seen, out = set(), []
    for v in x:
        if v not in seen:
            seen.add(v); out.append(v)
    return out


def _lum(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def make_3d(probe_data: dict, label: str, path: str, brain_mesh=None, dpi=170):
    """Render probes as PCA best-fit tracks in the CCF (optionally in brain shell)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    pn = sorted(probe_data.keys())
    pcol = dict(zip(pn, _tab(len(pn))))
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if brain_mesh is not None:
        verts, faces = brain_mesh
        tri = np.stack([verts[faces][..., 0], verts[faces][..., 2], -verts[faces][..., 1]], axis=-1)
        coll = Poly3DCollection(tri, alpha=0.10, facecolor="#cfd4da", edgecolor="none", linewidths=0)
        coll.set_sort_zpos(-1)
        ax.add_collection3d(coll)
    for p in pn:
        d = probe_data[p]
        P = np.vstack([d["ap"], d["dv"], d["ml"]]).T.astype(float)
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c)
        axis = Vt[0]
        proj = (P - c) @ axis
        ends = c + np.outer([proj.min(), proj.max()], axis)
        ax.plot(ends[:, 0], ends[:, 2], -ends[:, 1], color=pcol[p], lw=3,
                solid_capstyle="round", label=p)
    ax.set_xlabel("AP (µm)", labelpad=0)
    ax.set_ylabel("ML (µm)", labelpad=0)
    ax.set_zlabel("DV (µm)", labelpad=0)
    ax.set_zticks([0, -2000, -4000, -6000])
    ax.set_zticklabels([0, 2000, 4000, 6000])
    ax.view_init(elev=16, azim=-68)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2, bbox_to_anchor=(0.0, 0.97))
    ax.set_title(f"Neuropixels penetrations in Allen CCF · {label}", fontsize=9)
    ax.set_box_aspect((13200, 11400, 8000))
    ax.set_xlim(0, 13200); ax.set_ylim(0, 11400); ax.set_zlim(-8000, 0)
    ax.grid(False)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_probe(ax_anat, ax_phys, probe_data, p, show_ylab, bg=None):
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    d = probe_data[p]
    order = np.argsort(d["dv"])
    dv, reg = d["dv"][order], d["region"][order]
    bp, gam = d["bp_lf"][order], d["bp_gamma"][order]
    cmap = _region_colors(reg, bg=bg)
    for i in range(len(dv)):
        y0 = dv[i] - 8 if i == 0 else (dv[i - 1] + dv[i]) / 2
        y1 = dv[i] + 8 if i == len(dv) - 1 else (dv[i] + dv[i + 1]) / 2
        ax_anat.axhspan(y0, y1, color=cmap[reg[i]], lw=0)
    runs, s = [], 0
    for i in range(1, len(reg) + 1):
        if i == len(reg) or reg[i] != reg[s]:
            runs.append((reg[s], dv[s], dv[i - 1])); s = i
    for name, d0, d1 in runs:
        if name in ("root", "void") or (d1 - d0) < 55:
            continue
        tc = "w" if _lum(cmap[name]) < 0.5 else "k"
        ax_anat.text(0.5, (d0 + d1) / 2, name, ha="center", va="center",
                     fontsize=5.0, color=tc, clip_on=True)
    ax_anat.set_xlim(0, 1); ax_anat.set_xticks([])
    ax_anat.set_ylim(dv.max() + 50, dv.min() - 50)
    if show_ylab:
        ax_anat.set_ylabel("DV depth (µm)")
    ax_anat.set_title(p, fontsize=8, pad=2)
    ax_phys.plot(10 * np.log10(bp), dv, color="#1f77b4", lw=1.2)
    ax_phys.plot(10 * np.log10(gam), dv, color="#d62728", lw=1.0)
    ax_phys.set_ylim(dv.max() + 50, dv.min() - 50)
    ax_phys.set_yticklabels([])
    ax_phys.set_xlabel("LFP (dB)", fontsize=6.5)
    ax_phys.tick_params(labelsize=5.5)
    axm = ax_phys.twiny()
    axm.plot(d["mua_dv"], d["mua_dv_centers"], color="#2ca02c", lw=1.2)
    axm.set_xlabel("MUA (spk/s)", color="#2ca02c", fontsize=6)
    axm.tick_params(axis="x", labelcolor="#2ca02c", labelsize=5)
    axm.set_ylim(dv.max() + 50, dv.min() - 50)
    axm.xaxis.set_major_locator(MaxNLocator(4))


def make_laminar(probe_data: dict, label: str, path: str, bg=None, dpi=150):
    """Per-probe laminar cross-check: CCF strip vs LFP power & MUA depth profiles."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    pn = sorted(probe_data.keys())
    npb = len(pn)
    ncols = 3
    nrows = int(np.ceil(npb / ncols))
    fig = plt.figure(figsize=(16, 4.5 * nrows))
    wr = []
    for _ in range(ncols):
        wr += [0.42, 0.75, 0.22]
    wr = wr[:-1]
    gs = fig.add_gridspec(nrows, ncols * 3 - 1, width_ratios=wr,
                          hspace=0.62, wspace=0.08, left=0.05, right=0.995, top=0.86, bottom=0.07)
    for idx, p in enumerate(pn):
        r = idx // ncols
        cc = (idx % ncols) * 3
        axa = fig.add_subplot(gs[r, cc])
        axp = fig.add_subplot(gs[r, cc + 1])
        _draw_probe(axa, axp, probe_data, p, show_ylab=(cc == 0), bg=bg)
    handles = [Line2D([0], [0], color="#1f77b4", lw=2, label="LFP power 1–100 Hz"),
               Line2D([0], [0], color="#d62728", lw=2, label="LFP γ 30–90 Hz"),
               Line2D([0], [0], color="#2ca02c", lw=2, label="MUA firing rate"),
               mpl.patches.Patch(color="#cccccc", label="CCF region (colour = Allen atlas)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"Laminar cross-check: CCF vs LFP power & MUA · {label} (DANDI 001637)",
                 fontsize=10, y=0.93)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
