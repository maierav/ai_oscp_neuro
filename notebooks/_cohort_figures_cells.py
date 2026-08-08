# Cells for cohort_diagnostics.ipynb — three cohort-level figures, all from committed data
# (no DANDI streaming). Each returns/saves PNG+PDF+SVG. Colorblind-safe (Okabe-Ito).
# ---- CELL: setup ----
import os, sys, json, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
from scipy import stats as ss

DATA = "../data" if os.path.basename(os.getcwd()) == "notebooks" else "data"
FIGS = "../figures" if os.path.basename(os.getcwd()) == "notebooks" else "figures"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")
import inclusion_flow_helpers as ifh

def _savefig_all(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{FIGS}/{stem}.{ext}", dpi=185, bbox_inches="tight")

def _overlaps(fig):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    T = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
    return [(a.get_text()[:12], b.get_text()[:12]) for i, (a, ba) in enumerate(T)
            for b, bb in T[i+1:] if ba.overlaps(bb)]

def _base_area(a):
    m = re.match(r"(VIS[a-z]+)", str(a)); return m.group(1) if m else str(a)


# ---- CELL: Figure 1 — inclusion flow (CONSORT) ----
def make_inclusion_flow():
    SC = pd.read_csv(f"{DATA}/paradigm_session_counts.csv").set_index("paradigm")
    OD = pd.read_parquet(f"{DATA}/oddball_confirmatory_units.parquet")
    SEQ = pd.read_parquet(f"{DATA}/sequence_units.parquet")
    DUR = pd.read_parquet(f"{DATA}/duration_units_layered.parquet")
    SM = pd.read_parquet(f"{DATA}/sensorimotor_multisession_units.parquet")
    CM = pd.read_parquet(f"{DATA}/crossscale_mechanism.parquet")
    def sc(p, k): return int(SC.loc[p, k])
    NP = dict(
      standard_oddball=[dict(label="ecephys\nsessions", count=sc("standard_oddball","n_sessions"), drop=f"−{sc('standard_oddball','n_sessions')-sc('standard_oddball','n_ccf_sessions')} no CCF"),
                        dict(label="CCF-aligned\nsessions", count=sc("standard_oddball","n_ccf_sessions")),
                        dict(label="QC & VIS-area\nunits", count=len(OD), drop=f"−{len(OD)-int((OD.resp_p<0.05).sum())} non-responsive"),
                        dict(label="Final: responsive\n(resp_p<0.05)", count=int((OD.resp_p<0.05).sum()))],
      sequence=[dict(label="ecephys\nsessions", count=sc("sequence","n_sessions"), drop=f"−{sc('sequence','n_sessions')-sc('sequence','n_ccf_sessions')} no CCF"),
                dict(label="CCF-aligned\nsessions", count=sc("sequence","n_ccf_sessions")),
                dict(label="QC & VIS-area\nunits", count=len(SEQ)),
                dict(label="Final: all QC&VIS\n(no resp. gate)", count=len(SEQ))],
      duration=[dict(label="ecephys\nsessions", count=sc("duration","n_sessions"), drop=f"−{sc('duration','n_sessions')-sc('duration','n_ccf_sessions')} no CCF"),
                dict(label="CCF-aligned\nsessions", count=sc("duration","n_ccf_sessions")),
                dict(label="QC & VIS-area\nunits", count=len(DUR)),
                dict(label="Final: all QC&VIS\n(no resp. gate)", count=len(DUR))],
      sensorimotor=[dict(label="ecephys\nsessions", count=sc("sensorimotor","n_sessions"), drop=f"−{sc('sensorimotor','n_sessions')-sc('sensorimotor','n_ccf_sessions')} no CCF"),
                    dict(label="CCF-aligned\nsessions", count=sc("sensorimotor","n_ccf_sessions"), drop="−2 no running"),
                    dict(label="QC & VIS units\n(6 run sessions)", count=len(SM), drop=f"−{len(SM)-int((SM.rstd>0.1).sum())} rstd≤0.1"),
                    dict(label="Final: responsive\n(rstd>0.1 Hz)", count=int((SM.rstd>0.1).sum()))],
    )
    def xs_funnel(mod):
        s = CM[CM.modality == mod]; nresp = int(s.resp.sum())
        stem = "ecephys" if mod == "ecephys" else ("mesoscope" if mod == "mesoscope" else "SLAP2")
        if mod == "ecephys":
            return [dict(label="ecephys\nsessions", count=int(s.subject.nunique())),
                    dict(label="CCF-aligned\nsessions", count=int(s.subject.nunique())),
                    dict(label="QC & VIS-area\nunits", count=len(s), drop=f"−{len(s)-nresp} non-resp."),
                    dict(label="Final: responsive", count=nresp)]
        return [dict(label=f"{stem}\nsessions", count=int(s.subject.nunique())),
                dict(label="CCF\n(n/a for 2p)", na=True),
                dict(label="somatic ROIs", count=len(s), drop=f"−{len(s)-nresp} non-resp."),
                dict(label="Final: responsive", count=nresp)]
    XS = {m: xs_funnel(m) for m in ["ecephys", "mesoscope", "slap2"]}
    fig = plt.figure(figsize=(13, 9.2))
    gsp = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.0], hspace=0.16)
    axA = fig.add_subplot(gsp[0]); axB = fig.add_subplot(gsp[1])
    for ax in (axA, axB): ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    labels = {"standard_oddball": "Feature-oddball", "sequence": "Sequence",
              "duration": "Duration/timing", "sensorimotor": "Sensorimotor"}
    for xc, (para, st) in zip([0.13, 0.38, 0.63, 0.88], NP.items()):
        ifh.draw_funnel(axA, xc, 0.185, st, ifh.PARADIGM_COLOR[para], labels[para])
    mlab = {"ecephys": "Neuropixels", "mesoscope": "Mesoscope", "slap2": "SLAP2"}
    for xc, (mod, st) in zip([0.18, 0.50, 0.82], XS.items()):
        ifh.draw_funnel(axB, xc, 0.20, st, ifh.MODALITY_COLOR[mod], mlab[mod])
    from inclusion_flow_helpers import draw_funnel  # noqa
    for ax, lt in [(axA, "a"), (axB, "b")]:
        ax.text(-0.02, 1.06, lt, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    axA.set_title("A · Neuropixels — inclusion flow per paradigm", loc="left", fontsize=9, y=1.02)
    axB.set_title("B · Feature-oddball cross-scale — inclusion flow per recording modality", loc="left", fontsize=9, y=1.02)
    fig.suptitle("Inclusion flow: sessions → CCF → QC units → responsiveness → final analysis set", fontsize=10.5, y=0.985)
    _savefig_all(fig, "inclusion_flow")
    return fig, _overlaps(fig)


# ---- CELL: Figure 2 — cross-modality per-animal ----
def make_crossmodality_per_animal():
    CM = pd.read_parquet(f"{DATA}/crossscale_mechanism.parquet")
    CM["barea"] = CM.area.map(_base_area)
    recs = []
    for mod in ["ecephys", "mesoscope"]:
        for ar in ["VISp", "VISl"]:
            s = CM[(CM.modality == mod) & (CM.barea == ar) & (CM.resp)].dropna(subset=["DvI"])
            if not len(s): continue
            for subj, v in s.groupby("subject").DvI.median().items():
                recs.append(dict(modality=mod, area=ar, subject=subj, dvi=float(v),
                                 n_units=int((s.subject == subj).sum())))
    PA = pd.DataFrame(recs); PA.to_csv(f"{DATA}/crossmodality_per_animal.csv", index=False)
    MC = ifh.MODALITY_COLOR
    fig = plt.figure(figsize=(11, 4.8))
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.30)
    axA = fig.add_subplot(gsp[0]); axB = fig.add_subplot(gsp[1])
    groups = [("ecephys", "VISp"), ("ecephys", "VISl"), ("mesoscope", "VISp"), ("mesoscope", "VISl")]
    rng = np.random.default_rng(1)
    for i, (mod, ar) in enumerate(groups):
        s = PA[(PA.modality == mod) & (PA.area == ar)]; col = MC[mod]
        axA.scatter(i + rng.uniform(-0.11, 0.11, len(s)), s.dvi, s=40 + 22*np.sqrt(s.n_units/200),
                    color=col, alpha=0.85, edgecolors="white", lw=0.6, zorder=3)
        med = s.dvi.median(); axA.plot([i-0.22, i+0.22], [med, med], color=col, lw=2.6, solid_capstyle="round", zorder=4)
        axA.text(i, -0.585, f"{s.subject.nunique()} mice\n{s.n_units.sum():,} units", ha="center", va="top", fontsize=6.2, color="#444")
    axA.axhline(0, color="#999", lw=1, ls="--", zorder=0)
    axA.set_xticks(range(4)); axA.set_xticklabels(["VISp", "VISl", "VISp", "VISl"], fontsize=8)
    axA.set_xlim(-0.5, 3.5); axA.set_ylim(-0.72, 0.92)
    axA.set_ylabel("per-animal median DvI (feature-oddball 90°)")
    for x0, x1, mod, lab in [(-0.35, 1.35, "ecephys", "Neuropixels"), (1.65, 3.35, "mesoscope", "Mesoscope")]:
        axA.plot([x0, x1], [0.86, 0.86], color=MC[mod], lw=2.2, solid_capstyle="round")
        axA.text((x0+x1)/2, 0.89, lab, ha="center", va="bottom", fontsize=8, fontweight="bold", color=MC[mod])
    for sp in ["top", "right"]: axA.spines[sp].set_visible(False)
    axA.text(0.5, 1.09, "dot area ∝ units/animal · tick = group median", transform=axA.transAxes, fontsize=5.8, color="#777", ha="center")
    # gate sensitivity
    gs_rows = []
    for mod in ["ecephys", "mesoscope"]:
        for ar in ["VISp", "VISl"]:
            on = CM[(CM.modality == mod) & (CM.barea == ar) & (CM.resp)].dropna(subset=["DvI"])
            al = CM[(CM.modality == mod) & (CM.barea == ar)].dropna(subset=["DvI"])
            f = lambda d: d.groupby("subject").DvI.median().median()
            gs_rows.append((mod, ar, f(on), f(al)))
    for mod, ar, on, al in gs_rows:
        col = MC[mod]; ls = "-" if ar == "VISp" else "--"
        dy = -0.028 if (mod == "ecephys" and ar == "VISl") else 0
        axB.plot([0, 1], [on, al], color=col, lw=1.8, ls=ls, marker="o", ms=5, zorder=3)
        axB.text(1.04, al + dy, f"{mod[:4]} {ar}", fontsize=6, color=col, va="center")
    axB.axhline(0, color="#999", lw=1, ls="--", zorder=0)
    axB.set_xticks([0, 1]); axB.set_xticklabels(["responsive\nonly", "all QC\ncells"], fontsize=7)
    axB.set_xlim(-0.15, 1.6); axB.set_ylim(-0.1, 0.5); axB.set_ylabel("group median DvI")
    for sp in ["top", "right"]: axB.spines[sp].set_visible(False)
    for ax, lt in [(axA, "a"), (axB, "b")]:
        ax.text(-0.155, 1.10, lt, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    axA.set_title("A · Feature-oddball DvI per animal — matched areas, two scales", loc="left", fontsize=8.5, y=1.03)
    axB.set_title("B · Robust to responsiveness gate", loc="left", fontsize=8.5, y=1.03)
    fig.suptitle("Cross-modality prediction error: one point per animal (SLAP2 excluded — no control-referenced DvI)", fontsize=9.5, y=1.02)
    _savefig_all(fig, "crossmodality_per_animal")
    return fig, _overlaps(fig)


# ---- CELL: Figure 3 — sensorimotor equivalence + confound ----
def make_sensorimotor_equivalence(SESOI=0.10):
    SM = pd.read_parquet(f"{DATA}/sensorimotor_multisession_units.parquet")
    BB = pd.read_parquet(f"{DATA}/sensorimotor_behavioral_balance.parquet")
    g = SM[SM.rstd > 0.1].dropna(subset=["motor_orientation_90"])
    x = g.motor_orientation_90.values; n = len(x)
    mean = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    p_lower = ss.t.sf((mean - (-SESOI)) / se, n - 1)
    p_upper = ss.t.cdf((mean - SESOI) / se, n - 1)
    tost_p = max(p_lower, p_upper)
    ci90 = (mean - 1.6449 * se, mean + 1.6449 * se)
    t_stat = mean / se
    bf01 = float(np.sqrt(n) * (1 + t_stat**2 / (n - 1)) ** (-n / 2))
    # per-session medians + confound table
    eff = g.groupby("subject").motor_orientation_90.median()
    fc = BB.groupby(["subject", "arm"]).frac_session.median().unstack()
    conf = eff.rename("dvi").to_frame().join((fc["open"] - fc["closed"]).rename("order_gap"))
    conf["n_units"] = g.groupby("subject").size()
    conf.reset_index().to_csv(f"{DATA}/sensorimotor_confound.csv", index=False)
    GREY = "#555555"; ALARM = "#D55E00"; CB = "#0072B2"; CO = ALARM
    fig = plt.figure(figsize=(11, 4.6))
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.30)
    axA = fig.add_subplot(gsp[0]); axB = fig.add_subplot(gsp[1])
    axA.axvspan(-SESOI, SESOI, color=GREY, alpha=0.10, zorder=0)
    axA.axvline(0, color="#999", lw=1, ls="--", zorder=1)
    for xv in (-SESOI, SESOI): axA.axvline(xv, color=GREY, lw=1, ls=":", zorder=1)
    rng = np.random.default_rng(2)
    axA.scatter(eff.values, 0.35 + rng.uniform(-0.05, 0.05, len(eff)),
                s=30 + 20*np.sqrt(conf.n_units.values/45), color=GREY, alpha=0.7, edgecolors="white", lw=0.5, zorder=3)
    axA.plot(ci90, [0.75, 0.75], color=GREY, lw=3, solid_capstyle="round", zorder=4)
    axA.plot(mean, 0.75, "D", color=GREY, ms=9, mfc="white", mew=1.8, zorder=5)
    axA.text(mean, 0.86, f"pooled {mean:+.3f}\n90% CI [{ci90[0]:+.2f}, {ci90[1]:+.2f}]", ha="center", va="bottom", fontsize=6.5, color="#222")
    axA.set_ylim(0, 1.15); axA.set_xlim(-0.45, 0.45); axA.set_yticks([])
    axA.set_xlabel("closed − open DvI (90° deviant)")
    axA.text(0.5, 0.46, f"per-session medians (n={len(eff)})", transform=axA.transAxes, ha="center", fontsize=6, color="#777")
    axA.text(0.02, 0.06, f"TOST p={tost_p:.2f} → equivalence NOT established\n(90% CI exceeds −SESOI)\nBF₀₁≈{bf01:.0f} → moderate evidence for null",
             transform=axA.transAxes, fontsize=6.2, color="#222", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#ccc", lw=0.8))
    for sp in ["top", "right", "left"]: axA.spines[sp].set_visible(False)
    for subj, d in BB.groupby("subject"):
        cc = d[d.arm == "closed"].frac_session.median(); oo = d[d.arm == "open"].frac_session.median()
        axB.plot([0, 1], [cc, oo], color="#bbb", lw=1, zorder=2)
        axB.scatter([0], [cc], color=CB, s=28, zorder=3); axB.scatter([1], [oo], color=CO, s=28, zorder=3)
    axB.set_xticks([0, 1]); axB.set_xticklabels(["closed-loop\n(deviants)", "open-loop\n(control)"], fontsize=7.5)
    axB.set_xlim(-0.25, 1.25); axB.set_ylim(0, 1.02); axB.set_ylabel("median time in session (fraction)")
    axB.text(0.5, 0.5, "control block always\nrun LATER than deviants\n→ drift confounds the\nclosed−open contrast",
             transform=axB.transAxes, ha="center", va="center", fontsize=6.5, color=ALARM, fontweight="bold")
    for sp in ["top", "right"]: axB.spines[sp].set_visible(False)
    for ax, lt in [(axA, "a"), (axB, "b")]:
        ax.text(-0.12, 1.10, lt, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    axA.set_title("A · Equivalence test — non-detection is not proven absence", loc="left", fontsize=8.5, y=1.03)
    axB.set_title("B · The block-order confound (structural, all 6 sessions)", loc="left", fontsize=8.5, y=1.03)
    fig.suptitle("Sensorimotor: not detected, not equivalent, and confounded by block order", fontsize=9.5, y=1.0)
    _savefig_all(fig, "sensorimotor_equivalence")
    return fig, _overlaps(fig), dict(mean=mean, se=se, ci90=ci90, tost_p=tost_p, bf01=bf01, n=n)
