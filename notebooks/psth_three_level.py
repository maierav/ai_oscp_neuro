# Shared three-level PSTH helper (inlined into each result notebook for Colab-standalone use).
# Levels: (A) one example unit with a trial-variability band, (B) one example session
# (mean over units +/- SEM), (C) grand average across sessions (per-session lines + mean +/- SEM
# across session means). Designed as a systems-neuro-standard diagnostic: trigger/timebase
# problems, window mis-selection, and outlier sessions are all visible at a glance.
import numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def _sem(x, axis=0):
    x = np.asarray(x, float); n = np.sum(~np.isnan(x), axis=axis)
    return np.nanstd(x, axis=axis) / np.sqrt(np.maximum(n, 1))

def psth_three_level(cen, example, sessions, resp_win=None, dev_label="deviant",
                     ctl_label="control", dev_color="#c0392b", ctl_color="#3b6ea5",
                     smooth=1.5, ylabel="firing rate (Hz)", suptitle=None, savepath=None):
    """
    Four-level PSTH diagnostic (systems-neuro standard):
      A example CHANNEL (MUA, all units on one channel) - trial mean +/- SEM  [trigger/timebase check]
      B example UNIT (single sorted unit)               - trial mean +/- SEM
      C example SESSION                                  - unit  mean +/- SEM
      D grand average across sessions                    - per-session lines + across-session mean +/- SEM

    cen       : (T,) bin-centre times in SECONDS.
    example   : dict with 'subject','uid','dev_trials','ctl_trials' (n_trials x T) for the example unit,
                and 'mua_ch','mua_dev','mua_ctl' (n_trials x T) for the example channel's MUA.
    sessions  : list of dicts, each {'subject','dev' (n_units x T),'ctl' (n_units x T)}.
    resp_win  : optional (t0,t1) seconds; shaded response window on every panel.
    Returns (fig, overlaps_count). Gaussian smoothing is display-only.
    """
    t = np.asarray(cen) * 1000.0
    def sm(a): return gaussian_filter1d(np.asarray(a, float), smooth, axis=-1)
    fig, (axA, axB, axC, axD) = plt.subplots(1, 4, figsize=(16.0, 4.5))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.27, wspace=0.30)

    # ---- Panel A: example channel MUA, mean +/- SEM across trials ----
    if example.get("mua_dev") is not None:
        for M, c, lab in [(example["mua_dev"], dev_color, dev_label),
                          (example["mua_ctl"], ctl_color, ctl_label)]:
            m = sm(np.nanmean(M, 0)); e = sm(_sem(M, 0))
            axA.plot(t, m, color=c, lw=1.8, label=f"{lab} ({M.shape[0]} tr)")
            axA.fill_between(t, m - e, m + e, color=c, alpha=0.22, lw=0)
        axA.set_title(f"A · example channel MUA ({example['subject']} ch{example.get('mua_ch','?')})\n"
                      f"mean ± SEM across trials", loc="left", fontsize=8)
    axA.set_ylabel("MUA rate (Hz)")

    # ---- Panel B: example unit, trial band ----
    for M, c, lab in [(example["dev_trials"], dev_color, dev_label),
                      (example["ctl_trials"], ctl_color, ctl_label)]:
        m = sm(np.nanmean(M, 0)); e = sm(_sem(M, 0))
        axB.plot(t, m, color=c, lw=1.8, label=f"{lab} ({M.shape[0]} tr)")
        axB.fill_between(t, m - e, m + e, color=c, alpha=0.22, lw=0)
    axB.set_title(f"B · example unit ({example['subject']} #{example['uid']})\nmean ± SEM across trials",
                  loc="left", fontsize=8)
    axB.set_ylabel(ylabel)

    # ---- Panel C: example session, mean +/- SEM across units ----
    exs = next((s for s in sessions if s["subject"] == example["subject"]), sessions[0])
    for A, c, lab in [(exs["dev"], dev_color, dev_label), (exs["ctl"], ctl_color, ctl_label)]:
        m = sm(np.nanmean(A, 0)); e = sm(_sem(A, 0))
        axC.plot(t, m, color=c, lw=1.8, label=f"{lab} (n={A.shape[0]} units)")
        axC.fill_between(t, m - e, m + e, color=c, alpha=0.22, lw=0)
    axC.set_title(f"C · example session ({exs['subject']})\nmean ± SEM across units", loc="left", fontsize=8)
    axC.set_ylabel(ylabel)

    # ---- Panel D: grand average, per-session lines + mean +/- SEM across sessions ----
    for cond, c in [("dev", dev_color), ("ctl", ctl_color)]:
        permean = np.array([np.nanmean(s[cond], 0) for s in sessions])  # (n_sess x T)
        for row in permean:
            axD.plot(t, sm(row), color=c, lw=0.5, alpha=0.30)
        gm = sm(np.nanmean(permean, 0)); ge = sm(_sem(permean, 0))
        axD.plot(t, gm, color=c, lw=2.2, label=f"{'dev' if cond=='dev' else 'ctl'} mean (n={len(sessions)} sess)")
        axD.fill_between(t, gm - ge, gm + ge, color=c, alpha=0.20, lw=0)
    axD.set_title(f"D · grand average ({len(sessions)} sessions)\nthin = per-session mean; band = ± SEM across sessions",
                  loc="left", fontsize=8)
    axD.set_ylabel(ylabel)

    for ax in (axA, axB, axC, axD):
        ax.axvline(0, color="k", lw=0.7, ls=":")
        if resp_win is not None:
            ax.axvspan(resp_win[0] * 1000, resp_win[1] * 1000, color="0.85", alpha=0.5, zorder=0)
        ax.set_xlabel("time from onset (ms)")
        ax.margins(x=0.02)
        ax.legend(frameon=False, fontsize=6.3, loc="upper left",
                  bbox_to_anchor=(0.0, -0.16), ncol=1, handlelength=1.4)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    if suptitle: fig.suptitle(suptitle, fontsize=9.2, y=0.965)

    # overlap self-check (draw first so bbox_to_anchor legend positions are resolved)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    T_ = [(x, x.get_window_extent(r)) for x in fig.findobj(plt.matplotlib.text.Text)
          if x.get_text().strip() and x.get_visible()]
    tl = set(x for a in fig.axes for x in a.get_xticklabels() + a.get_yticklabels())
    ov = [(a.get_text()[:10], b.get_text()[:10]) for i, (a, ba) in enumerate(T_)
          for b, bb in T_[i + 1:] if ba.overlaps(bb) and not (a in tl and b in tl)]
    if savepath: fig.savefig(savepath, dpi=185, bbox_inches="tight")
    return fig, len(ov)
