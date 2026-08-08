# Inclusion-flow (CONSORT-style) funnel drawing — colorblind-safe (Okabe-Ito).
# Each track is a vertical stack of stage boxes with counts, connected by arrows;
# per-arrow drop annotations name how many units/sessions were excluded and why.
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Okabe-Ito, CVD-safe
OKABE = dict(black="#000000", orange="#E69F00", skyblue="#56B4E9", green="#009E73",
             yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", purple="#CC79A7")
PARADIGM_COLOR = {"standard_oddball": OKABE["vermillion"], "sequence": OKABE["green"],
                  "duration": OKABE["blue"], "sensorimotor": "#555555"}
MODALITY_COLOR = {"ecephys": OKABE["blue"], "mesoscope": OKABE["orange"], "slap2": OKABE["purple"]}


def _tint(hexcol, alpha, bg=(1, 1, 1)):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hexcol)
    return tuple(alpha * c + (1 - alpha) * w for c, w in zip((r, g, b), bg))


def draw_funnel(ax, xc, width, stages, color, header, top=0.90, bottom=0.05, box_h=0.115):
    """Draw one funnel of stage boxes at column center xc (axes coords 0..1).

    stages: list of dict(label=str, count=int|None, na=bool, drop=str|None) top→bottom.
    `drop` annotates the arrow BELOW that box. na=True renders a greyed N/A box.
    """
    n = len(stages)
    ys = np.linspace(top - box_h / 2, bottom + box_h / 2, n)
    ax.text(xc, top + 0.055, header, ha="center", va="center", fontsize=8.2,
            fontweight="bold", color=color, zorder=5)
    for i, st in enumerate(stages):
        y = ys[i]; na = st.get("na", False)
        fc = "#ffffff" if na else _tint(color, 0.16)
        ec = "#bcbcbc" if na else color
        ax.add_patch(FancyBboxPatch((xc - width / 2, y - box_h / 2), width, box_h,
                     boxstyle="round,pad=0.004,rounding_size=0.010", fc=fc, ec=ec, lw=1.3, zorder=3))
        cnt = "N/A" if na else f"{st['count']:,}"
        ax.text(xc, y + 0.021, cnt, ha="center", va="center", fontweight="bold",
                fontsize=8.6, color=("#aaaaaa" if na else "#111111"), zorder=4)
        ax.text(xc, y - 0.026, st["label"], ha="center", va="center",
                fontsize=6.0, color=("#aaaaaa" if na else "#333333"), zorder=4, linespacing=0.95)
        if i < n - 1:
            y2 = ys[i + 1]
            ax.annotate("", xy=(xc, y2 + box_h / 2), xytext=(xc, y - box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", lw=1.1), zorder=2)
            if st.get("drop"):
                ax.text(xc + width / 2 + 0.006, (y - box_h / 2 + y2 + box_h / 2) / 2,
                        st["drop"], ha="left", va="center", fontsize=5.5,
                        color="#b03020", zorder=4, linespacing=0.95)
