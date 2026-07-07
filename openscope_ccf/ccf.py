"""CCF acronym decoding for the Allen Mouse Common Coordinate Framework.

Turns a per-channel CCF acronym (e.g. ``VISp5``, ``CA1``, ``DG-mo``, ``LGd-sh``)
into structured fields: cortical/nuclear ``area``, ``layer`` (isocortex only),
a coarse anatomical ``group``, and a ``tissue`` class (grey matter / fiber tract /
unassigned). No network or atlas volume required — the alignment team writes the
acronym into the NWB ``electrodes.location`` column; this module parses it.

The parsing rules encode a few anatomy facts that a naive regex gets wrong:

* Hippocampal ``CA1``/``CA2``/``CA3`` are *subfields*, not layers.
* Dentate gyrus ``DG-mo``/``DG-po``/``DG-sg`` split into area ``DG`` + layer.
* Fiber tracts (``alv``, ``ccb``, ``fi``, ...) and ``root``/``void`` are not
  grey matter and are flagged so they can be excluded from unit analyses.
"""
from __future__ import annotations
import re

# White-matter / fiber-tract acronyms seen in OpenScope PP ecephys sessions.
FIBER_TRACTS = {
    "alv", "ccb", "ccg", "ccs", "cing", "dhc", "fa", "fi", "fp", "or",
    "scwm", "int", "ee", "st", "ar", "SH", "fx", "opt", "em", "cc",
}
UNASSIGNED = {"root", "void", ""}

_LAYER_RE = re.compile(r"^(?P<area>[A-Za-z][A-Za-z\-]*?)(?P<layer>1|2/3|4|5|6a|6b)$")

_THALAMUS = {
    "LGd", "LGd-sh", "LGd-co", "LGd-ip", "LGv", "LP", "LD", "AV", "AD",
    "AMd", "AMv", "AM", "MGd", "MGv", "MGm", "PO", "VPM", "VPL", "VL",
    "VAL", "CL", "RT", "TH", "MD", "IntG", "IGL", "IAD", "PIL", "PF",
    "PoT", "SGN", "Eth", "REth",
}
_STRIATUM = {"CP", "STR", "LSr", "LSc", "ACB", "LSv", "SF"}


def decode_ccf(acronym: str) -> dict:
    """Decode one CCF acronym into ``{area, layer, tissue, group}``.

    Parameters
    ----------
    acronym : str
        CCF acronym from ``electrodes.location`` (e.g. ``"VISp5"``).

    Returns
    -------
    dict with keys ``area`` (str), ``layer`` (str | None),
    ``tissue`` (``"grey"`` | ``"fiber_tract"`` | ``"unassigned"``),
    ``group`` (coarse anatomical group).
    """
    if acronym in FIBER_TRACTS:
        return dict(area=acronym, layer=None, tissue="fiber_tract", group="white_matter")
    if acronym in UNASSIGNED:
        return dict(area=acronym, layer=None, tissue="unassigned", group="unassigned")
    if acronym in {"CA1", "CA2", "CA3"}:
        return dict(area=acronym, layer=None, tissue="grey", group="hippocampus")
    if acronym in {"DG-mo", "DG-po", "DG-sg"}:
        return dict(area="DG", layer=acronym.split("-")[1], tissue="grey", group="hippocampus")
    if acronym in {"SUB", "ProS", "PRE", "POST", "PAR"}:
        return dict(area=acronym, layer=None, tissue="grey", group="hippocampus")

    m = _LAYER_RE.match(acronym)
    area, layer = (m.group("area"), m.group("layer")) if m else (acronym, None)

    if area.startswith("VIS"):
        group = "visual_ctx"
    elif area.startswith(("MOp", "MOs")):
        group = "motor_ctx"
    elif area.startswith(("ACA", "PL", "ILA", "DP", "RSP", "ORB")):
        group = "cingulate/PFC"
    elif area.startswith(("SSp", "SS")):
        group = "somatosensory_ctx"
    elif area in _THALAMUS:
        group = "thalamus"
    elif area in _STRIATUM:
        group = "striatum"
    else:
        group = "other_grey"
    return dict(area=area, layer=layer, tissue="grey", group=group)


def decode_many(acronyms) -> "list[dict]":
    """Vectorised convenience wrapper over :func:`decode_ccf`."""
    return [decode_ccf(a) for a in acronyms]
