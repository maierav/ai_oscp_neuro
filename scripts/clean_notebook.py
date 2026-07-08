#!/usr/bin/env python
"""Strip the Colab ``metadata.widgets`` block that breaks GitHub notebook rendering.

Colab writes ``metadata.widgets['application/vnd.jupyter.widget-state+json']``
without the required ``state`` key when a notebook runs a widget (e.g. a tqdm
progress bar or a pip cell). GitHub's nbviewer then refuses the whole notebook
with "the 'state' key is missing from 'metadata.widgets'".

This script removes only that widgets block (notebook-level and, defensively,
per-cell). **Cell outputs — including figure images — are left untouched.** It is
the safe alternative to "Clear all outputs", which would delete the figures.

Usage
-----
    python scripts/clean_notebook.py [notebook.ipynb ...]

With no arguments, cleans every ``*.ipynb`` under ``notebooks/``. Exit code is 0
whether or not anything changed; prints which files it modified. Designed to be
idempotent and safe to run from a git pre-commit hook.
"""
import json
import sys
from pathlib import Path


def clean_notebook(path: Path) -> bool:
    """Remove widgets metadata from one notebook. Returns True if it changed."""
    nb = json.loads(path.read_text())
    changed = False
    if "widgets" in nb.get("metadata", {}):
        nb["metadata"].pop("widgets")
        changed = True
    for cell in nb.get("cells", []):
        if "widgets" in cell.get("metadata", {}):
            cell["metadata"].pop("widgets")
            changed = True
    if changed:
        # preserve nbformat's conventional 1-space indent + trailing newline
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return changed


def main(argv):
    if argv:
        paths = [Path(p) for p in argv]
    else:
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "notebooks").glob("*.ipynb"))
    any_changed = False
    for p in paths:
        if not p.exists():
            print(f"skip (not found): {p}")
            continue
        if clean_notebook(p):
            print(f"cleaned metadata.widgets: {p}")
            any_changed = True
        else:
            print(f"already clean: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
