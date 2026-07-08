# ai_oscp_neuro

CCF annotation and penetration figures for the **OpenScope Community Predictive
Processing** Neuropixels dataset ([DANDI 001637](https://dandiarchive.org/dandiset/001637)).

> Python package `openscope_ccf` — import name is `openscope_ccf`; the GitHub
> repository is `maierav/ai_oscp_neuro`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maierav/ai_oscp_neuro/blob/main/notebooks/ccf_penetration_figures.ipynb)

This toolkit turns the preliminary Allen CCF alignment packaged into the DANDI
NWB files into two reusable products:

1. **Attachable CCF sidecars** — small per-session Parquet tables (one per unit,
   one per channel) that carry area / layer / coarse group / CCF coordinates,
   keyed to the NWB `units` and `electrodes` row indices. Join them onto any
   SUA / MUA / LFP / CSD analysis with a single `attach()` call.
2. **Penetration figures** — a 3D render of the probe tracks inside a translucent
   Allen brain, and a per-probe laminar cross-check that overlays CCF region/layer
   boundaries on spontaneous LFP band power and MUA depth profiles, so the
   alignment can be validated against the recordings.

## Dataset at a glance

The community project spans **three recording modalities** and **four
predictive-processing paradigms**. The matrix below shows what is available so
far (sessions / mice per cell; CCF = sessions with preliminary Allen CCF
alignment):

![Data available across scales](figures/modality_paradigm_matrix.png)

- **Neuropixels** (DANDI 001637) and **Mesoscope 2p** (001768) carry all four
  paradigms with an identical named-block design (including an open-loop
  prerecorded control block), enabling a cross-modal comparison of the same
  prediction-error contrast across paradigms.
- **SLAP2** (001424) stores its stimuli as a single monolithic `gratings`
  stream rather than named paradigm blocks. It contains an embedded
  **orientation oddball** (a dominant "standard" orientation with rarer
  orientation deviants and omissions) plus orientation-tuning and RF blocks,
  recoverable by segmenting the stream on orientation statistics. This provides
  a standard/feature-oddball contrast at dendritic / glutamate (iGluSnFR)
  resolution; the full four-paradigm SLAP2 set is expected in a later release.
- The **standard / feature oddball** contrast (rare orientation deviant vs.
  frequent standard) is therefore the one comparison expressible in all three
  modalities; the sensorimotor, sequence, and duration paradigms exist in
  Neuropixels and mesoscope only.

## Validation — receptive fields across all three scales

Before running any prediction-error / mismatch analysis, we sanity-checked the
full pipeline (stream NWB → align to stimulus trials → extract response → build
a receptive field) against a known answer: a real visual neuron should respond
to a **compact patch of visual space**. Using each dataset's `RF mapping` block,
we recover clean, retinotopically localized receptive fields in all three
modalities — spikes (Neuropixels), somatic ΔF/F (mesoscope), and dendritic
glutamate ΔF/F (SLAP2):

![Example receptive fields across three recording scales](figures/rf_examples_three_modalities.png)

Each panel is one unit/ROI, selected by **2-D Gaussian fit quality** (R²), with a
diverging colour map centred at zero and a black half-maximum contour; titles give
the fitted RF width (FWHM). The fitted widths are mouse-appropriate — median FWHM
≈ 25° (ecephys), 18° (mesoscope), 15° (SLAP2, best session). We deliberately do
**not** rank by peak/std "SNR" (which is biased toward spiky one-pixel maps) nor
render with `vmin=0` (which crushes the graded surround to black) — both make real
RFs look artificially point-like. Reproduce end-to-end in Colab:
[`notebooks/rf_sanity_check_three_modalities.ipynb`](notebooks/rf_sanity_check_three_modalities.ipynb).
The notebook also documents SLAP2 gotchas — the per-DMD onset offset, a compressed
DMD timebase (rebuilt over the other DMD's simultaneous span), and strong
session-to-session variation in RF yield (pick a good session before judging the
modality).

**Are these RFs real, or just structure we selected for?** Because the examples
are hand-picked (highest Gaussian-fit R²), a selection process could in principle
dress up noise. We test against three noise controls that do not depend on the
selection: split-half reliability (even vs. odd trials), a per-unit trial-label
permutation null, and a negative control (non-visual units for ecephys; responses
re-aligned to random times for the imaging arms).

![RF significance across three modalities](figures/rf_significance_three_modalities.png)

The RFs are stimulus-locked and reproducible: 16–18 % of units/ROIs carry a
significant RF at true stimulus onsets in **Neuropixels and mesoscope**,
collapsing to ~1–2 % (chance) in the controls — and for ecephys the significant
RFs concentrate in visual cortex and visual thalamus while motor cortex and
hippocampus sit at chance. **SLAP2 glutamate** shows a real but weaker population
effect (7 % vs. 5 % control); its best dendritic ROIs are reliable (split-half
r > 0.35, permutation p below the 1/300 resolution floor) but population fractions
should be read cautiously. Note the SLAP2 significance panel was computed on an
earlier, lower-yield session (sub-801381, 41 ROIs); RF yield varies strongly
across sessions, so this 7 % is a conservative lower bound — the best session
(sub-796630, shown in the gallery above) gives ~15 well-formed RFs. Section 5 of
the notebook reproduces all three tests.

## Why this exists

The DANDI NWBs ship a per-channel CCF acronym in `electrodes.location` plus CCF
coordinates in `electrodes.x/y/z`. Two things make these awkward to use directly:

- The acronyms need decoding into area + layer + tissue class (e.g. `VISp5` →
  area `VISp`, layer `5`; `DG-mo` → area `DG`, layer `mo`; `fi` → fiber tract).
- `units.extremum_channel_index` is a **per-probe** index, but the `electrodes`
  table stacks all probes. Indexing directly assigns every unit to the first
  probe — a subtle bug this package fixes in `unit_electrode_rows()`.

## Install

```bash
pip install -e .
```

## Quick start

```python
import openscope_ccf as o

idx = o.load_session_index()          # registry of CCF-labeled sessions
row = idx.iloc[0]
tag = f"{row.subject}_{row.date}"

# 1. Build attachable sidecars for a session
o.build_session_sidecars(row.aid, str(row.subject), row.date, row.paradigm)

# 2. Annotate any unit-level result
annotated = o.attach(my_sua_df, tag, on="unit_index")      # adds area, layer, group, ccf_xyz
# ...or any channel-level (LFP/CSD) result:
annotated = o.attach(my_lfp_df, tag, on="channel", kind="channels")

# 3. Penetration figures
pd_ = o.build_probe_data(row.aid)
o.make_3d(pd_, tag, "fig_3d.png", brain_mesh=o.load_root_mesh())
o.make_laminar(pd_, tag, "fig_laminar.png")
```

## Colab

`notebooks/ccf_penetration_figures.ipynb` runs the whole flow for any session on
free Colab CPU (data is streamed, not downloaded).

## Batch

```bash
python scripts/build_all.py --sidecars --figures --out data/
```

## Layout

```
openscope_ccf/          package
  ccf.py                CCF acronym decoder
  nwbio.py              DANDI streaming + corrected unit→electrode mapping
  sidecar.py            build/load/attach sidecar tables
  figures.py            build_probe_data, make_3d, make_laminar
  data/ccf_session_index.csv   registry of CCF sessions
  data/sidecars/          prebuilt per-session sidecars (Parquet), shipped with the package
notebooks/              Colab notebook
scripts/build_all.py    batch driver
```

`load_ccf`/`attach` resolve sidecars from `./data/sidecars` if present, else fall
back to the copy shipped inside the installed package — so they work from a clone
or a bare `pip install`.

## Data & attribution

- Data: OpenScope Community Predictive Processing, [DANDI 001637](https://dandiarchive.org/dandiset/001637).
- CCF alignment: OpenScope community ([discussion #163](https://github.com/AllenNeuralDynamics/openscope-community-predictive-processing/discussions/163)).
- Atlas: `allen_mouse_25um` via [BrainGlobe](https://brainglobe.info/).

## Notebook hygiene (Colab ↔ GitHub)

Saving a notebook from Colab can insert a `metadata.widgets` block without the
required `state` key, which makes GitHub refuse to render it ("Invalid Notebook").
`scripts/clean_notebook.py` strips only that block — **figures and all other
outputs are preserved** (unlike "Clear all outputs", which deletes them).

Two layers keep this automatic:

* **Local pre-commit hook** — `pip install pre-commit && pre-commit install`, then
  every local `git commit` cleans notebooks first (see `.pre-commit-config.yaml`).
* **GitHub Action** — `.github/workflows/clean-notebooks.yml` cleans and commits
  back on any pushed notebook, which covers saving from Colab straight to GitHub
  (that path bypasses local hooks).

## License

MIT — see [LICENSE](LICENSE).
