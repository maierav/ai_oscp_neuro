# ai_oscp_neuro

Analysis toolkit and cross-scale validation for the **OpenScope Community
Predictive Processing** dataset — Allen Institute for Neural Dynamics.

> Python package `openscope_ccf` — import name is `openscope_ccf`; the GitHub
> repository is `maierav/ai_oscp_neuro`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maierav/ai_oscp_neuro/blob/main/notebooks/ccf_penetration_figures.ipynb)

## Background & motivation

The [OpenScope Community Predictive Processing](https://allenneuraldynamics.github.io/openscope-community-predictive-processing/)
project asks how the brain implements **predictive processing** — whether the
cortex learns to predict upcoming sensory input and signals *prediction errors*
when reality violates expectation. The central scientific question is whether
different kinds of violation (a sensory oddball, a broken sensorimotor
contingency, an omitted stimulus) are computed by **distinct specialized
circuits** or by a **common canonical mechanism** repeated across the brain. To
test this, the same battery of "mismatch" paradigms is recorded at **three
spatial scales in mouse visual cortex**, so that error signals can be compared
from single spikes up to population and dendritic activity.

The conceptual and methodological background — the predictive-processing
framework, the paradigm design, and the experimental program — is laid out in the
community white paper:

> Aizenbud et al. (2025), *Neural mechanisms of predictive processing: a
> collaborative community experiment through the OpenScope program.*
> [arXiv:2504.09614](https://arxiv.org/abs/2504.09614)

**What this repository does.** It is a practical analysis layer on top of the
public data: it makes the preliminary anatomical (CCF) alignment usable, provides
attachable annotations for downstream spike/LFP/imaging analyses, and establishes
a set of **confidence-building validations** (receptive fields and direction
tuning across all three modalities) that must pass before the harder
prediction-error analyses are trusted. It is deliberately kept separate from any
one analysis so the products are reusable.

## What's in this repository

The core toolkit turns the preliminary Allen CCF alignment packaged into the DANDI
NWB files into two reusable products:

1. **Attachable CCF sidecars** — small per-session Parquet tables (one per unit,
   one per channel) that carry area / layer / coarse group / CCF coordinates,
   keyed to the NWB `units` and `electrodes` row indices. Join them onto any
   SUA / MUA / LFP / CSD analysis with a single `attach()` call.
2. **Penetration figures** — a 3D render of the probe tracks inside a translucent
   Allen brain, and a per-probe laminar cross-check that overlays CCF region/layer
   boundaries on spontaneous LFP band power and MUA depth profiles, so the
   alignment can be validated against the recordings.

Alongside these are the **cross-scale validation notebooks** (receptive fields,
direction tuning) documented below, which double as worked examples of streaming
and analyzing each modality.

The first prediction-error analysis (the feature/orientation oddball) is
**pre-registered** before any confirmatory result:
[`docs/oddball_analysis_plan.md`](docs/oddball_analysis_plan.md). It commits the
H0/H1 hypotheses, the adaptation-vs-deviance control (a verified equal-probability
"many-standards" block), response windows, and statistics up front — the key
design decisions, fixed before the p-values exist.

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

## Data particulars & gotchas (read before analyzing)

These are non-obvious properties of the data that cost real debugging time. If
you are an analyst — human or LLM — picking this up cold, read this first; several
of them silently produce wrong-but-plausible results.

**Access.** The clean HDF5 NWB files live on DANDI (001637 / 001768 / 001424); the
same data is also on `s3://aind-open-data` as `.nwb.zarr`. We stream the DANDI
HDF5 over HTTP (`remfile` + `h5py`) rather than downloading — see `nwbio.py`. No
DANDI credentials are needed for these public dandisets.

**Stimulus blocks (Neuropixels & mesoscope).** Stimuli are organized into named
`intervals` blocks, and the useful ones are easy to misread:
- `Control block 1` (`standard_control`) — the **14-direction drifting-grating
  sweep** used for tuning (below). *Not* a mismatch block.
- `Control block 4` (`open_loop_prerecorded`) — the **open-loop comparator** for
  the sensorimotor paradigm (the paper-prescribed control: same stimuli, motor
  coupling removed).
- `Standard mismatch block` — the actual oddball block; `Orientation`,
  `TrialType`, `contrast` columns define standard vs. deviant vs. omission.
- Column names are **capitalized** (`Orientation`, `TemporalFrequency`,
  `SpatialFrequency`), and orientations are in **radians**, not degrees.

**Direction ≠ orientation.** The 14-condition sweep is **drifting** gratings
(`TemporalFrequency = 2 Hz`), which measure **direction** tuning over 0–360°.
There is **no static-grating orientation sweep** in this dataset (the `TF = 0`
trials in those blocks are a single orientation). Report DSI as measured; an
orientation index (OSI) is only obtainable by *folding* the curve 360°→180°.
Calling the drifting-grating result "orientation tuning" is the mistake to avoid.

**Ecephys electrode/unit mapping.** `units.extremum_channel_index` is a
**per-probe** index (0–~382), but the `electrodes` table **stacks all probes**.
Indexing `electrodes` directly with it assigns every unit to the first probe.
Correct mapping (add the per-probe row offset) lives in
`nwbio.unit_electrode_rows()`. Also: `units.device_name` is a per-session device
identifier, **not** an anatomical label — get area/layer from CCF, not from the
device name. Real CCF alignment is present in **30 of 48** usable ecephys sessions
so far (`electrodes.location` / `x` / `y` / `z` populated); the rest carry
placeholder `"unknown"` locations.

**SLAP2 is structurally different.** All stimuli sit in one monolithic
`intervals/gratings` table (no named blocks); segment it by stimulus statistics.
The oddball is *embedded* (a dominant standard orientation with rarer deviants +
contrast-0 omissions). Three imaging quirks, all load-bearing:
- **Two DMD paths** (`Fluorescence_DMD1` / `Fluorescence_DMD2`) image
  **simultaneously** but with a small fixed onset offset (DMD1 ≈ +0.115 s).
- A DMD's **stored timestamps can be compressed** (e.g. labeled over ~1000 s when
  the recording is ~3020 s). Because the two DMDs are simultaneous, rebuild the
  bad timebase as a uniform axis over the *other* DMD's intact span.
- **RF/tuning yield varies strongly across sessions.** Pick a good session before
  judging the modality (we use sub-796630 2025-10-01 DMD1). One early SLAP2 format
  (sub-794237) differs and is skipped.

**CCF acronyms encode area *and* layer.** `electrodes.location` gives e.g. `VISp5`
(area `VISp`, layer `5`), `DG-mo` (area `DG`, layer `mo`), `CA1` (a hippocampal
subfield, *not* a layer), or a fiber-tract code like `fi`. `ccf.py` decodes these;
`electrodes.x/y/z` are absolute CCF µm, while `units.estimated_x/y/z` are
probe-local relative coordinates.

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

## Validation — direction tuning across all three scales

A second known-answer check: direction/orientation selectivity is the canonical
property of mouse visual cortex. Each session's `Control block 1`
(`standard_control`) carries a full **14-direction drifting-grating sweep** (0–315°
in 22.5° steps, TF = 2 Hz); in SLAP2 the same directions live in the full-field
gratings.

**Direction vs. orientation.** The sweep is presented as *drifting* gratings, which
natively measure **direction tuning** (0–360°). True *orientation* tuning
(indifferent to drift direction) needs *static* gratings, which are not available
as a sweep here. We therefore report **DSI** as the primary metric and **OSI** only
as a value **derived** by folding the tuning curve 360°→180° (standard 2θ vector
method) — not as a static-grating measurement.

![Direction tuning across three recording scales](figures/direction_tuning_three_modalities.png)

| Modality | median DSI | OSI (derived) | tuning HWHM | % direction-selective |
|---|---|---|---|---|
| Neuropixels (spikes) | 0.18 | 0.39 | 28° | 11 % |
| Mesoscope (ΔF/F soma) | 0.42 | 0.64 | 16° | 53 % |
| SLAP2 (glutamate) | 0.31 | 0.49 | 21° | 24 % |

All three show strong, well-formed tuning with realistic half-widths (16–28° HWHM).
Most cells are orientation-selective (two roughly equal lobes); a direction-selective
minority (one dominant lobe) is largest in mesoscope. Two method choices keep this
honest and mirror the RF check: examples are selected by **von Mises fit quality**
(not by a selectivity index, which over-selects spiky near-line curves), and we
report the fitted **tuning width** so narrow curves are shown as narrow rather than
driving the selection. RF mapping validates **spatial** sensitivity; direction
tuning validates **feature** sensitivity — together the pipeline reads real visual
signals at all three scales. Reproduce in Colab:
[`notebooks/direction_tuning_three_modalities.ipynb`](notebooks/direction_tuning_three_modalities.ipynb).

## First analysis — feature-oddball prediction error (Neuropixels)

With the pipeline validated, the first prediction-error result asks whether a rare
*oddball* orientation drives a genuine **prediction error** or merely reflects
**stimulus-specific adaptation** of the frequent standard. The distinction is the
whole game: a rare deviant almost always fires more than a repeated standard simply
because the standard is fatigued. To rule that out we compare the deviant not only
against the standard but against the **same grating shown equiprobably** in the
`standard_control` block — physically identical, equally rare, but *not* surprising.
The **deviance index** `DvI = (oddball − equiprobable control)/…` is therefore
adaptation-free; the naive `OI = (oddball − standard)/…` is shown alongside.

![Feature-oddball prediction error, example Neuropixels session](figures/oddball_ecephys_single_session.png)

One example session (sub-830851, 141 responsive visual units):
- **A** — the 90° oddball response exceeds both the *identical* equiprobable
  control and the frequent standard.
- **B** — DvI vs OI per unit; the deviance signal is present in the adaptation-free
  index, not just the naive one.
- **C** — deviance scales with feature distance: the orthogonal 90° deviant is
  strongly significant (median DvI ≈ +0.34, p ≈ 9×10⁻⁹), the intermediate 45° is
  not.
- **D** — the 90° deviance is positive and consistent across visual areas and
  cortical layers — a canonical rather than compartment-specific signature.

**This is one example session, not the confirmatory claim.** The full analysis
pools the 9 Neuropixels sessions with Allen CCF alignment, with cross-session
bootstrap CIs and FDR correction across the area × layer grid, exactly as
[pre-registered](docs/oddball_analysis_plan.md). Reproduce the example in Colab:
[`notebooks/oddball_prediction_error_ecephys.ipynb`](notebooks/oddball_prediction_error_ecephys.ipynb).

## Why a CCF package

The raw NWB CCF fields are awkward to use directly for the two reasons detailed in
*Data particulars* above — the acronyms need decoding into area + layer + tissue
class, and the per-probe `extremum_channel_index` must be offset before indexing
the stacked `electrodes` table. This package handles both once (`ccf.py`,
`nwbio.unit_electrode_rows()`) and ships the results as attachable sidecars, so
every downstream analysis inherits correct anatomy without re-solving them.

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
