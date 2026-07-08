# Analysis plan — standard / feature-oddball prediction error across scales

**Project:** OpenScope Community Predictive Processing (Allen Institute for Neural Dynamics)
**Repo:** maierav/ai_oscp_neuro
**Status:** pre-registration, written before running the confirmatory analysis.
**Background:** Aizenbud et al. 2025, arXiv:2504.09614.

This document commits the hypotheses, contrasts, controls, response windows, and
statistics *before* the confirmatory analysis is run, to avoid the garden-of-
forking-paths bias. Exploratory deviations from it will be labeled as such.

---

## 1. Scientific question (H0 vs H1)

Does the cortex signal a **feature (orientation) oddball** as a genuine
*prediction error*, and is that signal produced by:

- **H0 — distinct/specialized mechanism:** deviance signaling is confined to, or
  qualitatively differs across, specific areas/layers (e.g. a dedicated error
  compartment; superficial vs deep asymmetry), and/or differs by modality/cell
  compartment (soma vs dendrite).
- **H1 — common canonical mechanism:** the same deviance computation appears
  across areas, layers, and scales, differing in magnitude but not in kind.

Operationally we test H1's prediction of a *consistent* deviance signature across
the area x layer x modality grid, versus H0's prediction of *localized or
qualitatively divergent* signatures.

## 2. The central confound — adaptation vs. genuine deviance

A rare deviant almost always fires more than a frequent standard. The trivial
explanation is **stimulus-specific adaptation (SSA)**: the *standard* response is
fatigued by repetition, so the deviant only looks large by contrast. This is not
prediction. We were bitten by exactly this in the GO/LO pilot (the physically
identical global-deviant showed *lower* firing — an adaptation signature).

**The control that dissociates them (verified present in the data):**

| Block | Role | 45°/90° presentation |
|---|---|---|
| `Standard mismatch block` (`standard_oddball`) | standard=0° (freq), deviant=45°,90° (rare, **surprising**) | rare AND violates expectation |
| `Control block 1` (`standard_control`) | 14 orientations **equiprobable**, 68 trials each | rare but **NOT** surprising (many-standards) |

**Genuine deviance = oddball response to X° that EXCEEDS the equiprobable-control
response to the same X°.** This is the classic many-standards (Ulanovsky/Harms)
control. If oddball ≈ control, the "enhancement" is rarity/adaptation, not
prediction error.

**PRE-ANALYSIS VERIFICATION (step 0) — DONE, PASSED (sub-830851):** the 45°/90°
deviant gratings are physically identical between `Standard mismatch block` and
`Control block 1` — same temporal frequency (2 Hz), spatial frequency, diameter
(360°), and median duration (367 ms). The only difference between the two contexts
is the statistical one we want to isolate (rare-and-surprising vs. equiprobable).
The many-standards control is therefore **valid**.
- *Data-cleaning refinement (pre-committed):* 1 of 68 control trials at 45° is a
  truncated 67-ms presentation (0.1 % of all control gratings). Apply a
  `duration >= 0.3 s` filter to control trials, leaving 67 clean control trials at
  45° and 68 at 90°.
- Step 0 will be re-run per session before pooling; any session failing the
  physical-match check is dropped from the confirmatory set.

## 3. Datasets & scope

- **Primary:** Neuropixels ecephys, standard-oddball paradigm — the **9 sessions
  with real CCF alignment** (area + layer available). This is the only arm with
  laminar resolution.
- **Cross-scale replication:** mesoscope (somatic dF/F) and SLAP2 (dendritic
  glutamate) standard-oddball sessions, for the modality axis of H0/H1. SLAP2
  oddball is embedded in the monolithic `gratings` stream (segment by orientation
  statistics; see repo Data-particulars).
- Confirmatory claims are made on ecephys; imaging arms are treated as
  replication/extension given smaller n and different signal.

## 4. Primary contrasts & hypothesis-linked predictions

For each unit/ROI and each deviant orientation d in {45°, 90°}:

- **R_oddball(d)** = response to d as a rare deviant (mismatch block)
- **R_control(d)** = response to d in the equiprobable control block
- **R_standard**  = response to the frequent standard (0°) in the mismatch block

Two indices (both pre-committed):

1. **Deviance index (prediction-error):** `DvI = (R_oddball − R_control)/(|R_oddball| + |R_control|)`
   — the SSA-free measure. **This is the primary outcome.**
2. **Raw oddball index (for comparison/legacy):** `OI = (R_oddball − R_standard)/(|R_oddball| + |R_standard|)`
   — inflated by adaptation; reported alongside to quantify how much of the naive
   effect is SSA.

Predictions:
- **Prediction error exists** iff population median DvI > 0 (bootstrap p<0.01).
- **H1** predicts DvI > 0 with consistent sign across areas and layers, largest
  where hierarchy theory expects (higher visual areas ≥ V1; superficial/deep
  asymmetry consistent across areas).
- **H0** predicts DvI > 0 confined to specific areas/layers, or sign/kind
  differences across the grid.
- **Adaptation-only null:** DvI ≈ 0 while OI > 0 (naive effect fully explained by
  the standard's adaptation).

## 5. Response windows & common cross-scale index

- **Data-driven windows, not ad hoc.** Estimate population onset latency per
  modality from the standard PSTH/dF-F (first bin exceeding baseline + k·SD), then
  fix a single response window per modality for the whole analysis:
  - ecephys: spikes, expected ~[onset, onset+~250 ms] (paper uses ~0–275/300 ms)
  - mesoscope: soma dF/F, slower kinetics → longer window
  - SLAP2: dendritic glutamate, apply the DMD onset offset (+0.115 s) first
- Baseline: pre-stimulus window of matched length.
- The **same DvI formula** is applied to all three, so magnitudes are comparable
  in index space even though raw units differ.

## 6. Analysis unit, gating, and statistics

- **Unit gating:** include only units/ROIs that are *visually responsive* to the
  standard (pre-committed responsiveness test vs baseline, p<0.05), so DvI is
  defined on cells that actually see the stimulus. Report n gated per area/layer.
- **Aggregation:** per-unit DvI → population distributions per (area, layer,
  modality). Pool across the 9 ecephys sessions; include session as a grouping
  factor (mixed-effects or session-stratified bootstrap) to avoid pseudo-
  replication.
- **Significance:** trial-shuffle bootstrap matching the paper (p<0.01); CIs by
  bootstrap over units. **Multiple comparisons:** FDR (Benjamini–Hochberg) across
  the area x layer cells tested.
- **Effect sizes reported, not just p:** median DvI + bootstrap CI per cell.

## 7. Behavioral state

Locomotion strongly modulates mouse V1 and could masquerade as deviance. **Check
running-speed (and pupil, if present) availability per session; verify running is
balanced between oddball and control blocks; if imbalanced, condition on a
stationary subset or include speed as a covariate.** Pre-committed: report the
speed balance; do not silently pool across states.

## 8. Layer & area resolution

Attach CCF area+layer via the repo's sidecars (`attach(..., on="unit_index")`).
Restrict laminar claims to isocortical units with a recovered layer. Report the
area x layer coverage matrix (n responsive units per cell) before interpreting —
sparse cells (<~15 units) are descriptive only.

## 9. Power reality-check

Only ~35 deviant trials per orientation per session. Per-unit DvI is noisy; the
confirmatory claim is at the **population** level with cross-session pooling.
Pre-committed: if a given area x layer cell has <~15 responsive units pooled, it
is reported descriptively (no inferential claim).

## 10. What would falsify each hypothesis

- **No prediction error:** population DvI CI includes 0 while OI>0 → the paradigm
  elicits adaptation, not deviance detection (at least at this scale/window).
- **Against H1 (favoring H0):** DvI>0 in some areas/layers but sign-flips or
  vanishes in others beyond what n explains; or modalities disagree in kind.
- **Against H0 (favoring H1):** DvI>0 with consistent sign and ordered magnitude
  across the whole grid.

## 11. Deliverables

- `deviance_index_by_area_layer.csv` — per (area, layer, modality) median DvI, CI, n
- a figure: DvI vs OI (how much is adaptation), and the area x layer DvI grid
- one Colab notebook reproducing the ecephys confirmatory analysis end to end
- a short results paragraph explicitly stating which hypothesis the data support

---

*Deviations from this plan in the executed analysis will be marked EXPLORATORY.*
