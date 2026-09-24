# oscNext L4 — code guide

This guide explains the code in this repository: what it is made of, how the
pieces fit together, and what every function does.  It is written for someone
who knows IceCube and IceTray in broad terms but has not seen this code before.

- **Part 0** gives the big picture: what the pipeline does, how
  data flows through it, where each file sits, and step-by-step recipes for
  the common tasks.
- **Part A** covers the tray side: the modules that run inside IceTray and
  compute the L4 variables, plus a glossary of tray terms.
- **Part B** covers running the processing: `process_L4.py`, the runners, and
  the environment scripts.
- **Part C** covers the analysis side: from booked HDF5 files to training
  sets, trained models and their checks, plus the two config files.
- **Part D** covers the interfaces and checks: the notebook, the pass2
  cross-check, the release builder, and the presentation scripts.

Parts A-D go file by file, and within each file function by function.  To
find one function, search for its name.

## Contents

- [Part 0 — the big picture](#part-0--the-big-picture)
  - [1. What the pipeline does](#1-what-the-pipeline-does)
  - [2. The big picture: how data flows](#2-the-big-picture-how-data-flows)
  - [3. Repository map](#3-repository-map)
  - [4. The ideas the code is built on](#4-the-ideas-the-code-is-built-on)
  - [5. Recipes](#5-recipes)
  - [6. The guards: what stops a silent error](#6-the-guards-what-stops-a-silent-error)
- [Part A: the tray side](#part-a-the-tray-side)
  - [How the pieces fit (the order modules run in)](#how-the-pieces-fit-the-order-modules-run-in)
  - [`oscnext_l4/__init__.py`](#oscnext_l4__init__py)
  - [`oscnext_l4/variables.py`](#oscnext_l4variablespy)
  - [`oscnext_l4/rewritten.py`](#oscnext_l4rewrittenpy)
  - [`oscnext_l4/frame_objects.py`](#oscnext_l4frame_objectspy)
  - [`oscnext_l4/l3vars.py`](#oscnext_l4l3varspy)
  - [`oscnext_l4/tray_io.py`](#oscnext_l4tray_iopy)
  - [`oscnext_l4/classifier.py`](#oscnext_l4classifierpy)
  - [Glossary of tray terms](#glossary-of-tray-terms)
- [Part B: Running the processing (entry points, drivers and environment)](#part-b-running-the-processing-entry-points-drivers-and-environment)
  - [`scripts/process_L4.py`](#scriptsprocess_l4py)
  - [`oscnext_l4/runner.py`](#oscnext_l4runnerpy)
  - [`scripts/scan_files.py`](#scriptsscan_filespy)
  - [`scripts/diagnose_env.py`](#scriptsdiagnose_envpy)
  - [`scripts/collect_meta.sh`](#scriptscollect_metash)
  - [`setup_env.sh`](#setup_envsh)
- [Part C — the analysis side: booked HDF5 → training sets → models → checks](#part-c--the-analysis-side-booked-hdf5--training-sets--models--checks)
  - [1. The two config files](#1-the-two-config-files)
  - [2. The HDF5 layout the code expects](#2-the-hdf5-layout-the-code-expects)
  - [3. The methods, in prose](#3-the-methods-in-prose)
  - [4. File by file](#4-file-by-file)
  - [`oscnext_l4/varmap.py`](#oscnext_l4varmappy)
  - [`oscnext_l4/data.py`](#oscnext_l4datapy)
  - [`oscnext_l4/dataset.py`](#oscnext_l4datasetpy)
  - [`scripts/make_dataset.py`](#scriptsmake_datasetpy)
  - [`scripts/train_L4_classifier.py`](#scriptstrain_l4_classifierpy)
  - [`scripts/check_application.py`](#scriptscheck_applicationpy)
  - [`scripts/check_leakage.py`](#scriptscheck_leakagepy)
  - [`scripts/plot_inputs.py`](#scriptsplot_inputspy)
  - [`scripts/inspect_production_table.py`](#scriptsinspect_production_tablepy)
- [Part D — Interfaces and checks](#part-d--interfaces-and-checks)
  - [notebooks/oscNext_L4.ipynb](#notebooksoscnext_l4ipynb)
  - [verification/](#verification)
  - [verification/\_\_init\_\_.py](#verification__init__py)
  - [verification/pass2.py](#verificationpass2py)
  - [verification/compare_pass2.py](#verificationcompare_pass2py)
  - [verification/pass2_verification.ipynb](#verificationpass2_verificationipynb)
  - [verification/README.md](#verificationreadmemd)
  - [scripts/make_release.py](#scriptsmake_releasepy)
  - [presentation/make_figures.py](#presentationmake_figurespy)
  - [presentation/make_update_pdf.py](#presentationmake_update_pdfpy)

---

# Part 0 — the big picture

## 1. What the pipeline does

The oscNext low-energy event selection goes through several levels.  At
**Level 3 (L3)** the sample is still dominated by two kinds of background:

- **pure noise**: vuvuzela, i.e. random dark-noise hits that happened to
  trigger;
- **atmospheric muons**: muons from cosmic-ray air showers that enter from
  above.

**Level 4 (L4)** removes most of both with two boosted decision trees (BDTs),
trained with LightGBM:

| classifier | separates neutrinos from | inputs | output frame key | cut |
|---|---|---|---|---|
| **noise BDT** | vuvuzela noise | 5 variables | `L4_NoiseClassifier_ProbNu` | P ≥ 0.70 |
| **muon BDT** | atmospheric muons | 10 variables | `L4_MuonClassifier_Data_ProbNu` | P ≥ 0.65 |

An event passes L4 when it passes **both** cuts.  The result is written as
`L4_oscNext_bool`.

This repository rebuilds that step for **IceTray v1.17.0 on cvmfs**.  That
IceTray does not contain the `oscNext` project, nor three modules the original
L4 script called, so those three were rewritten in pure Python and checked
event by event against the real pass2 production.  The reference for the
method is the oscNext technical note v00.07 (sections 3.4-3.6, Tables 10-13).
The hyperparameters, input lists and cut values are the production's own.

### The 15 BDT inputs, and where each one comes from

| variable | noise | muon | where it comes from |
|---|:-:|:-:|---|
| `NchCleaned` | ✓ | ✓ | ready-made in L3 (`IC2018_LE_L3_Vars`) |
| `micro_count` | ✓ | | computed at L4 by a chain of IceTray modules (`variables.py`) |
| `iLineFit_speed` | ✓ | | IceTray `linefit.simple` (improved LineFit) |
| `fill_ratio` | ✓ | | IceTray `I3FillRatioModule` |
| `FullTimeLengthRatio` | ✓ | | computed at L4 from two L3 numbers (`l3vars.py`) |
| `ICVetoHits` | | ✓ | ready-made in L3 |
| `RTVeto250Hits` | | ✓ | ready-made in L3 |
| `NAbove200Hits` | | ✓ | ready-made in L3 |
| `VICH_nch` | | ✓ | **rewritten** (`rewritten.py`, `vich`): stands in for `tau_bdt.I3CutL7Module` |
| `accumulated_time` | | ✓ | **rewritten** (`rewritten.py`, `dunkman`): stands in for `analysis.CalculateVariables` |
| `first_hlc_rho` | | ✓ | **rewritten** (`rewritten.py`, `first_hlc`): stands in for `SimpleVertex FirstHLC` |
| `cog_z`, `z_sigma`, `z_travel` | | ✓ | IceTray `common_variables.hit_statistics` |

The rewritten variables were compared with the real pass2 L4 files: 14 of 15
compared rows are **bitwise identical** over 56,301 events.  The exception is
`accumulated_time` at 99.84%, whose residual comes from the order in which
pulses with the same time are sorted (`CLAUDE.md`, open risk 2).

---

## 2. The big picture: how data flows

```
   L3 .i3.zst files (pass2 or pass3)                         GCD file
          │                                                      │
          ▼                                                      ▼
 ┌──────────────────────────── scripts/process_L4.py ─────────────────────────┐
 │  I3Reader → stream filter (InIceSplit) → oscNext_L4 segment (variables.py) │
 │     L3 cut → pulse-series check → common vars → hit statistics             │
 │     → noise vars → muon vars → [classifiers + L4 cut, with --apply-cut]    │
 │  → I3Writer (--output-i3)      → hdfwriter booker (--output-hdf5)          │
 └────────────────────────────────────────────────────────────────────────────┘
          │ L4 .i3 (for analysis)             │ HDF5 tables + .meta.json
          ▼                                   ▼
   ready L4 sample              <HDF>/<sample>/L4_<sample>*.hdf5
   (with --apply-cut)                         │
                                              ▼
              scripts/make_dataset.py  (or notebook sections 4-6)
                load_sample → add_weights → split → build_dataset
                                              │
                                              ▼
                             L4_noise_dataset.npz, L4_muon_dataset.npz
                                              │
                                              ▼
              scripts/train_L4_classifier.py  (or notebook section 7)
                                              │
                                              ▼
            L4_<tag>_model.txt (LightGBM trees) + L4_<tag>_model.json (sidecar)
                                              │
                                              ▼
       back into the tray: process_L4.py --apply-cut --model-dir <dir>
       (oscnext_l4/classifier.py: L4Classifier, add_L4_classifiers)
```

The order matters in one place.  The **noise** model is trained first.  The
**muon** training set is then built only from events that pass that noise
model (P ≥ 0.70), which is how the production did it.

### Two worlds that never import each other

The library `oscnext_l4/` is split by **what a module needs to import**, not
by topic:

| side | modules | needs |
|---|---|---|
| **tray side** | `variables`, `rewritten`, `frame_objects`, `l3vars`, `tray_io`, `classifier` | IceTray (`from icecube import ...`) |
| **analysis side** | `data`, `dataset` | numpy, pytables (lightgbm for the noise cut) |
| **both** | `varmap` | the standard library only |
| **driver** | `runner` | the standard library; runs `process_L4.py` as a subprocess |

The two sides meet in exactly two places:

1. **The HDF5 file** carries the data from the tray to training.
2. **`config/variables.json`** carries the naming.  Each variable is one row
   that holds both the HDF5 column that training reads and the frame field
   that the tray application reads.  `varmap.py` turns that one table into the
   dictionaries both sides use.

This is why the analysis scripts (`make_dataset.py`, `train_L4_classifier.py`,
`check_application.py`) run without IceTray.

### The life of one event

1. **In the L3 file**, a Physics (P) frame carries the pulse series
   (`SplitInIcePulses`, and the cleaned `SRTTWSplitInIcePulsesDC` at pass3 or
   `SRTTWOfflinePulsesDC` at pass2), the L3 variable map `IC2018_LE_L3_Vars`,
   and the L3 cut flags.
2. **`process_L4.py`** reads the frame.  Frames outside the `InIceSplit`
   sub-event stream are dropped, and so are frames that fail the L3 cut.
3. **The `oscNext_L4` segment** writes the `L4_*` keys into the frame:
   `L4_first_hlc`, `L4_first_hlc_rho`, `L4_iLineFit(Params)`, `L4_VICH_*`,
   `L4_accumulated_time`, `L4_micro_count`, `L4_fill_ratio`,
   `L4_FullTimeLengthRatio`, the hit statistics, and
   `L4_NoiseStraightCuts_Bool`.
4. **The booker** (hdfwriter) turns every requested key into its own HDF5
   table, one row per event.  It also writes an index,
   `/__I3Index__/<key>`, which says which row belongs to which frame.
5. **`load_sample`** (`data.py`) reads the columns listed in
   `variables.json`.  It lines them up event by event through that index and
   joins the parts of one sample.
6. **`add_weights`** gives every event a physical rate weight, `w_phys` in
   Hz, using the formula for its sample's weighting scheme (GENIE, noise,
   CORSIKA, MuonGun, or detector data).
7. **`make_dataset.py`** stacks the signal and background samples.  It draws
   the train/test split, balances the two classes, and writes a `.npz`.
8. **`train_L4_classifier.py`** trains LightGBM.  It writes the model as
   LightGBM's own text format plus a JSON sidecar, which records the feature
   order, the training numbers and where the data came from.
9. **Back in the tray** (`--apply-cut`), `L4Classifier` reads the same
   quantities from the frame, calls the model, and writes the probability.
   `add_L4_classifiers` then writes `L4_oscNext_bool`.

---

## 3. Repository map

```
oscnext_l4/                   the library
  variables.py                the tray segments: oscNext_L4 and its sub-segments      [A]
  rewritten.py                first_hlc, dunkman (accumulated_time, separation), vich [A]
  frame_objects.py            calc_rho_36, get_pulses, iter_map, deepcore_doms,
                              PropagateGenieInfo                                      [A]
  l3vars.py                   FullTimeLengthRatio                                     [A]
  tray_io.py                  validate_files (input check), add_booker (HDF5 out)     [A]
  classifier.py               L4Classifier (applies a model), add_L4_classifiers      [A]
  runner.py                   runs process_L4.py: serial, parallel, per run; bars     [B]
  varmap.py                   variables.json -> the dicts both sides use              [C]
  data.py                     HDF5 loading, weights, livetime                         [C]
  dataset.py                  stacking, splitting, writing the training .npz          [C]
config/
  variables.json              one row per variable: HDF5 side + frame side            [C]
  productions.json            pass2 / pass3: samples, their roles and weighting       [C]
  README.md                   why every value is what it is
scripts/
  process_L4.py               L3 .i3 -> L4 .i3 and/or HDF5                            [B]
  scan_files.py               find corrupt .i3.zst files                              [B]
  diagnose_env.py             what the IceTray environment has                        [B]
  collect_meta.sh             copy the production's source for reading                [B]
  make_dataset.py             HDF5 -> training .npz (no notebook needed)              [C]
  train_L4_classifier.py      .npz -> model .txt + .json                              [C]
  check_application.py        do the tray's scores equal training's?                  [C]
  check_leakage.py            train/test overlap check                                [C]
  plot_inputs.py              input distributions                                     [C]
  inspect_production_table.py look inside a production HDF5 table                    [C]
  make_release.py             the runnable package for a collaborator                 [D]
setup_env.sh                  find and enter the IceTray environment                  [B]
notebooks/oscNext_L4.ipynb    the interactive interface, sections 0-10                [D]
verification/                 the pass2 cross-check (NOT part of the pipeline)        [D]
release/                      templates make_release.py fills                         [D]
presentation/                 slides, weekly updates, figure scripts                  [D]
docs/                         pipeline.md, technical_note_comparison.md,
                              production_build.md, this guide
reference/                    the technical note and first-hand source material
                              (kept exactly as received; never edited)
icetray-oscNext/              a copy of the official oscNext project, for reading
CLAUDE.md                     the project's full record: decisions, measurements,
                              open risks
```

---

## 4. The ideas the code is built on

These explain why the code looks the way it does.  Each one exists because
the failure it prevents gives **no error**: without it, the pipeline would
produce wrong numbers silently.

1. **One row per variable** (`config/variables.json`).  Training reads a
   column from an HDF5 file, and the tray reads a field from a frame object.
   The two are named differently (hdfwriter renames `fill_ratio_from_mean` to
   `fillratio_from_mean`).  If the two sides ever named different quantities,
   the model would score events wrongly without raising.  With both names in
   one row, they cannot be edited apart, and `data.check_feature_map()`
   checks that they still agree.

2. **The feature order is the model's.**  The order of `bdt_features` in
   `variables.json` is the order in which the model expects its inputs.  The
   sidecar JSON records it, and the tray reads the inputs in the booster's own
   order.  A mismatch raises.

3. **Samples are described by role, not by name.**  Every sample in
   `productions.json` has a `kind` (`signal`, `noise_bg`, `muon_bg`) and a
   `weight` scheme (`genie`, `noise`, `corsika`, `muongun`, `data`).  All code
   keys on those two fields.  That is how one code base serves pass3 (muon
   background: `corsika`) and pass2 (muon background: `muongun` and `data`).

4. **Match tables to events through the index.**  `(Run, Event, SubEvent)`
   repeats across the L3 files that one HDF5 part holds.  So tables are lined
   up through hdfwriter's `/__I3Index__/<key>`, never by those IDs.

5. **Divide weights by the true number of L3 files.**  Every output has a
   `.meta.json` recording how many L3 files went into it (and which ones).
   The weights are divided by that count.  A training set is refused when a
   part lacks it.

6. **One shared signal split.**  The final L4 cut is the AND of both
   classifiers, so both must be tested on the same held-out signal events.
   The split is drawn from a fixed seed over the same stacked signal.  The
   muon stage checks, through a hash of the signal event IDs, that it sees
   exactly the events the noise model was trained with.

7. **Where the production and the technical note disagree, the default follows
   the production.**  Two variables are affected: `micro_count` starts from
   the uncleaned pulses, and `accumulated_time` uses the charge-quartile rule.
   The note's reading of each is one flag away (`--micro-count-cleaned`,
   `--accumulated-time-note`).

---

## 5. Recipes

All commands run inside the IceTray environment unless marked "no IceTray".
See Part B for `setup_env.sh`.

```bash
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)
/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/RHEL_9_x86_64_v2/metaprojects/icetray/v1.17.0/env-shell.sh
python scripts/diagnose_env.py            # what is and is not available
```

**A smoke test on one file** (`--n` counts frames, not events):

```bash
python scripts/process_L4.py --gcd <GCD> --input <one L3 file> \
    --output-hdf5 /tmp/smoke.hdf5 --mc --genie --n 200 --scan off
```

**Book a whole sample** (pass3 GENIE νe; each sample's flags are in
`config/productions.json`; for pass2 add
`--cleaned-pulses SRTTWOfflinePulsesDC`):

```bash
python scripts/scan_files.py "/data/ana/.../23800/*.i3.zst" --good-list nue_good.txt
python scripts/process_L4.py --gcd <GCD> --input-list nue_good.txt --scan off \
    --output-hdf5 <HDF>/nue/L4_nue.hdf5 --chunk-files 10 --mc --genie
```

From the notebook, `run_all(jobs=8, chunk_files=10)` does the same for every
sample in parallel (Part B, `runner.py`).

**Build the training sets and train** (no IceTray):

```bash
python scripts/make_dataset.py --stage noise --production pass2 --hdf-base <HDF> --outdir <DS>
python scripts/train_L4_classifier.py --tag noise --dataset <DS>/L4_noise_dataset.npz --outdir my_models
python scripts/make_dataset.py --stage muon --production pass2 --hdf-base <HDF> --outdir <DS> \
    --noise-model my_models/L4_noise_model.txt
python scripts/train_L4_classifier.py --tag muon --dataset <DS>/L4_muon_dataset.npz --outdir my_models
```

**Apply the models: L3 .i3 → L4 .i3 with scores and `L4_oscNext_bool`:**

```bash
python scripts/process_L4.py --gcd <GCD> --input <L3 files> \
    --output-i3 L4_out.i3.zst --apply-cut --model-dir my_models --mc --genie
```

**Check that the tray scores match training** (do this once, on one file):

```bash
python scripts/process_L4.py --gcd <GCD> --input <one L3 file> \
    --output-hdf5 check.hdf5 --apply-cut --model-dir my_models --mc --genie
python scripts/check_application.py check.hdf5 --model-dir my_models     # must say PASS
```

**Build the package for a collaborator** (no IceTray):

```bash
python scripts/make_release.py --out dist/oscnext_l4 --models my_models
```

---

## 6. The guards: what stops a silent error

| what could go wrong silently | what stops it | where |
|---|---|---|
| the HDF5 side and the frame side name different quantities | one row per variable; `check_feature_map()` | `varmap.py`, `data.py` |
| inputs fed to the model in the wrong order | the booster's order is used; a mismatch with the sidecar raises | `classifier.py`, `dataset.py` |
| a model input the tray never finds | missing in all of the first 100 frames → raise | `classifier.py` |
| pass2 run without `--cleaned-pulses` | a pulse series missing from the first 20 frames → raise | `variables.py` |
| one corrupt L3 file kills the whole run | pre-scan + retry with a blacklist | `process_L4.py`, `tray_io.py` |
| a resumed chunked run books files twice or not at all | each part records its L3 files; a mismatch or a stale part is refused | `process_L4.py` |
| the same L3 file loaded from two parts | `_refuse_double_booking` | `data.py` |
| weights divided by a guessed file count | `.meta.json` required for every part | `data.py`, `make_dataset.py` |
| noise weight in the wrong unit (1e9 off) | the unit is declared per production | `productions.json`, `data.py` |
| noise and muon sets with different signal splits | signal-identity hash compared | `make_dataset.py` |
| CORSIKA shower copies on both sides of the split | split by (HDF5 part, Run) | `dataset.py` |
| an input missing in every event of a class | refused | `make_dataset.py` |
| a shipped muon model paired with the wrong noise model | sha256 check | `make_release.py` |
| the tray's scores differ from training's | the end-to-end comparison | `check_application.py` |


---

# Part A: the tray side

This part covers the code that runs **inside IceTray**. It reads L3 frames,
computes the L4 variables, writes them back into the frame, and (optionally)
applies the trained classifiers. Seven files:

| file | in one line |
|---|---|
| `oscnext_l4/__init__.py` | Package marker. Lists the modules. |
| `oscnext_l4/variables.py` | The tray segments: the L4 processing chain itself. |
| `oscnext_l4/rewritten.py` | Pure-Python replacements for three IceTray modules this environment lacks (first HLC, Dunkman variables, VICH). |
| `oscnext_l4/frame_objects.py` | Small frame helpers: rho_36, DeepCore DOM lists, pulse-map access, the GENIE-info propagator. |
| `oscnext_l4/l3vars.py` | `FullTimeLengthRatio`, an L3 variable that pass3's L3 does not store. |
| `oscnext_l4/tray_io.py` | The tray's two file ends: the input file check and the HDF5 writer. |
| `oscnext_l4/classifier.py` | Applies the trained LightGBM models to frames and writes the L4 cut. |

A glossary of the IceTray terms used below is at the end of this part.

## How the pieces fit (the order modules run in)

`scripts/process_L4.py` builds one tray per run: `I3Reader` -> a sub-event
stream filter -> the `oscNext_L4` segment -> optional `I3Writer` -> the HDF5
booker (`tray_io.add_booker`). Inside `oscNext_L4`, the modules run in this
order. "cleaned" and "uncleaned" are the two pulse series explained in the
glossary.

| # | module (added by) | runs on | pulse series it reads | writes |
|---|---|---|---|---|
| 1 | `PropagateGenieInfo` (only if `is_genie`) | every frame | -- | `L4_n_flux_events` |
| 2 | `l3_cut` (only if `apply_l3_cut`) | P frames | -- | nothing; drops frames |
| 3 | `require_keys` | P frames | checks both series exist | nothing; may raise |
| 4 | `_first_hlc` | P | cleaned | `L4_first_hlc` |
| 5 | `_add_rho_36` | P | -- | `L4_first_hlc_rho` |
| 6 | `_full_time_length_ratio` | P | L3 map (fallback: cleaned + uncleaned) | `L4_FullTimeLengthRatio` |
| 7 | `I3HitStatisticsCalculatorSegment` (if `compute_hit_statistics`) | P | cleaned | `<cleaned>HitStatistics` |
| 8 | `I3HitMultiplicityCalculatorSegment` (same) | P | cleaned | `<cleaned>HitMultiplicity` |
| 9 | `I3StaticTWC<I3RecoPulseSeries>` | P | uncleaned (default) | `L4_TWPulses` |
| 10 | `I3OMSelection<I3RecoPulseSeries>` | P | `L4_TWPulses` | `L4_TWPulses_DCFid` |
| 11 | `I3TimeWindowCleaning<I3RecoPulse>` | P | `L4_TWPulses_DCFid` | `L4_TWPulses_DCFid_DTW200` |
| 12 | `_micro_count` | P | `L4_TWPulses_DCFid_DTW200` | `L4_micro_count` |
| 13 | `I3FillRatioModule` | P | cleaned, vertex `L4_first_hlc` | `L4_fill_ratio` |
| 14 | `I3TensorOfInertia` (only if `run_optional`) | P | cleaned | `L4_ToI` (+ params) |
| 15 | `linefit.simple` segment | P | cleaned | `L4_iLineFit`, `L4_iLineFitParams` |
| 16 | `_accumulated_time` | P | cleaned | `L4_accumulated_time` |
| 17 | `_separation_in_cogs` (only if `run_optional`) | P | cleaned | `L4_separation_in_cogs` |
| 18 | `_vich` | P | uncleaned | `L4_VICH_nch`, `L4_VICH_npulses`, `L4_VICH_qtot` |
| 19 | `L4_noise_straight_cuts` (only if `apply_cut`) | P | -- | `L4_NoiseStraightCuts_Bool` |
| 20 | `L4Classifier` noise (only if `apply_cut`) | P | -- | `L4_NoiseClassifier_ProbNu` |
| 21 | `L4Classifier` muon (only if `apply_cut`) | P | -- | `L4_MuonClassifier_Data_ProbNu` |
| 22 | `overall_cut` (only if `apply_cut`) | P | -- | `L4_oscNext_bool` |

---

## `oscnext_l4/__init__.py`

**What this file is for.** It makes `oscnext_l4` a Python package. Its
docstring says the library lives here, the command-line tools in `scripts/`
and the notebook in `notebooks/`. It also states the import policy: modules
import `from icecube import ...` directly, because the code always runs inside
one IceTray environment (v1.17.0). The projects that environment lacks
(`oscNext`, `tau_bdt`, `analysis`, `SimpleVertex`) are never imported.

**Where it sits.** Python runs it on any `import oscnext_l4...`. It imports
nothing from the package itself, so importing the package does not pull in
IceTray.

**Constants.**

| name | value | meaning |
|---|---|---|
| `__all__` | sorted list of every `*.py` in the package folder except `__init__.py` | the list of modules. It is built from the folder contents rather than written by hand, because `scripts/make_release.py` ships only some modules, and a fixed list would name modules the release left out. |

**Functions and classes.** None.

---

## `oscnext_l4/variables.py`

**What this file is for.** It holds the L4 **tray segments**: the functions
that add the L4 modules to a tray, in the right order, with the right
parameters. It is written to be read next to the production's own
`oscNext/python/selection/oscNext_L4.py` (whose body is commented out in the
official project). It keeps that file's key names (`L4_*`), its segment names
and its segment order. Code that is not a segment was moved to the neighbour
files (`rewritten.py`, `frame_objects.py`, `l3vars.py`).

**Where it sits.**
- Imported by `scripts/process_L4.py` (which takes `oscNext_L4`,
  `L4_HDF5_KEYS`, `HITSTAT_KEY`, `HITMULT_KEY` and the two pulse-series
  defaults) and by `oscnext_l4/classifier.py` (which takes
  `L4_CUT_BOOL_KEY`).
- It imports `calc_rho_36`, `iter_map`, `get_pulses`, `deepcore_doms`,
  `PropagateGenieInfo` and `L4_NFLUX_KEY` from `frame_objects.py`;
  `_full_time_length_ratio` and `L4_FTLR_KEY` from `l3vars.py`;
  `_first_hlc`, `_accumulated_time`, `_separation_in_cogs`, `_vich` from
  `rewritten.py`; `load_deserialization_libs` from `tray_io.py`. Because these
  are imported here, they are also attributes of this module
  (`from oscnext_l4.variables import _vich` works).
- Inside `compute_L4_cut` it imports `add_L4_classifiers` from
  `classifier.py` (a local import, to avoid a circular import, since
  `classifier.py` imports this file).
- IceTray projects imported at module level: `dataclasses`, `icetray`,
  `linefit`, `DomTools` (registers `I3OMSelection` and
  `I3TimeWindowCleaning`), `fill_ratio` (registers `I3FillRatioModule`).
  Loaded on demand: `static-twc` (in `oscNext_L4_noise_cut_variables`),
  `common_variables` (in `oscNext_L4_hit_statistics`), `tensor_of_inertia`
  and `slc-veto` (only on the optional paths).
- **Side effect on import:** it calls `load_deserialization_libs()` once, so
  that frame objects from simulation libraries can be unpacked.

**Constants / configuration.**

| name | value | meaning |
|---|---|---|
| `L4_CUT_BOOL_KEY` | `"L4_oscNext_bool"` | the final L4 pass/fail flag. Same name as the production, which matters because the collaboration's tools cut on this key. |
| `L4_FIRST_HLC_KEY` | `"L4_first_hlc"` | vertex of the earliest HLC hit (an `I3Particle`) |
| `L4_FIRST_HLC_RHO_KEY` | `"L4_first_hlc_rho"` | horizontal distance of that vertex from string 36 |
| `L4_TOI_KEY` | `"L4_ToI"` | tensor-of-inertia fit (optional) |
| `L4_LINEFIT_KEY` | `"L4_iLineFit"` | improved LineFit; its parameters go to `"L4_iLineFitParams"` |
| `L4_QRBOX_KEY` | `"L4_QR_Box"` | slc-veto QR box (optional; see the watch-out under `oscNext_L4_atm_muon_classifier_variables`) |
| `L4_VICH_NCH_KEY` | `"L4_VICH_nch"` | VICH: number of DOMs with a veto-causal pulse |
| `L4_VICH_NPULSES_KEY` | `"L4_VICH_npulses"` | VICH: number of veto-causal pulses |
| `L4_VICH_QTOT_KEY` | `"L4_VICH_qtot"` | VICH: total charge of those pulses |
| `L4_SEP_IN_COGS_KEY` | `"L4_separation_in_cogs"` | Dunkman separation of charge-quartile COGs (optional) |
| `L4_ACC_TIME_KEY` | `"L4_accumulated_time"` | Dunkman accumulated time |
| `L4_MICROCOUNT_KEY` | `"L4_micro_count"` | micro count, an `I3MapStringInt` |
| `L4_FILL_RATIO_KEY` | `"L4_fill_ratio"` | fill ratio, an `I3FillRatioInfo` |
| `L4_NOISE_STRAIGHT_CUT_KEY` | `"L4_NoiseStraightCuts_Bool"` | the loose noise straight cuts (computed, not used in the cut) |
| `L4_NOISE_MODEL_PREDICTION_KEY` | `"L4_NoiseClassifier_ProbNu"` | noise-classifier score |
| `L4_MUON_MODEL_PREDICTION_DATA_KEY` | `"L4_MuonClassifier_Data_ProbNu"` | muon-classifier score |
| `UNCLEANED_PULSES_DEFAULT` | `"SplitInIcePulses"` | uncleaned series; the same name in pass2 and pass3 |
| `CLEANED_PULSES_DEFAULT` | `"SRTTWSplitInIcePulsesDC"` | cleaned series, **pass3 name**. pass2 calls it `"SRTTWOfflinePulsesDC"`; pass it via `--cleaned-pulses`. |
| `HITSTAT_KEY` | `"SRTTWSplitInIcePulsesDCHitStatistics"` | hit statistics output key (built from `CLEANED_PULSES_DEFAULT`) |
| `HITMULT_KEY` | `"SRTTWSplitInIcePulsesDCHitMultiplicity"` | hit multiplicity output key (same) |
| `STW_MINUS` | `3500.` | micro count static time window, ns before the trigger |
| `STW_PLUS` | `4000.` | micro count static time window, ns after the trigger |
| `DTW` | `200` | micro count dynamic time window length, ns |
| `MICROCOUNT_SUBKEY` | `"STW_m3500p4000_DTW200"` | the map key inside `L4_micro_count` (the production's format string, verbatim) |
| `FILL_RATIO_SPHERICAL_RADIUS_MEAN` | `1.6` | fill-ratio sphere radius, as a multiple of the mean hit distance. Tuned for GRECO, never re-tuned for oscNext. |
| `L4_HDF5_KEYS` | list of 20 keys | the L4 keys `process_L4.py` asks the booker to write (see the key table at the end of this section) |
| `PULSE_CHECK_FRAMES` | `20` | how many Physics frames `_require_keys` watches before deciding a pulse series is missing |

> **Watch out: `HITSTAT_KEY` / `HITMULT_KEY` always carry the pass3 name.**
> They are built from the constant, not from the `cleaned_pulses` argument.
> On a pass2 run the values are computed from `SRTTWOfflinePulsesDC` but
> stored under the pass3-spelled key. `classifier.py` (via
> `config/variables.json`) reads the same spelling, so writer and reader
> agree. This is deliberate.

### `_add_rho_36(frame, particle_key, output_key)`

A function module. If `particle_key` is in the frame and `output_key` is not,
it takes the particle's position and writes
`I3Double(calc_rho_36(pos.x, pos.y))` to `output_key`. Always returns `True`
(it never drops a frame).

- Reads: `particle_key` (here `L4_first_hlc`). Writes: `output_key` (here
  `L4_first_hlc_rho`).
- Watch out: when the event has no HLC hit, `_first_hlc` writes a sentinel
  vertex at (1000, 1000, 1000), so this writes rho = 1407.3. That is what the
  production did too, and both the fill ratio and the muon BDT saw that value.

### `oscNext_L4_common_variables(tray, name, cleaned_pulses, uncleaned_pulses=None)`

Tray segment. Adds the variables shared by both classifiers.

Modules added, in order:
1. `_first_hlc` (named `<name>_FirstHLC`) on `cleaned_pulses` -> writes
   `L4_first_hlc`.
2. `_add_rho_36` (`<name>_FirstHLCRho`) on `L4_first_hlc` -> writes
   `L4_first_hlc_rho`.
3. `_full_time_length_ratio` (`<name>_FTLR`) -> writes
   `L4_FullTimeLengthRatio`. It reads the L3 map first; `cleaned_pulses` and
   `uncleaned_pulses` are only used as a fallback (see `l3vars.py`). With
   `uncleaned_pulses=None` the fallback is disabled.

### `oscNext_L4_atm_muon_classifier_variables(tray, name, uncleaned_pulses, cleaned_pulses, run_qr_box=False, run_optional=False, accumulated_time_pass2=True)`

Tray segment. Adds the inputs of the atmospheric-muon classifier.

Modules added, in order:
1. **Only if `run_optional`:** imports `tensor_of_inertia` and adds
   `I3TensorOfInertia` (`<name>_ToI`) on `cleaned_pulses`, with
   `AmplitudeOption=1`, `AmplitudeWeight=1`, `InputSelection=""`,
   `MinHits=3`, `Name="L4_ToI"`. Not a BDT input.
2. `linefit.simple` segment (`<name>_iLineFit`), `inputResponse=cleaned_pulses`,
   `fitName="L4_iLineFit"`. This is the "improved LineFit"; it writes the fit
   particle and `L4_iLineFitParams`. The BDT uses the speed from the params
   object.
3. **Only if `run_qr_box`:** loads the `slc-veto` library and adds
   `SmallQ_Box` (`<name>_QRBox`) on `cleaned_pulses`, writing `L4_QR_Box`. If
   loading or adding fails, it logs a warning and carries on.
4. `_accumulated_time` (`<name>_AccTime`) on `cleaned_pulses` -> writes
   `L4_accumulated_time`. `before_crossing=accumulated_time_pass2`.
5. **Only if `run_optional`:** `_separation_in_cogs` (`<name>_SepCOG`) on
   `cleaned_pulses` -> writes `L4_separation_in_cogs`. Not a BDT input.
6. `_vich` (`<name>_VICH`) on **`uncleaned_pulses`** -> writes
   `L4_VICH_nch`, `L4_VICH_npulses`, `L4_VICH_qtot`.

Parameters:
- `accumulated_time_pass2=True` (default) reproduces the production.
  `False` follows the technical note (Table 12). See `_accumulated_time`.
- `run_optional=False` (default) skips the two non-BDT computations.

> **Watch out: the QR box never runs from the main segment.** `oscNext_L4`
> does not pass `run_qr_box`, so it stays `False`, even with `--run-optional`.
> `L4_QR_Box` is in `L4_HDF5_KEYS` but will not be produced by
> `process_L4.py`. (The `--run-optional` help text only promises ToI and
> separation_in_cogs, so this matches the CLI, but not every comment that
> groups slc-veto with `--run-optional`.)
>
> **Watch out:** the comment above the LineFit call says the BDT field is
> `L4_iLineFitParams.LFVel`. The variable table (`config/variables.json`)
> lists the frame field as `lf_vel` first, with `LFVel` and `speed` as
> alternatives. `classifier.read_feature` tries all three.

### `_micro_count(frame, pulses_key, output_key, subkey)`

Function module. Counts the **DOMs** (not pulses) in the pulse series
`pulses_key` and writes the count as `I3MapStringInt({subkey: n})` to
`output_key`, if that key is not already there. Always returns `True`.

Steps:
1. Get the series with `get_pulses` (applies a mask if needed).
2. If the series is absent, the count is 0.
3. Otherwise count its entries with `len(pmap)`; if `len` fails, count by
   iterating with `iter_map`.
4. Write the map.

- Reads: `pulses_key` (here `L4_TWPulses_DCFid_DTW200`). Writes:
  `L4_micro_count` with subkey `STW_m3500p4000_DTW200`.
- Watch out: a missing input series gives micro_count = 0, not a missing
  value.

### `oscNext_L4_noise_cut_variables(tray, name, fill_ratio_vertex, cleaned_pulses, micro_count_pulses=UNCLEANED_PULSES_DEFAULT)`

Tray segment. Adds the inputs of the pure-noise classifier: micro count and
fill ratio. If `micro_count_pulses` is `None`, the cleaned series is used for
micro count instead.

It first loads the `static-twc` library. Modules added, in order:

1. `I3StaticTWC<I3RecoPulseSeries>` (`<name>_StaticTWC_DC`).
   Input `micro_count_pulses` (default: the **uncleaned** series). Keeps the
   pulses in the window [trigger time - 3500 ns, trigger time + 4000 ns]
   around the trigger with config id 1010 or 1011, found in
   `I3TriggerHierarchy`. Output `L4_TWPulses`.
2. `I3OMSelection<I3RecoPulseSeries>` (`<name>_DCFidPulses`). Input
   `L4_TWPulses`. `OmittedKeys` is the DeepCore fiducial DOM list from
   `deepcore_doms("IC86")`, and `selectInverse=True` inverts the selection, so
   the result keeps **only** the fiducial DOMs. Output `L4_TWPulses_DCFid`.
3. `I3TimeWindowCleaning<I3RecoPulse>` (`<name>_DynamicTW`). Input
   `L4_TWPulses_DCFid`. Keeps only the pulses inside the densest 200 ns
   window. Output `L4_TWPulses_DCFid_DTW200`.
4. `_micro_count` (`<name>_MicroCount`) on `L4_TWPulses_DCFid_DTW200` ->
   writes `L4_micro_count["STW_m3500p4000_DTW200"]`, the number of DOMs left.
5. `I3FillRatioModule` (`<name>_FillRatio`). `RecoPulseName=cleaned_pulses`,
   `VertexName=fill_ratio_vertex` (the main segment passes `L4_first_hlc`),
   `SphericalRadiusMean=1.6`, `ResultName="L4_fill_ratio"`. It writes an
   `I3FillRatioInfo`; the BDT uses its `fill_ratio_from_mean` field (booked
   in HDF5 as `fillratio_from_mean`). Roughly: the fraction of DOMs inside a
   sphere around the vertex that were hit, where the sphere radius is 1.6
   times the mean distance of the hits from the vertex.

> **Watch out: micro count follows the production, not the note.** The
> technical note (Table 11) says to start from the **cleaned** series. The
> production starts from the **uncleaned** series, and that is the default
> here, because it reproduces pass2 bitwise. `micro_count_pulses=None`
> (`--micro-count-cleaned`) gives the note's version; the two agree in about
> 91% of signal and 94% of noise events.
>
> The production also ran an `I3SeededRTCleaning` step whose output nobody
> read. It is left out here; that is why the result is still identical.
>
> The fill ratio always uses the cleaned series.
>
> The three intermediate series (`L4_TWPulses...`) are written into the frame
> under fixed names, regardless of the segment name.

### `oscNext_L4_hit_statistics(tray, name, cleaned_pulses)`

Tray segment. Recomputes the hit statistics that pass3's L3 computes and then
deletes. It imports `hit_statistics` and `hit_multiplicity` from
`icecube.common_variables`. Modules added:

1. `I3HitStatisticsCalculatorSegment` (`<name>_HitStatistics`) on
   `cleaned_pulses` -> writes `HITSTAT_KEY` (an `I3HitStatisticsValues`:
   `cog`, `z_sigma`, `z_travel`, ...). `BookIt=False`.
   `If=lambda f: HITSTAT_KEY not in f`, so it is skipped when the key already
   exists.
2. `I3HitMultiplicityCalculatorSegment` (`<name>_HitMultiplicity`) on
   `cleaned_pulses` -> writes `HITMULT_KEY` (an `I3HitMultiplicityValues`,
   including `n_hit_doms`). Same `If` guard.

`cog.z`, `z_sigma` and `z_travel` are muon BDT inputs. `n_hit_doms` is used by
the straight cuts.

- Watch out: pass2 L3 already stores hit statistics, but under the pass2 name
  (`SRTTWOfflinePulsesDCHitStatistics`). The `If` checks the pass3-spelled
  key, so on pass2 they are recomputed and stored under the pass3 name.

### `L4_noise_straight_cuts(frame, output_key=L4_NOISE_STRAIGHT_CUT_KEY)`

Function module. Computes the production's loose noise straight cuts and
writes the result as an `I3Bool`. The result is **not** used in the L4 cut. It
is kept as a fallback and as a guide to which direction each variable
separates. Always returns `True`.

The event passes if **all** of these hold:
- `HITMULT_KEY.n_hit_doms >= 8`
- `IC2018_LE_L3_Vars["STW9000_DTW300Hits"] >= 2`
- `L4_micro_count["STW_m3500p4000_DTW200"] >= 2`
- `L4_fill_ratio.fill_ratio_from_mean >= 0.03`
- `HITSTAT_KEY.z_sigma >= 8`
- `HITSTAT_KEY.z_travel >= -50`

If any key or field is missing (`KeyError`/`AttributeError`), the result is
`False`.

- Reads: the keys above. Writes: `L4_NoiseStraightCuts_Bool`.
- Watch out: it writes without checking whether the key already exists, so
  running it on a frame that already has the key would fail.

### `compute_L4_cut(tray, name, classifier_model_dir, noise_cut=0.70, muon_cut=0.65)`

Tray segment. Applies the trained classifiers and computes the L4 cut. Only
added when `oscNext_L4(apply_cut=True)`.

Modules added, in order:
1. `L4_noise_straight_cuts` (`<name>_straight_cuts`).
2. `add_L4_classifiers` segment (`<name>_classifiers`) from `classifier.py`,
   with `model_dir=classifier_model_dir`, the two cut values, and
   `apply_cut=True`. That adds the two `L4Classifier` modules and
   `overall_cut` (see `classifier.py`).

- Writes: `L4_NoiseStraightCuts_Bool`, `L4_NoiseClassifier_ProbNu`,
  `L4_MuonClassifier_Data_ProbNu`, `L4_oscNext_bool`.
- Watch out: this does **not** remove events. It writes the pass/fail flag
  `L4_oscNext_bool`, and all events stay in the output. Both model files
  (`L4_noise_model.txt` and `L4_muon_model.txt`) must exist in
  `classifier_model_dir`, or `L4Classifier.Configure` raises.

### `_require_keys(*keys)` and its nested `require_keys(frame)`

`_require_keys` is a factory: it returns a function module, `require_keys`,
that checks the given frame keys exist. It keeps a counter and a per-key
"seen" count in its closure.

What `require_keys(frame)` does on each Physics frame:
1. After `PULSE_CHECK_FRAMES` (20) frames it does nothing and returns `True`.
2. Otherwise it counts the frame and notes which of the keys are present.
3. On the 20th frame, if any key was absent from **all** 20 frames, it raises
   `RuntimeError`. The message says the likely cause: the pass2 cleaned series
   is `SRTTWOfflinePulsesDC`, so use `--cleaned-pulses SRTTWOfflinePulsesDC`.
4. Always returns `True` otherwise (never drops a frame).

Why it exists: a wrong pulse-series name makes no module fail. Every variable
built on it is just never written, and the HDF5 fills with missing values that
training accepts silently.

- Watch out: it only checks that the key is present. A mask whose base series
  is missing still passes this check, and `get_pulses` would then return
  `None` silently. A run shorter than 20 Physics frames is never checked.

### `oscNext_L4(tray, name, uncleaned_pulses=UNCLEANED_PULSES_DEFAULT, cleaned_pulses=CLEANED_PULSES_DEFAULT, apply_l3_cut=True, is_genie=False, compute_hit_statistics=True, run_optional=False, apply_cut=False, classifier_model_dir=None, micro_count_uncleaned=True, accumulated_time_pass2=True)`

**The main segment.** `scripts/process_L4.py` adds this one segment, and it
adds everything else.

Modules and segments added, in order:
1. **If `is_genie`:** `PropagateGenieInfo` (`<name>_genie_info`). It is placed
   before the L3 cut so that the non-Physics frames it reads are not affected
   by the cut.
2. **If `apply_l3_cut`:** the nested function module `l3_cut`
   (`<name>_L3_cut`), described below. It drops Physics frames that fail L3.
3. `_require_keys(cleaned_pulses, uncleaned_pulses)` (`<name>_pulse_check`).
4. `oscNext_L4_common_variables` (`<name>_common`).
5. **If `compute_hit_statistics`:** `oscNext_L4_hit_statistics`
   (`<name>_hitstats`).
6. `oscNext_L4_noise_cut_variables` (`<name>_noise_vars`), with
   `fill_ratio_vertex="L4_first_hlc"` and `micro_count_pulses` set to the
   uncleaned series if `micro_count_uncleaned`, else `None` (cleaned).
7. `oscNext_L4_atm_muon_classifier_variables` (`<name>_muon_vars`), passing
   `run_optional` and `accumulated_time_pass2` (not `run_qr_box`).
8. **If `apply_cut`:** `compute_L4_cut` (`<name>_cut`). Raises `ValueError` at
   configuration time if `classifier_model_dir` is `None`.

Parameters that matter:
- `apply_cut=False` (default): compute variables only, no classifier. This is
  the mode for producing training data.
- `accumulated_time_pass2=True` and `micro_count_uncleaned=True` (defaults):
  **both reproduce the production, not the technical note.** This is
  deliberate: the model must be trained on the same quantity the production
  used. Setting either to `False` gives the note's reading.

#### nested `l3_cut(frame)`

Function module, defined inside `oscNext_L4`. L3 does not drop events; it
only writes flags. This applies the flag:
1. If `L3_oscNext_bool` is in the frame, keep the frame when it is true.
   (This flag is `IC2018_LE_L3_Full` AND data quality.)
2. Otherwise, if `IC2018_LE_L3_bools` is present, keep the frame when its
   `IC2018_LE_L3_Full` entry is true.
3. Otherwise drop the frame.

Returning `False` from a function module drops the frame.

### The `L4_*` frame keys this file produces

"BDT input" means the key (or a field of it) is one of the classifier inputs
(5 noise + 10 muon; 14 distinct, since `NchCleaned` is in both). The other BDT inputs come from L3
(`IC2018_LE_L3_Vars`: `NchCleaned`, `ICVetoHits`, `RTVeto250Hits`,
`NAbove200Hits`) and from `HITSTAT_KEY` (`cog.z`, `z_sigma`, `z_travel`).

| frame key | type | written by | when | BDT input? |
|---|---|---|---|---|
| `L4_n_flux_events` | I3Double | `PropagateGenieInfo` (frame_objects.py) | `is_genie` | no (weights) |
| `L4_first_hlc` | I3Particle | `_first_hlc` (rewritten.py) | always | no; it is the fill-ratio vertex |
| `L4_first_hlc_rho` | I3Double | `_add_rho_36` | always | **muon** |
| `L4_FullTimeLengthRatio` | I3Double | `_full_time_length_ratio` (l3vars.py) | always | **noise** |
| `L4_TWPulses` | pulse series | `I3StaticTWC` | always | no (intermediate, not booked) |
| `L4_TWPulses_DCFid` | pulse series | `I3OMSelection` | always | no (intermediate, not booked) |
| `L4_TWPulses_DCFid_DTW200` | pulse series | `I3TimeWindowCleaning` | always | no (intermediate, not booked) |
| `L4_micro_count` | I3MapStringInt | `_micro_count` | always | **noise** (`STW_m3500p4000_DTW200`) |
| `L4_fill_ratio` | I3FillRatioInfo | `I3FillRatioModule` | always | **noise** (`fill_ratio_from_mean`) |
| `L4_ToI`, `L4_ToIParams` | fit result | `I3TensorOfInertia` | `run_optional` | no |
| `L4_iLineFit`, `L4_iLineFitParams` | I3Particle, I3LineFitParams | `linefit.simple` | always | **noise** (speed, from `...Params`) |
| `L4_QR_Box` | slc-veto output | `SmallQ_Box` | only `run_qr_box`, never from `oscNext_L4` | no |
| `L4_accumulated_time` | I3Double | `_accumulated_time` (rewritten.py) | when the cleaned series has > 4 DOMs | **muon** |
| `L4_separation_in_cogs` | I3Double | `_separation_in_cogs` (rewritten.py) | `run_optional` | no |
| `L4_VICH_nch` | I3Double | `_vich` (rewritten.py) | always | **muon** |
| `L4_VICH_npulses` | I3Double | `_vich` | always | no |
| `L4_VICH_qtot` | I3Double | `_vich` | always | no |
| `L4_NoiseStraightCuts_Bool` | I3Bool | `L4_noise_straight_cuts` | `apply_cut` | no |
| `L4_NoiseClassifier_ProbNu` | I3Double | `L4Classifier` (classifier.py) | `apply_cut` | output of the noise BDT |
| `L4_MuonClassifier_Data_ProbNu` | I3Double | `L4Classifier` (classifier.py) | `apply_cut` | output of the muon BDT |
| `L4_oscNext_bool` | I3Bool | `overall_cut` (classifier.py) | `apply_cut` | the final cut flag |

Also written (not `L4_*`): `SRTTWSplitInIcePulsesDCHitStatistics` and
`SRTTWSplitInIcePulsesDCHitMultiplicity` by `oscNext_L4_hit_statistics`.

pass2 stores `L4_VICH_nch` and `L4_VICH_npulses` as `I3Int`; this code writes
`I3Double`. Both book to a `value` column, so the HDF5 comparison is not
affected.

---

## `oscnext_l4/rewritten.py`

**What this file is for.** The production's L4 script calls three IceTray
modules that the environment used here (IceTray v1.17.0 on CVMFS) does not
have. This file rewrites each one in pure Python (numpy), as a function
module:

| rewrite | replaces | production module |
|---|---|---|
| `_first_hlc` | `FirstHLC<I3RecoPulse>` | `SimpleVertex` (C++) |
| `_accumulated_time`, `_separation_in_cogs` | `CalculateVariables` ("Dunkman variables") | `analysis.event_selection` (C++) |
| `_vich` | `I3CutL7Module` | `tau_bdt` (Python) |

Each was checked against the real pass2 L4 files. The docstrings record where
each definition came from and what was measured; read them before changing
anything.

**Where it sits.** Imported by `variables.py`, which adds these functions to
the tray. It imports `iter_map`, `get_pulses` and `_LC_FLAG` from
`frame_objects.py`, plus numpy and `icecube.dataclasses`.

**Constants.**

| name | value | meaning |
|---|---|---|
| `FIRST_HLC_SENTINEL_POS` | `(1000.0, 1000.0, 1000.0)` | vertex written when the event has no HLC hit (copies the original's behaviour) |
| `FIRST_HLC_SENTINEL_TIME` | `1e10` | time written in that case; also the starting value of the search |

All four functions below are function modules: they take the frame plus
keyword parameters, and always return `True` (they never drop a frame). Each
does nothing if its output key is already in the frame.

### `_first_hlc(frame, pulses_key, output_key, geometry_key="I3Geometry")`

Finds the earliest HLC pulse in the series and writes the position of its DOM
as an `I3Particle`.

**Algorithm:**
1. If `output_key` exists, stop. Get the series with `get_pulses`. If the
   series or `I3Geometry` is missing, stop **without writing anything**.
2. Set `best_time = 1e10`, `best_pos = None`.
3. Loop over DOMs in the map's order (ascending OMKey). Skip a DOM that is not
   in the geometry.
4. Within a DOM, find the first pulse with the LC flag set. (Pulses in a DOM
   are time-ordered, so this is the DOM's earliest HLC pulse.) If its time is
   `<= best_time`, set `best_time` to that time and `best_pos` to the DOM
   position. Then move on to the next DOM.
5. Build an `I3Particle`: position `best_pos` and time `best_time`, or the
   sentinel (1000, 1000, 1000) and 1e10 if no HLC pulse was found. Shape
   `Cascade`, fit status `OK`. Write it to `output_key`.

- Reads: `pulses_key` (cleaned series), `I3Geometry`. Writes: `L4_first_hlc`.
- The vertex is the **DOM position**, not charge-weighted.
- **Two details copied from the C++ on purpose:**
  - Ties go to the **last** DOM in OMKey order (the comparison is `<=`),
    because the C++ only skips strictly later hits.
  - An event with no HLC pulse still gets a vertex (the sentinel), because
    the C++ ignores the "found" flag. That gives `first_hlc_rho` = 1407.3.
- Verified: `first_hlc_rho` and `fill_ratio` are bitwise identical to pass2
  over 56,301 events.

### `_accumulated_time(frame, pulses_key, output_key, fraction=0.75, before_crossing=True)`

The Dunkman "accumulated time": the time from the first pulse to roughly the
point where the event has collected 75% of its charge.

**Algorithm (default, `before_crossing=True`, the production rule):**
1. If `output_key` exists, or the series is missing, stop.
2. Loop over the map in OMKey order. Count DOMs. Collect every pulse's time
   and charge. Add up the total charge `Q` **in this map order**, one pulse at
   a time. (This matches the original's summation order; a different order
   can change the last bit of `Q` and move the 75% boundary.)
3. If there are **4 or fewer DOMs**, or no pulses, stop **without writing**.
   The original writes nothing in that case either.
4. If `Q` is not finite or `Q <= 0`, stop without writing.
5. Sort pulses by time with a **stable** sort. Compute the running sum of
   charge `cum` in time order. Negative charges are not clamped.
6. Find the pulses in the third charge quartile:
   `0.5*Q < cum <= 0.75*Q`.
7. If there are none, write `0.0`. (The original's struct default is zero.
   This happens when one pulse carries the running sum from below 0.5Q to
   above 0.75Q.)
8. Otherwise take the **last** pulse in that band, and write
   `time(that pulse) - time(first pulse)` as an `I3Double`.

**With `before_crossing=False` (the technical note's reading):** steps 1-5
are the same; then take the **first** pulse whose running sum reaches
`fraction * Q` (`np.searchsorted`, capped at the last pulse), and write its
time minus the first pulse time. This gives "the time to reach 75%".

- Reads: `pulses_key` (cleaned). Writes: `L4_accumulated_time`.
- Why the default looks odd: the C++ module bins pulses into charge quartiles
  and sets the variable inside the Q3 branch for every pulse in that band. So
  the value it keeps is the last pulse before 75% is passed, not the crossing.
- Watch out: `fraction` is only used when `before_crossing=False`. The default
  path uses fixed 0.5 and 0.75.
- Watch out: the variable is **missing** (not zero) for events with 4 or fewer
  cleaned DOMs, and **zero** when the Q3 band is empty.
- Verified: agrees with pass2 in 99.84% of 56,301 events; max difference
  286 ns. The rest comes from pulses with equal times. The C++ uses an
  unstable sort, so the order of tied pulses there is unspecified and cannot
  be reproduced. The note's reading (`False`) agrees in only 0.98%.

### `_separation_in_cogs(frame, pulses_key, output_key, geometry_key="I3Geometry")`

The distance between the charge-weighted centre of gravity (COG) of the
event's **first** charge quartile and that of its **fourth**. A track moves,
so the two COGs are far apart; a cascade is compact, so they are close.

**Algorithm:**
1. If `output_key` exists, or the series or geometry is missing, stop.
2. Loop over DOMs in map order. Count every DOM. For DOMs present in the
   geometry, collect `(time, charge, x, y, z)` for each pulse (DOM position)
   and add the charge to the total `Q` (map order).
3. If 4 or fewer DOMs, or no pulses, stop without writing. If `Q` is not
   finite or `<= 0`, stop.
4. Sort the pulses by time (Python's sort, which is stable). Running sum of
   charge `cum`.
5. First quartile: pulses with `cum <= 0.25*Q`. Fourth quartile: pulses with
   `cum > 0.75*Q`. If either is empty, stop without writing.
6. For each quartile, the charge-weighted mean position (the nested `cog`
   helper). If the charge sum of a quartile is `<= 0`, stop without writing.
7. Write the Euclidean distance between the two COGs as an `I3Double`.

- Reads: `pulses_key` (cleaned), `I3Geometry`. Writes:
  `L4_separation_in_cogs`.
- Only runs with `run_optional`; not a BDT input.
- Verified: the definition was read from the C++ source
  (`separation = distance(cog_q1, cog_q4)`). **Its values have not been
  compared against pass2** (it is not in the cross-check table).

#### nested `cog(mask)`

Takes a boolean mask over the time-sorted pulses. Returns the
charge-weighted mean `(x, y, z)` of the selected pulses as a numpy array, or
`None` if their charges sum to 0 or less.

### `_vich(frame, uncleaned_pulses, nch_key, npulses_key, qtot_key, geometry_key="I3Geometry", trigger_key="I3TriggerHierarchy", config_ids=(1010, 1011))`

**VICH, "Veto Identified Causal Hits".** Counts the pulses that could be
light from something (e.g. an incoming muon) that caused the DeepCore
trigger: pulses in a region of (distance, time difference) around a reference
pulse near the trigger.

**Algorithm:**
1. If `nch_key` exists, stop. Get the uncleaned series. If it or the
   geometry is missing, stop without writing.
2. Collect trigger times: loop over the triggers in `I3TriggerHierarchy` (in
   the order Python iteration gives them). Keep the time of each trigger
   whose config id is in `config_ids` (1010, 1011). Triggers without a config
   id (MERGED, THROUGHPUT) are skipped safely.
3. Collect every pulse of the uncleaned series, one by one (pulse level, not
   DOM level): DOM position `(x, y, z)`, time, charge, and the DOM
   `(string, om)`. Skip DOMs not in the geometry.
4. If there are no pulses or no matching trigger, write **0** to all three
   keys and stop.
5. Take **only the first** matching trigger. The reference pulse `i` is the
   pulse closest in time to it (`argmin |t - t_trigger|`; ties go to the
   first in map order).
6. For every pulse, with `d` = distance to the reference DOM and
   `dt = t_ref - t_pulse`, select it if **all four** hold:
   - `d < 750` m
   - `dt > -5*d + 500`
   - `dt < d/0.3 + 150`
   - `dt > d/0.3 - 1850`
7. Write, as `I3Double`:
   - `nch_key`: number of distinct DOMs with at least one selected pulse;
   - `npulses_key`: number of selected pulses;
   - `qtot_key`: sum of the selected pulses' charges (no clamping).

- Reads: `uncleaned_pulses` (`SplitInIcePulses`), `I3Geometry`,
  `I3TriggerHierarchy`. Writes: `L4_VICH_nch`, `L4_VICH_npulses`,
  `L4_VICH_qtot`.
- There is **no veto-DOM list and no speed window**: the whole uncleaned
  series is scanned, and the region is set only by the four conditions.
  (Section 3.4 of the technical note describes a different algorithm, the L2
  DeepCore filter.)
- Watch out: an event with no 1010/1011 trigger, or no `I3TriggerHierarchy`,
  gets VICH = 0 rather than a missing value.
- One deliberate difference from the original: the original starts its
  reference search at t = 0, position (0, 0, 0). If no pulse were closer to
  the trigger than t = 0, the origin would be used. This never happens in
  practice, and the rewrite does not copy it.
- Verified: 100.00% agreement with pass2 on `nch`, `npulses` and `qtot` over
  8144 events. The details (first trigger only, `[1010, 1011]`, `nch` as DOMs)
  were confirmed by reading the production's `I3CutL7Module.py`.

---

## `oscnext_l4/frame_objects.py`

**What this file is for.** Small helpers that the production keeps in its
`oscNext/python/frame_objects/` directory, which this environment does not
have. One file mirrors that directory, with one section per original file:
`geom.py` (rho_36, DOM lists), `pulses.py` (pulse-map access), `weighting.py`
(the GENIE-info propagator).

**Where it sits.** Imported by `variables.py`, `rewritten.py` and
`l3vars.py`. It imports numpy and `icecube.dataclasses`/`icetray`;
`deepcore_doms` imports `icecube.DeepCore_Filter` on first use.

**Constants.**

| name | value | meaning |
|---|---|---|
| `STRING36_X` | `46.29` | x of string 36 (DeepCore centre), m |
| `STRING36_Y` | `-34.88` | y of string 36, m |
| `_doms_cache` | `{}` | cache of `DOMS.DOMS(detector)` objects, per detector string |
| `_LC_FLAG` | `I3RecoPulse.PulseFlags.LC` | the pulse flag bit that marks an HLC pulse |
| `L4_NFLUX_KEY` | `"L4_n_flux_events"` | frame key `PropagateGenieInfo` writes |

### `calc_rho_36(x, y)`

Returns the horizontal distance from string 36:
`sqrt((x - 46.29)**2 + (y + 34.88)**2)`, as a Python float.

- Pure function; no frame access.
- Written exactly like the production's formula (not `np.hypot`), because
  `hypot` differs in the last bit; with this form `first_hlc_rho` matches
  pass2 bitwise.

### `deepcore_doms(detector="IC86")`

Returns `icecube.DeepCore_Filter.DOMS.DOMS(detector)`, building it only once
per detector string and caching it in `_doms_cache`. The returned object has
the attributes `DeepCoreFiducialDOMs` and `DeepCoreVetoDOMs` (lists of
OMKeys).

- Used by `oscNext_L4_noise_cut_variables` (micro count's fiducial selection).
- There is no fallback: if `DeepCore_Filter` cannot be imported, this raises.

### `iter_map(pulse_map)`

Returns something to loop over as `(omkey, pulses)` pairs. Returns `[]` for
`None`. Otherwise returns `pulse_map.items()`, or, if the object has no
`.items()`, `iter(pulse_map)`.

- Why: in some IceTray versions, iterating a pulse map directly gives keys,
  not pairs, and `for omkey, pulses in pmap` then fails.

### `get_pulses(frame, key)`

Returns the pulse map stored at `key`, or `None` if the key is absent. If the
object has an `.apply` method (a mask or union), it returns
`obj.apply(frame)`, the real pulse map.

- Watch out: if `.apply` raises (for example because the mask's base series is
  missing), the exception is swallowed and `None` is returned. Every rewrite
  then silently writes nothing. Nothing is logged.

### `class PropagateGenieInfo(icetray.I3Module)`

A module that copies `I3GenieInfo.n_flux_events` into every Physics frame as
`I3Double` under `L4_n_flux_events`. `I3GenieInfo` sits once per file in a
non-Physics frame; HDF5 booking works per Physics frame, so the value has to
be copied to be usable in the weight calculation (`data.py`). Only added when
`oscNext_L4(is_genie=True)`.

Parameter: `OutputKey` (default `"L4_n_flux_events"`). One outbox.

#### `__init__(self, context)`
Registers the `OutputKey` parameter and the `OutBox`.

#### `Configure(self)`
Reads `OutputKey`. Sets `n_flux = None` (no value yet), `warned = False`,
`n_seen = 0`.

#### `_grab(self, frame)`
If the frame has `I3GenieInfo`, read `n_flux_events` as a float and store it
in `self.n_flux`, replacing any earlier value (each L3 file has its own
`I3GenieInfo`, and one tray may read several files). Logs at info level,
including a message when the value changes. If reading fails (for example the
GENIE library is not loaded), it logs one warning and leaves `n_flux` as it
was.

#### `Process(self)`
Handles **every** frame (it overrides `Process`, so the per-stream methods
like `Physics` are never called):
1. Pop the frame.
2. Non-Physics frame: call `_grab`, push the frame, done.
3. Physics frame: call `_grab`; if a value is known and the output key is not
   yet in the frame, write `I3Double(n_flux)`. If no value is known yet, log
   one warning that the weight will fall back to `NEvents * 0.7/0.3`.
4. Push the frame.

- Reads: `I3GenieInfo`. Writes: `L4_n_flux_events`.
- Watch out: `_grab` also runs on Physics frames. If `I3GenieInfo` is visible
  from Physics frames (through frame mixing), `n_seen` counts frames rather
  than files, and the info message is logged for every event. `n_seen` is
  only used in a log message, so values are not affected. I could not tell
  from the code alone whether this happens.
- Context: the official production computes the GENIE weight from
  `NEvents * gen_ratio` and never reads `I3GenieInfo`; see CLAUDE.md. The
  weight itself is computed later, in `data.py`, not here.

---

## `oscnext_l4/l3vars.py`

**What this file is for.** `FullTimeLengthRatio` is an L3 variable: the
production's L3 writes it into `IC2018_LE_L3_Vars`. pass2's L3 map carries it,
but pass3's only carries its two parts (`CleanedFullTimeLength`,
`UncleanedFullTimeLength`). So this file computes the ratio at L4.

**Where it sits.** Imported by `variables.py`
(`oscNext_L4_common_variables` adds it). It imports `iter_map` and
`get_pulses` from `frame_objects.py`.

**Constants.**

| name | value | meaning |
|---|---|---|
| `L4_FTLR_KEY` | `"L4_FullTimeLengthRatio"` | output frame key |

### `_full_time_length_ratio(frame, output_key, l3_key="IC2018_LE_L3_Vars", cleaned_pulses=None, uncleaned_pulses=None)`

Function module. Writes cleaned duration / uncleaned duration, where a
duration is the latest pulse time minus the earliest. A noise BDT input.
Always returns `True`.

Steps:
1. If `output_key` exists, stop.
2. If the L3 map exists and contains both `CleanedFullTimeLength` and
   `UncleanedFullTimeLength`, use them.
3. Otherwise, if both pulse-series names were given, measure the two
   durations directly from the series (nested `duration`).
4. If a duration is missing, or the uncleaned one is `<= 0`, stop without
   writing.
5. Compute the ratio. If it is not finite (NaN or inf), stop without writing.
6. If the ratio is above 1, log a warning **once per process** (a flag stored
   on the function object). The cleaned series is a subset of the uncleaned
   one, so a ratio above 1 means the two were measured on different series.
   The value is still written.
7. Write the ratio as an `I3Double`.

- Reads: `IC2018_LE_L3_Vars` (or the two pulse series). Writes:
  `L4_FullTimeLengthRatio`.
- Direction: cleaned / uncleaned, so in [0, 1]. This matches the production
  code and Figure 13 of the note. Verified: identical to pass2's stored ratio
  over 8144 events.
- Physics note from the docstring: the uncleaned series spans the whole
  ~10 µs readout, so the ratio is in practice "cleaned duration / 10 µs", and
  noise events have a *larger* ratio than neutrino events (median 0.27 vs
  0.16 on a small sample).
- On pass2 the stored `FullTimeLengthRatio` is not read; the ratio is always
  recomputed from the two parts.

#### nested `duration(key)`

Gets the series with `get_pulses`. Returns `max(time) - min(time)` over all
pulses, or `None` if the series is missing or empty.

---

## `oscnext_l4/tray_io.py`

**What this file is for.** The two ends of the tray that compute nothing:
checking the input files before the run, and writing the HDF5 output. Also
the list of libraries needed to unpack frame objects.

**Where it sits.**
- `load_deserialization_libs`: called by `variables.py` on import, by
  `scripts/process_L4.py`, and by `verification/compare_pass2.py`.
- `validate_files`: used by `scripts/process_L4.py` (`--scan`) and
  `scripts/scan_files.py`.
- `add_booker`: used by `scripts/process_L4.py` (last module of the tray) and
  `verification/compare_pass2.py` (`book`).
- It imports `icecube.dataio` at module level and `icecube.hdfwriter` inside
  `add_booker`.

**Constants.**

| name | value | meaning |
|---|---|---|
| `_DESERIALIZE_LIBS` | `("simclasses", "recclasses", "genie_icetray", "genie_reader", "sim_services", "phys_services")` | IceTray libraries whose classes may appear in a frame. Without them, reading such a frame object fails with "Deserialization failed". |

### `load_deserialization_libs()`

Tries to import `icecube.<lib>` for each library in `_DESERIALIZE_LIBS`.
Libraries that are not installed are skipped silently. Returns the list of
names that did import.

- Side effect: registers those libraries' classes with IceTray.

### `validate_files(paths, n_frames=25, verbose=True)`

Checks input files before the run. `I3Reader` reads the whole file list in one
go, so a single broken file kills the whole tray; this finds them first.

Steps, per file:
1. If the file size is 0, or the size cannot be read, mark it bad.
2. Open it with `dataio.I3File` and read frames with `pop_frame` until the
   file ends or `n_frames` frames have been read (`n_frames=0` reads the whole
   file). Close the file.
3. If any exception occurs, mark it bad with the first line of the error (up
   to 160 characters). Otherwise mark it good.
4. With `verbose`, print progress every 100 files.

Returns `(good, bad)`: `good` is a list of paths, `bad` a list of
`(path, reason)` pairs.

- Frames of every type count towards `n_frames` (G, C, D, Q, P, ...).
- A quick scan does not prove a file is sound; damage in the middle is only
  found by a full scan. In practice, the broken files seen were truncated and
  fail early.

### `add_booker(tray, name, output, keys, sub_event_streams=("InIceSplit",))`

Adds the HDF5 writer: `hdfwriter.I3HDFWriter` with `Output=output`,
`Keys=keys`, `SubEventStreams=list(sub_event_streams)`.

- Writes: the HDF5 file `output`. For every booked key, hdfwriter writes a
  data table `/<key>` and an index table `/__I3Index__/<key>` (one row per
  frame, with `exists`, `start`, `stop`).
- Why it is a function: the index tables are the only reliable way to match a
  table row to its event, because `(Run, Event, SubEvent)` is not unique
  across the several L3 files one output part can contain. `data.py` and
  `verification/pass2.py` both depend on them. Whatever replaces this writer
  must keep writing that index.
- Watch out: importing `hdfwriter` in IceTray v1.17.0 prints a deprecation
  warning pointing to `tableio`. The module comment explains that the two are
  layers: the HDF5 back end (`I3HDFTableService`) lives inside hdfwriter, so
  switching to `tableio` would not remove the hdfwriter dependency.

---

## `oscnext_l4/classifier.py`

**What this file is for.** It applies the trained classifiers inside the tray.
It replaces the production's `icecube.oscNext.tools.classifier.I3Classifier`,
which is not available. For each Physics frame it reads the model's input
variables from the frame, runs the LightGBM model, and writes the score as an
`I3Double`. It also has a segment that runs both classifiers and writes the
combined L4 cut flag. It needs only `lightgbm` and `numpy`; the model is
LightGBM's native text file plus a JSON sidecar, because the IceTray
environment has no sklearn or joblib.

**Where it sits.**
- Used by `variables.compute_L4_cut` (via `add_L4_classifiers`), and so by
  `process_L4.py --apply-cut`. The notebook shows `add_L4_classifiers` as a
  usage example.
- It imports `FEATURE_MAP` and `COLUMN_ALTS` from `oscnext_l4/varmap.py`
  (built from `config/variables.json`), and `L4_CUT_BOOL_KEY` from
  `variables.py`. `lightgbm` is imported inside `load_model`.
- Model files come from `scripts/train_L4_classifier.py`, which writes
  `L4_<tag>_model.txt` and `L4_<tag>_model.json`.

**Constants / configuration.**

| name | value | meaning |
|---|---|---|
| `FEATURE_MAP` | from `varmap` | model variable name -> `(frame key, field)`, e.g. `"micro_count"` -> `("L4_micro_count", "STW_m3500p4000_DTW200")`, `"cog_z"` -> `("SRTTWSplitInIcePulsesDCHitStatistics", "cog.z")` |
| `COLUMN_ALTS` | from `varmap` | `(frame key, field)` -> list of alternative field names, e.g. `("L4_iLineFitParams", "lf_vel")` -> `["LFVel", "speed"]` |
| `FAIL_AFTER` | `100` | raise if a model input is missing in every one of the first 100 Physics frames |
| `MIN_FRAMES_TO_JUDGE` | `30` | in a run shorter than `FAIL_AFTER`, only raise at the end if there were at least 30 frames |
| `CUT_KEY` | `L4_CUT_BOOL_KEY` = `"L4_oscNext_bool"` | the final cut flag (imported, not redefined) |
| `NOISE_KEY` | `"L4_NoiseClassifier_ProbNu"` | noise score key |
| `MUON_KEY` | `"L4_MuonClassifier_Data_ProbNu"` | muon score key |

### `_get_col(obj, col)`

Reads one field `col` from a frame object `obj`. Returns a float, or `None`
if the field is not there.
1. A dotted name walks attributes: `"cog.z"` -> `obj.cog.z` (recursively).
2. A map-like object (has `.keys`): `float(obj[col])` if `col` is a key,
   else `None`. (This covers `I3MapStringDouble`/`I3MapStringInt`.)
3. `col == "value"`: `float(obj.value)` (for `I3Double`, `I3Int`, ...).
4. Any other attribute: `float(getattr(obj, col))`.
5. Otherwise `None`.

### `read_feature(frame, name)`

Reads one model input from the frame by its variable name. Returns NaN if the
name is not in `FEATURE_MAP`, if the frame key is absent, or if no field name
works.

Steps: look up `(key, col)` in `FEATURE_MAP`; get `frame[key]`; try `col`,
then each alternative in `COLUMN_ALTS[(key, col)]`, with `_get_col`. Errors
in a try (`KeyError`, `AttributeError`, `TypeError`, `ValueError`) count as
"not found". The first value that is not `None` is returned.

### `load_model(model_file)`

Loads a model. `model_file` is the `.txt` path. Returns
`(booster, features, sidecar)`.

Steps:
1. If the file does not exist, raise `IOError`.
2. Load `lightgbm.Booster(model_file=...)`.
3. Look for the `.json` with the same base name. If present, load it and take
   `features` from its `"features"` list. If absent, log a warning and use the
   booster's own feature names; `sidecar = {}`.
4. If the booster has feature names and they differ from `features` (names or
   order), raise `ValueError`. The feature order is the model's input order,
   so a mismatch would give silently wrong scores.
5. If any feature is not in `FEATURE_MAP`, raise `KeyError` (it must be added
   to `config/variables.json`).

### `class L4Classifier(icetray.I3ConditionalModule)`

Applies one trained model to every Physics frame. Being an
`I3ConditionalModule`, it accepts the standard `If=` parameter.

Parameters:

| parameter | default | meaning |
|---|---|---|
| `ModelFile` | `None` (required) | path to `L4_<tag>_model.txt` |
| `OutputKey` | `None` (required) | frame key for the score |
| `MissingValue` | `NaN` | value used for a missing input (LightGBM handles NaN itself) |
| `SkipIfIncomplete` | `False` | if True, write no score when **every** input is missing |
| `FailAfter` | `100` | raise if an input was missing in every one of the first N Physics frames; 0 disables |

#### `__init__(self, context)`
Registers the five parameters and the `OutBox`.

#### `Configure(self)`
Reads the parameters; raises `ValueError` if `ModelFile` or `OutputKey` is
empty. Calls `load_model`. Sets up the per-feature missing counters
(`n_missing`), the frame counter and the `checked` flag. Prints a short
summary: model path, number of trees and features, and, if the sidecar has
them, the training time, LightGBM version and default cut.

#### `Physics(self, frame)`
1. If `OutputKey` is already in the frame, push it unchanged.
2. Build a 1 x N input row: `read_feature` for each feature, in model order.
   A NaN value counts as missing and is replaced by `MissingValue`. (Only NaN;
   an infinite value is passed through as it is, as in training.)
3. Count the frame. When the count reaches `FailAfter` (and no check has run
   yet), call `_fail_if_never_seen`.
4. If `SkipIfIncomplete` and every input was missing, push the frame without a
   score.
5. Otherwise `booster.predict(x)[0]` gives the probability of the positive
   class (label 1 = signal in training), written as `I3Double` to
   `OutputKey`. Push the frame.

- Reads: the frame keys in `FEATURE_MAP` for this model's features. Writes:
  `OutputKey`.

#### `_fail_if_never_seen(self)`
Marks the check as done. If any feature has been missing in every frame seen
so far, raises `RuntimeError` naming them. Such a feature is a wrong key or
field name, not an event property, and every score would be silently wrong.

#### `Finish(self)`
At the end of the run:
1. If the check never ran (the run had fewer than `FailAfter` frames) and
   there were at least 30 frames, run `_fail_if_never_seen` now (may raise).
2. If there were fewer than 30 frames, only print a warning for
   always-missing features.
3. Print a report of every feature that was missing at least once: count,
   percentage, and a flag when it was missing in more than 99.9% of frames.

- Watch out: some inputs are legitimately missing in some events (for
  example `accumulated_time` for 4 or fewer cleaned DOMs), so the checks only
  fire when an input is missing in **every** frame.

### `add_L4_classifiers(tray, name, model_dir, noise_cut=0.70, muon_cut=0.65, apply_cut=True)`

Tray segment. Adds both classifiers and, optionally, the combined cut flag.

Modules added, in order:
1. `L4Classifier` (`<name>_noise`), `ModelFile=<model_dir>/L4_noise_model.txt`,
   `OutputKey="L4_NoiseClassifier_ProbNu"`.
2. `L4Classifier` (`<name>_muon`), `ModelFile=<model_dir>/L4_muon_model.txt`,
   `OutputKey="L4_MuonClassifier_Data_ProbNu"`.
3. **If `apply_cut`:** the nested function module `overall_cut`
   (`<name>_overall_cut`).

The cut values 0.70 and 0.65 are the technical note's (v00.07) values, which
the production also uses.

- Watch out: the file names are fixed (`L4_noise_model.txt`,
  `L4_muon_model.txt`), so the training tags must be `noise` and `muon`.
  Both models must exist.

#### nested `overall_cut(frame)`

Writes `L4_oscNext_bool` as an `I3Bool`: `True` when
`noise score >= noise_cut` **and** `muon score >= muon_cut`. If either score
is missing, writes `False`. Always returns `True`, so **no frame is
dropped**; the flag is only recorded.

---

## Glossary of tray terms

**Tray (I3Tray).** The IceTray processing chain. Frames are read by
`I3Reader` and passed through the modules in the order they were added.

**Frame and frame types.** A frame is a container of named objects. Its
*stream* (type) is one letter:
- **G** Geometry (DOM positions: `I3Geometry`), **C** Calibration,
  **D** Detector status. Together "GCD". Their contents are visible in the
  frames that follow them ("frame mixing"), which is why the rewrites can read
  `frame["I3Geometry"]` from a Physics frame.
- **Q** DAQ: one per triggered readout. Holds the raw data and
  `I3TriggerHierarchy`.
- **P** Physics: one per *sub-event*. The event splitter cuts one Q frame
  into one or more P frames; the sub-event stream used here is `InIceSplit`.
  Almost all L4 work happens on P frames.
- **S** Simulation info: per-file simulation metadata such as `I3GenieInfo`.
  The code comments also mention **M** frames (another per-file metadata
  stream); `process_L4.py` writes both S and M frames to the `.i3` output.

**Frame key.** The name an object is stored under in a frame, e.g.
`"L4_micro_count"`. `key in frame` tests presence; `frame[key] = obj` writes.
A key can be written only once per frame (hence the many
`if output_key in frame: return` guards).

**Pulse series vs mask.** A *pulse series map*
(`I3RecoPulseSeriesMap`) maps each DOM (OMKey) to the list of its reconstructed
pulses, each with `time` (ns), `charge` (PE), `width` and `flags`, in time
order. A *mask* (`I3RecoPulseSeriesMapMask`) stores only which pulses of a
base map are kept, plus the base map's key. It must be *applied*
(`mask.apply(frame)`) to get a real map; `get_pulses` does that. Both L3
series used here are masks.
- **Uncleaned series:** `SplitInIcePulses` (all pulses of the sub-event).
- **Cleaned series:** noise-cleaned (SRT + time window) DeepCore series:
  `SRTTWSplitInIcePulsesDC` in pass3, `SRTTWOfflinePulsesDC` in pass2.

**DOM / OMKey.** A DOM (Digital Optical Module) is one sensor. An `OMKey` is
its address: `(string, om)` plus a PMT number. Pulse maps are sorted by
OMKey.

**HLC / SLC.** A *hard local coincidence* hit: the DOM's neighbour (nearby on
the same string) also saw light within a short time window. The pulse's `LC`
flag is set (`_LC_FLAG` in `frame_objects.py`). Pulses without it are *soft
local coincidence* (SLC) pulses, more often noise.

**Trigger config id 1010 / 1011.** Each trigger in `I3TriggerHierarchy` has a
key; for the simple-multiplicity triggers this includes a config id. 1011 is
the DeepCore SMT3 trigger (3 HLC hits in DeepCore in a short window). The
production's `I3StaticTWC` and `I3CutL7Module` settings list `[1010, 1011]`
as the DeepCore trigger ids; pass2 simulation contains no 1010 trigger. The
code does not document what 1010 is. MERGED and THROUGHPUT triggers carry no
config id, and reading one raises, so `_vich` guards it.

**DeepCore fiducial / veto DOMs.** From `DeepCore_Filter.DOMS.DOMS("IC86")`
(`deepcore_doms()`): `DeepCoreFiducialDOMs` (554 DOMs, the DeepCore volume)
and `DeepCoreVetoDOMs` (the other 4606 in-ice DOMs). Together they are the
whole 5160-DOM in-ice detector, so "veto" means "not fiducial". Micro count
keeps only fiducial DOMs; VICH uses no DOM list at all.

**rho_36.** Horizontal distance from string 36, the centre of DeepCore
(x = 46.29 m, y = -34.88 m).

**I3Double, I3Int, I3Bool.** Frame objects holding one number or flag, read
with `.value`. They book to a one-column HDF5 table (`value`).

**I3MapStringDouble / I3MapStringInt / I3MapStringBool.** Frame objects that
map string keys to numbers (or flags), read like a dict. Example:
`IC2018_LE_L3_Vars` (L3 variables, `I3MapStringDouble`) and `L4_micro_count`
(`I3MapStringInt` with one key).

**Other frame object types seen here.** `I3Particle` (a position, direction,
time, shape, fit status; used for `L4_first_hlc` and fits),
`I3LineFitParams` (LineFit parameters incl. speed), `I3FillRatioInfo`
(fill-ratio results), `I3HitStatisticsValues` (COG, z spread, z travel, ...),
`I3HitMultiplicityValues` (`n_hit_doms`, ...), `I3GenieInfo` (GENIE
generation info per file).

**Tray segment vs module vs function module.**
- A **module** is a class derived from `icetray.I3Module` (C++ or Python).
  It is added by class or by registered name, e.g.
  `tray.AddModule("I3OMSelection<I3RecoPulseSeries>", ...)`. It can handle
  frames of any stream. `PropagateGenieInfo` and `L4Classifier` are Python
  modules. An **I3ConditionalModule** also accepts `If=` to run only on some
  frames.
- A **function module** is a plain Python function `f(frame, **params)` added
  with `tray.Add(f, name, **params)`. By default it runs on Physics frames
  only. Returning `True` keeps the frame; returning `False` drops it. All the
  rewrites (`_first_hlc`, `_vich`, ...) are function modules that always
  return `True`; `l3_cut` returns `False` to drop events.
- A **tray segment** is a function decorated with `@icetray.traysegment`
  that takes `(tray, name, ...)` and adds several modules. It does not
  process frames itself; it only builds part of the chain when the tray is
  configured. `oscNext_L4` and the `oscNext_L4_*` functions are segments.

**Booking.** Writing chosen frame keys to HDF5 tables with hdfwriter
(`tray_io.add_booker`). Only Physics frames of the given sub-event stream are
booked.


---

# Part B: Running the processing (entry points, drivers and environment)

This part covers the code that runs *around* the IceTray tray (the chain of
modules that processes frames one by one). The tray itself, meaning the L4
variables, is covered in another part. Here you find out:

- how one L3 -> L4 job is started, what it writes, and how it survives broken
  input files (`scripts/process_L4.py`);
- how the notebook starts many such jobs at once and draws progress bars
  (`oscnext_l4/runner.py`);
- the helper scripts for scanning input files, checking the environment and
  copying the production sources (`scripts/scan_files.py`,
  `scripts/diagnose_env.py`, `scripts/collect_meta.sh`);
- how to get into the IceTray environment (`setup_env.sh`).

A few words used throughout:

- **Frame**: one record in an `.i3` file. `G`, `C`, `D` frames hold the
  detector geometry, calibration and status (together called the **GCD**).
  `Q` (DAQ) frames hold one triggered readout. `P` (Physics) frames are the
  sub-events split out of a `Q` frame. Only `P` frames become rows in the HDF5
  file.
- **Sub-event stream**: the name of the splitter that made a `P` frame
  (`I3EventHeader.sub_event_stream`). The analysis uses `InIceSplit`.
- **Booking**: writing chosen frame objects ("keys") into HDF5 tables. It is
  done by `hdfwriter`, which also writes an index table
  `/__I3Index__/<key>` for every key.
- **Part**: with `--chunk-files N`, the input list is cut into groups of N L3
  files, and each group is processed by its own tray into its own output
  file, `..._partNNN.hdf5`.
- **Anchor**: the output that the side files are named after. It is the HDF5
  file when one is requested, and the `.i3` file otherwise.

The data flow for this part:

```
setup_env.sh  (get into the IceTray environment)
      |
      v
notebook section 1 --> oscnext_l4/runner.py --(subprocess, one or many)--> scripts/process_L4.py
                                                                              |
      scripts/scan_files.py --good-list  --(optional healthy-file list)------>|
                                                                              v
                                               L4_<sample>[_jobJ|_RunNNN][_partNNN].hdf5
                                               + .meta.json  + .badfiles.txt
```

---

## `scripts/process_L4.py`

### What this file is for

This is the main L3 -> L4 program. It reads L3 `.i3` files together with a
GCD file, runs the oscNext L4 tray segment on every event, and writes the
result as an HDF5 file (for training), an `.i3` file (the L4 data itself), or
both. It also protects the run against corrupt input files, can cut a long
input list into resumable parts, and prints machine-readable progress lines
that the notebook turns into progress bars.

### Where it sits

- **Who runs it:**
  - `oscnext_l4/runner.py` starts it as a subprocess (`python -u
    process_L4.py ...`) from the notebook `notebooks/oscNext_L4.ipynb`,
    section 1.
  - People run it by hand from the command line.
  - `verification/compare_pass2.py plan` prints `process_L4.py` command lines
    for the pass2 cross-check, and `verification/pass2_verification.ipynb`
    runs it as a subprocess, one L3 file per output.
  - `scripts/make_release.py` treats it as one of the entry scripts of a
    release.
- **What it imports and calls:**
  - `icecube.icetray`, `icecube.dataio`, `icecube.dataclasses`, and
    `I3Tray` from `icecube.icetray`.
  - `oscnext_l4.tray_io`: `load_deserialization_libs()` (called at import
    time, so that frame objects from simulation libraries can be read),
    `validate_files()` (the pre-scan), and `add_booker()` (the HDF5 writer).
  - `oscnext_l4.variables`: the `oscNext_L4` tray segment, the list
    `L4_HDF5_KEYS`, the key names `HITSTAT_KEY` / `HITMULT_KEY`, and the
    default pulse-series names `UNCLEANED_PULSES_DEFAULT`
    (`SplitInIcePulses`) and `CLEANED_PULSES_DEFAULT`
    (`SRTTWSplitInIcePulsesDC`, the pass3 name).
  - `lightgbm` is imported only as a check when `--apply-cut` is given.
- The script adds the repository root to `sys.path`, so it works from any
  working directory.

### Command-line flags

All flags, in the order argparse defines them.

| Flag | Default | What it does |
|---|---|---|
| `--gcd PATH` | *(required)* | The GCD file. It is put first in the reader's file list, in front of the L3 files. |
| `--input PATTERN [PATTERN ...]` | `[]` | Input `.i3` files. Each entry is expanded as a glob and sorted. An entry that matches nothing is kept as a literal path. You need `--input` or `--input-list` (or both; they are appended). |
| `--output-i3 PATH` | none | Write an L4 `.i3` file (for example `.i3.zst`): the L3 frames plus every L4 key. Physics frames that fail the L3 cut or are not in `--sub-event-stream` are not written. |
| `--output-hdf5 PATH` | none | Write an HDF5 file with the booked keys, for training. At least one of `--output-i3` / `--output-hdf5` is required. |
| `--uncleaned-pulses NAME` | `SplitInIcePulses` | The uncleaned pulse series (used by VICH and, by default, by `micro_count`). |
| `--cleaned-pulses NAME` | `SRTTWSplitInIcePulsesDC` | The cleaned pulse series. This is the pass3 name. **For pass2 L3 give `SRTTWOfflinePulsesDC`.** |
| `--sub-event-stream NAME` | `InIceSplit` | Physics frames from any other sub-event stream are dropped. Also passed to the HDF5 writer. |
| `--mc` | off | Simulation: book the MC truth keys (`MC_KEYS`, see below). |
| `--noise` | off | Pure-noise (vuvuzela) MC: book `NOISE_MC_KEYS` (`noise_weight`). |
| `--muongun` | off | MuonGun MC: book the muon weight keys (`MUONGUN_KEYS`). |
| `--genie` | off | GENIE MC. Adds the module that copies `I3GenieInfo.n_flux_events` into every Physics frame (`L4_n_flux_events`). **Implies `--mc`.** |
| `--corsika` | off | CORSIKA MC: book `CORSIKA_KEYS` (`CorsikaWeightMap`, `PolyplopiaPrimary`, ...). |
| `--no-l3-cut` | off | Do not apply the L3 cut inside the segment. Useful to test why nothing gets booked. |
| `--apply-cut` | off | Run both trained classifiers and write their scores and `L4_oscNext_bool` into every event. No event is dropped. Needs `--model-dir`. |
| `--model-dir DIR` | none | Directory holding `L4_noise_model.txt`, `L4_noise_model.json`, `L4_muon_model.txt`, `L4_muon_model.json`. Only allowed together with `--apply-cut`. |
| `--n N` | `0` | Process only N **frames** (not events), then stop. 0 means everything. Meant for smoke tests. Cannot be combined with `--chunk-files`. |
| `--input-list FILE` | none | Read input paths from a text file, one per line. Blank lines and lines starting with `#` are ignored. Designed for the `scan_files.py --good-list` output. |
| `--scan {quick,full,off}` | `quick` | Pre-scan of the input files. `quick` reads the first `--scan-frames` frames of each file; `full` reads every frame (slow but certain); `off` skips the scan. |
| `--scan-frames N` | `25` | Frames read per file in `--scan quick`. |
| `--run-optional` | off | Also compute the variables that are not BDT inputs: `I3TensorOfInertia` (`L4_ToI`) and `separation_in_cogs`. Costs time only. |
| `--accumulated-time-pass2` | on (default) | `accumulated_time` follows the pass2 production's charge-quartile rule. Giving the flag changes nothing, because it is already the default. |
| `--accumulated-time-note` | off | `accumulated_time` follows the technical note instead (the pulse at the 75 % crossing). Sets the same internal switch as the flag above to False. |
| `--micro-count-uncleaned` | on (default) | The `micro_count` chain starts from the uncleaned series, as the pass2 code does. Giving the flag changes nothing. |
| `--micro-count-cleaned` | off | The `micro_count` chain starts from the cleaned series, as Table 11 of the note says. |
| `--usage` | off | After each tray finishes, print the CPU time per module (top 25). |
| `--progress N` | `5000` | Print a `[PROGRESS]` line every N Physics frames. 0 turns it off. The notebook's bars are drawn from these lines. |
| `--chunk-files N` | `0` | Process the input in parts of N files. Each part is its own tray and its own `<output>_partNNN.hdf5` (and/or `_partNNN.i3.zst`). Finished parts are skipped when you rerun, so a crashed run resumes. |
| `--overwrite` | off | With `--chunk-files`: redo parts that are already finished. |
| `--retries N` | `3` | How many times a file that turns out corrupt *during* the run may be dropped and the tray restarted. 0 means no retry. |
| `--no-hit-statistics` | off | Do not compute `HitStatistics` / `HitMultiplicity`. This leaves `cog_z`, `z_sigma`, `z_travel` and `n_hit_doms` missing in every event, so it is almost never what you want. |

**Checks done before anything runs** (in `main`, each ends with an argparse
error, exit code 2):

1. `--input` or `--input-list` must be given.
2. At least one of `--output-i3`, `--output-hdf5` must be given.
3. `--model-dir` without `--apply-cut` is refused (otherwise you would get
   an L4 file without scores and not notice).
4. `--apply-cut` needs `--model-dir`, all four model files in it, and a
   working `import lightgbm`.
5. `--n` together with `--chunk-files` is refused.

### Example command lines

pass3 GENIE nue, full production in parts of 10 files (the paths are the
ones in `config/productions.json`):

```bash
python scripts/process_L4.py \
    --gcd /cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz \
    --input "/data/ana/LE/oscNext/pass3/genie/level3/23800/*.i3.zst" \
    --output-hdf5 L4_output/hdf5/nue/L4_nue.hdf5 \
    --mc --genie --chunk-files 10
```

pass2 GENIE nue. The only pass2-specific flag is `--cleaned-pulses`:

```bash
python scripts/process_L4.py \
    --gcd /data/ana/LE/oscNext/pass2/genie/level4/121122/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz \
    --input "/data/ana/LE/oscNext/pass2/genie/level3/121122/oscNext_genie_level3_v02.00_pass2.121122.*.i3.zst" \
    --cleaned-pulses SRTTWOfflinePulsesDC \
    --output-hdf5 L4_output/hdf5_pass2/nue/L4_nue.hdf5 \
    --mc --genie --chunk-files 10
```

Smoke test on one file: 200 frames, no pre-scan (this is what
`runner.run_process("nue", n_frames=200)` runs):

```bash
python scripts/process_L4.py \
    --gcd /cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz \
    --input /data/ana/LE/oscNext/pass3/genie/level3/23800/genie_NuE_IC86.023800.000000.i3.zst \
    --output-hdf5 L4_output/hdf5/nue/L4_nue_smoke.hdf5 \
    --mc --genie --n 200 --scan off
```

All of these must run inside the IceTray environment, for example with
`./setup_env.sh run python scripts/process_L4.py ...`.

### What the tray contains, in order

`main` builds the tray inside a nested factory function, `build_tray(files)`,
so that it can be built again with a shorter file list (an `I3Tray` cannot be
executed twice). Each tray contains, in this order:

1. **`I3Reader`**, with `FilenameList = [--gcd] + files`.
2. **`count_physics`**: counts every Physics frame.
3. **`progress`** (only if `--progress` > 0): a `ProgressReporter` that
   prints a `[PROGRESS]` line every N frames it sees.
4. **`stream_filter`**: a lambda that keeps a Physics frame only if
   `I3EventHeader.sub_event_stream == --sub-event-stream`. Other Physics
   frames are dropped from the rest of the tray, including the `.i3` output.
5. **`count_stream`**: counts the Physics frames that passed the filter.
6. **`oscNext_L4`** (the tray segment from `oscnext_l4/variables.py`), with
   the pulse names and all the switches from the command line. In short, it
   adds: the `I3GenieInfo` propagation (if `--genie`); the L3 cut (unless
   `--no-l3-cut`), which drops Physics frames that fail L3; a check that
   both pulse series are present (it raises after 20 frames if one never
   appears, which catches a pass2 run without `--cleaned-pulses`); the common
   variables; the hit statistics (unless `--no-hit-statistics`); the noise
   variables; the muon variables; and, with `--apply-cut`, the two
   classifiers and `L4_oscNext_bool`. See the part of the guide on
   `variables.py` for details.
7. **`counter`**: counts the Physics frames that survived everything. This
   is the number reported as "booked".
8. **`I3Writer`** (only with `--output-i3`). It writes the TrayInfo, DAQ,
   Physics, `S` and `M` streams. It does **not** write G, C or D frames. It
   uses `DropOrphanStreams=[DAQ]`, so a `Q` frame whose `P` frames were all
   dropped is not written either.
9. **The HDF5 booker** (only with `--output-hdf5`): `tray_io.add_booker`,
   which adds `hdfwriter.I3HDFWriter` with the key list below and
   `SubEventStreams=[--sub-event-stream]`.

Both writers are given the temporary `_incomplete_` path, not the final
name (see below).

### The key lists per sample type

`build_key_list` always books:

- `BASE_KEYS`: `I3EventHeader`.
- `L3_KEYS`: `IC2018_LE_L3_Vars`, `IC2018_LE_L3_bools`, `L3_oscNext_bool`,
  `Data_quality_bool`. Several BDT inputs (for example `NchCleaned`,
  `ICVetoHits`) live in `IC2018_LE_L3_Vars`.
- `COMMON_VAR_KEYS`: `SRTTWSplitInIcePulsesDCHitStatistics` and
  `SRTTWSplitInIcePulsesDCHitMultiplicity`. These names always carry the pass3
  spelling, even on a pass2 run (the values are computed from whatever series
  was passed).
- `L4_HDF5_KEYS` (20 keys from `variables.py`): `L4_oscNext_bool`,
  `L4_first_hlc`, `L4_first_hlc_rho`, `L4_ToI`, `L4_ToIParams`,
  `L4_iLineFit`, `L4_iLineFitParams`, `L4_QR_Box`, `L4_VICH_nch`,
  `L4_VICH_npulses`, `L4_VICH_qtot`, `L4_separation_in_cogs`,
  `L4_accumulated_time`, `L4_micro_count`, `L4_fill_ratio`,
  `L4_FullTimeLengthRatio`, `L4_n_flux_events`, `L4_NoiseStraightCuts_Bool`,
  `L4_NoiseClassifier_ProbNu`, `L4_MuonClassifier_Data_ProbNu`.

That is 27 keys. Then exactly **one** sample-type list is added. The flags
are checked in this order, and the first one that is set wins:

| Flag set | List added | Keys |
|---|---|---|
| `--noise` | `NOISE_MC_KEYS` | `I3MCWeightDict`, `noise_weight` |
| `--corsika` | `CORSIKA_KEYS` | `CorsikaWeightMap`, `PolyplopiaPrimary`, `PolyplopiaInfo`, `I3PrimaryInjectorInfo`, `I3CorsikaInfo` |
| `--muongun` | `MUONGUN_KEYS` | `I3MCWeightDict`, `MuonWeight`, `MuonWeight_GaisserH4a`, `MuonGunWeight` |
| `--mc` (or `--genie`) | `MC_KEYS` | `I3MCWeightDict`, `NEvPerFile`, `I3GenieSystWeightDict`, `MCInIcePrimary`, `MCDeepCoreStartingEvent`, `MCExtraTruthInfo` |
| none (detector data) | nothing | |

So a GENIE run books 33 keys. A key that is missing from a frame is simply
not booked for that frame. Duplicates are removed, keeping the first
occurrence.

Note that `--muongun --mc` (the pass2 MuonGun setting in
`config/productions.json`) books only `MUONGUN_KEYS`, not `MC_KEYS`, because
the `elif` chain stops at the first match. Only `--genie` changes the tray
itself; `--mc`, `--noise`, `--corsika` and `--muongun` change only the key
list.

The `noise_weight` unit differs between productions (1/ns in pass3, Hz in
pass2). This script books the value as it is; the conversion happens when the
data is loaded (`oscnext_l4/data.py`).

### Output files and their names

For an output path `L4_nue.hdf5` (the same rules apply to the `.i3` output):

| File | When | What |
|---|---|---|
| `_incomplete_L4_nue.hdf5` | while the tray runs | The file is written under this name in the same directory. It is renamed to the real name only when the tray has finished without error. On any error, including Ctrl-C, it is deleted. If the process is killed hard (SIGKILL, walltime) it stays behind, but no glob that looks for finished output matches it, and the next run of the same output overwrites it. |
| `L4_nue.hdf5` | success, no `--chunk-files` | The finished output. |
| `L4_nue_part003.hdf5` | success, with `--chunk-files` | Part 3 (numbering starts at 000, at least three digits). For `.i3` outputs the compression suffix stays last: `L4_nue.i3.zst` becomes `L4_nue_part003.i3.zst`. |
| `<anchor>.meta.json` | after each successful tray | A JSON sidecar, written **last**. Its presence marks a part as finished. |
| `<anchor>.badfiles.txt` | when corrupt inputs are found | One line per dropped file: `path<TAB>reason`. The file is appended to, never truncated. |

The anchor is the HDF5 path if there is one, otherwise the `.i3` path. In
chunk mode, the per-part meta file is named after the part (for example
`L4_nue_part003.hdf5.meta.json`).

**Fields of `.meta.json`, single-run mode (no `--chunk-files`):**

- `n_l3_files`: how many L3 files were actually used (after dropping corrupt
  ones). **This is the divisor for the weight normalisation** that
  `oscnext_l4/data.py` reads. With `--n` it is `null`, because the tray
  stopped early and it is unknown how many files were read.
- `n_l3_files_unreliable`: `true` when `--n` was given.
- `n_l3_files_given`: how many files the tray was given.
- `l3_files_given`: the file list the tray was given.
- `l3_files`: the file list actually used.
- `physics_frames`: all Physics frames read.
- `sub_event_stream`: the stream kept.
- `after_stream_filter`: Physics frames in that stream.
- `booked`: Physics frames that survived the L3 cut.
- `n_frames_limit`: the value of `--n`.
- `elapsed_s`: wall time in seconds.

**Fields in chunk mode:** the same, except that `n_l3_files_unreliable` and
`n_frames_limit` are absent (`--n` cannot be used here), and there are two
extra fields: `chunk_index` and `n_chunks`. The counters are for this part
alone. `elapsed_s` is measured from the start of the whole run, not of this
part.

### The two layers of protection against corrupt input

Some production `.i3.zst` files are truncated. When `I3Reader` reaches one it
fails with `Error reading <file> at frame N: input stream error!`, and that
kills the whole tray, because `I3Reader` reads the whole file list as one
stream.

1. **Pre-scan** (`--scan quick` by default). Before any tray is built,
   `tray_io.validate_files` opens every file and reads its first
   `--scan-frames` frames (`--scan full`: all frames). Empty files and files
   that cannot be read are dropped. The first 10 are printed, all of them go
   to `<anchor>.badfiles.txt` with the reason `pre-scan: quick` (or `full`),
   and the run continues with the rest. If no file is left, the script exits.
   With `--n` and more than 5 files it prints a note that scanning the whole
   list is wasted time, but it still scans.
2. **Retry with a blacklist at run time** (`--retries 3` by default). If the
   tray still raises a `RuntimeError`, `_run_tray` looks for the file name in
   the error text with the pattern `Error reading (\S+?) at frame`. If it
   finds a name that is in the current input list, it:
   1. writes it to the badfiles list (`run time: input stream error`);
   2. removes the file from the input list;
   3. deletes the half-written `_incomplete_` outputs;
   4. builds a new tray and runs again.

   Any other error, or more than `--retries` corrupt files, is re-raised and
   the job fails.

The efficient route for a big set: scan it once with `scripts/scan_files.py
--good-list`, then run with `--input-list <good list> --scan off`.

### `--chunk-files`: parts, resuming, and why a part records its file list

With `--chunk-files N`, the (scanned) input list is cut into consecutive
groups of N files. Each group goes through its own tray into its own
`_partNNN` output. The benefits: the notebook gets a real percentage and ETA,
and a crash loses only the part that was running.

**Resuming.** For each part, in order:

1. If `--overwrite` is not given and the part is finished (`_part_done`:
   every requested output exists under its real name **and** the anchor's
   `.meta.json` exists), the part is skipped. Before skipping, the script
   compares the part's recorded `l3_files_given` with the files this run
   would give it. If they differ, it **exits** with an error. If the part has
   no recorded list (made by an older version), it prints a warning and
   skips anyway.
2. If the output exists but the part is not finished (no meta file, or one
   of the requested outputs is missing), it prints "redoing" and runs the
   part again.
3. Otherwise it runs the part, writes the meta file, and prints a `[CHUNK]`
   line.

**Why a part records its L3 file list.** A part is known only by its index,
and which files belong to index 3 depends on `--chunk-files`, on the input
list, and on the scan result. If any of those changed since the part was
made, skipping it by name would silently drop some L3 files and count others
twice. Comparing `l3_files_given` catches that.

**Stale parts.** Before the loop, `_stale_parts` looks for parts numbered
`n_chunks` or higher (left behind by an earlier run that cut the input into
more parts). They would match the loader's `L4_<sample>*.hdf5` glob and be
loaded twice. If any exist, the script exits and tells you to remove them.

### Progress lines

These lines are printed with an immediate flush, because the notebook reads
them from a pipe:

```
[CHUNK] 3/10 files=30/100 booked=1840 elapsed=312.4
[PROGRESS] frames=15000 physics=7100 booked=4300 elapsed=98.2 rate=152.7
```

- `[CHUNK] done/total files=done/total booked=N elapsed=S`: chunk mode only.
  One line `0/N` before the first part, then one after every part (finished
  or skipped). `booked` is summed over the parts finished in this run.
- `[PROGRESS] frames=... physics=... booked=... elapsed=... rate=...`: every
  `--progress` Physics frames, inside one tray. The counters start from zero
  in each part and after each retry.

At the end, `main` prints a summary: Physics frames, how many were in the
sub-event stream (with a percentage), how many survived the L3 cut, the
number of files processed, the output paths (`_partNNN` in chunk mode), and
any badfiles lists that exist. If Physics frames were read but nothing was
booked, it says where to look: a wrong `--sub-event-stream` if no frame was
in the stream, otherwise the L3 cut (test with `--no-l3-cut`).

### Module-level constants

- `L3_KEYS`, `COMMON_VAR_KEYS`, `BASE_KEYS`, `MC_KEYS`, `NOISE_MC_KEYS`,
  `MUONGUN_KEYS`, `CORSIKA_KEYS`: the key lists above.
- `_BAD_FILE_RE`: the pattern that finds the corrupt file's name in an
  `I3Reader` error.
- `_I3_COMPRESSION = (".zst", ".gz", ".bz2", ".xz")`: suffixes that must stay
  last when a part number is inserted.
- `_INCOMPLETE = "_incomplete_"`: the prefix for unfinished outputs.
- `_WANT_USAGE = {"on": False}`: a module-level switch that `main` sets from
  `--usage`, so `_print_usage` knows whether to print.

### Functions and classes

#### `_emit(line)`

Prints `line` and flushes stdout immediately. Used for the `[CHUNK]` and
`[PROGRESS]` lines, so the notebook sees them as they happen and not when the
job ends.

#### `class ProgressReporter`

A callable object added to the tray as a function module. It counts the
frames it is given and prints a `[PROGRESS]` line every `every` frames.

- **`__init__(self, every, counter, t0)`**: stores the interval, the shared
  counter dict of the tray (`physics`, `stream`, `n`), and the start time.
  Sets its own frame count to 0.
- **`__call__(self, frame)`**: adds 1 to its count. When the count is a
  multiple of `every`, prints
  `[PROGRESS] frames=<its count> physics=<counter["physics"]> booked=<counter["n"]> elapsed=<s> rate=<frames/s>`.
  Always returns `True`, so it never drops a frame.

*Watch out:* it is added without a `Streams` argument, so (as the code's
comment says) it sees Physics frames only, and it sits right after
`count_physics`. So `frames` and `physics` in the line are the same number:
both count Physics frames, not all frames.

#### `_part_path(path, index)`

Inserts `_partNNN` into a file name. Returns `None` for a `None` path.

- `L4_nue.hdf5` -> `L4_nue_part003.hdf5`
- `L4_nue.i3.zst` -> `L4_nue_part003.i3.zst`. If the last extension is one
  of `.zst`, `.gz`, `.bz2`, `.xz`, the extension before it is kept with it,
  so the compression suffix stays last and the name still matches `*.i3.zst`.

#### `_output_kind(path)`

Returns `"HDF5"` if the path ends in `.hdf5` or `.h5`, otherwise
`"I3 file"`. Used only in messages.

#### `_tmp_path(path)`

Returns the path under which `path` is written until its tray finishes: the
same directory, with `_incomplete_` in front of the file name. Returns `None`
for `None`. The extension is kept, because `I3Writer` reads the compression
from it.

#### `_discard(paths)`

Deletes whichever of the given paths exist (it skips `None`). Prints a
warning if a deletion fails. Returns `True` if at least one file was removed.

#### `_part_done(outputs)`

Decides whether a `--chunk-files` part is finished. It drops `None` entries,
then returns `True` only if `<first output>.meta.json` exists **and** every
output exists. The first output is the anchor (HDF5 part if there is one).
Requiring every output means a part first made without `--output-i3` is
redone when you now ask for an `.i3` too.

#### `_part_files_given(output)`

Reads `<output>.meta.json` and returns its `l3_files_given` list. Returns
`None` if the file cannot be read, is not valid JSON, or has no such field.

#### `_stale_parts(path, n_chunks)`

Finds leftover parts of `path` whose number is `n_chunks` or higher.

1. Builds the part-0 name with `_part_path(path, 0)`.
2. Turns it into a regular expression where `_part000` becomes
   `_part(\d{3,})`, anchored at the end (so `.meta.json` files do not
   match).
3. Lists the directory and returns every matching file whose number is
   `>= n_chunks`.

Returns `[]` for a `None` path. It only reports; `main` decides to exit.

#### `_write_meta(output, meta)`

Writes the dict `meta` as indented JSON to `<output>.meta.json`. Does nothing
for a `None` output. Prints a warning (and does not raise) if writing fails.

#### `_print_usage(tray)`

With `--usage` only. Reads `tray.Usage()` (IceTray's own per-module timing),
adds user and system CPU time for each module, sorts from slowest to fastest,
and prints the top 25 with CPU seconds, share of the total, and number of
calls. If the timings cannot be read, it prints a warning. It is called after
each successful `tray.Execute`, so in chunk mode you get one table per part.

#### `_pct(n, total)`

Formats `n/total` as a percentage string such as `"61.2%"`, or `"-"` when
`total` is 0.

#### `_bad_list_path(output)`

Returns `<output>.badfiles.txt` (or `process_L4.badfiles.txt` if `output` is
empty).

#### `_record_bad(output, paths, reason)`

Appends one line `path<TAB>reason` per corrupt file to
`<output>.badfiles.txt`, and prints the list's path. Does nothing for an
empty list. Prints a warning if the file cannot be written.

*Watch out:* the file is opened in append mode. If you rerun the same output,
old entries stay in the list.

#### `_run_tray(build, infiles, output_hdf5, output_i3, retries)`

Runs one tray to completion, with the run-time corrupt-file retry. `build`
is the `build_tray` factory; `output_hdf5` / `output_i3` are the final paths
(either may be `None`).

1. Works out the final paths, the anchor (the first of them), and their
   `_incomplete_` names. It sets `build.output_hdf5` and `build.output_i3`
   (attributes on the factory function) to the temporary names, so the
   writers write there.
2. Loop:
   1. Build a tray from the current file list and run it:
      `tray.Execute(n)` if `build.n_frames` is set (from `--n`), otherwise
      `tray.Execute()`. Then call `_print_usage`.
   2. If it raises `RuntimeError`: look for the corrupt file name in the
      message. If there is no name, the name is not in the input list, or
      the retries are used up, re-raise. Otherwise record the file in the
      badfiles list, remove it from the list, delete the partial outputs,
      print how many files are left, and loop again. If no file is left,
      raise `RuntimeError("Every input file turned out to be corrupt.")`.
   3. On success, rename every temporary output to its final name
      (`os.replace`) and return.
3. On any exception at all (including Ctrl-C), delete the temporary outputs
   and re-raise.

**Returns** `(counter, infiles)`: the tray's counter dict and the list of
files actually used.

*Watch out:* the retry only works if the file name in the error message is
spelled exactly like the entry in the input list. If `I3Reader` reports a
different spelling of the path, the error is re-raised and the job fails.

#### `build_key_list(is_mc=False, is_noise=False, is_muongun=False, is_corsika=False, extra=None)`

Returns the list of keys to book. It joins `BASE_KEYS`, `L3_KEYS`,
`COMMON_VAR_KEYS` and `L4_HDF5_KEYS`, then adds one sample list in the order
noise, corsika, muongun, mc (the first flag that is set wins), then `extra`
if given. Duplicates are removed, order kept. See "The key lists per sample
type" above. `main` never passes `extra`.

#### `main()`

The whole program. In order:

1. Parse the flags (table above). `--genie` sets `--mc`.
2. Run the checks listed under the flag table.
3. Pick the anchor (`--output-hdf5` or else `--output-i3`) and set the
   `--usage` switch.
4. Build the input list: first the lines of `--input-list`, then each
   `--input` pattern, globbed and sorted. A pattern that matches nothing is
   kept as a literal path. Exit if the list is empty.
5. Create the output directories.
6. Pre-scan (unless `--scan off`), write the badfiles list, exit if nothing
   is left.
7. Build the key list and print its length (or say that no HDF5 is written).
8. Define the tray factory `build_tray` (below) and set
   `build_tray.n_frames` from `--n`.
9. Run:
   - **chunk mode:** cut the list into parts, refuse stale parts, print the
     initial `[CHUNK] 0/N` line, then for each part skip it or run it with
     `_run_tray`, write its meta file and print a `[CHUNK]` line;
   - **single mode:** one `_run_tray` over the whole list, then one meta
     file.
10. Print the summary described under "Progress lines".

Exit codes: 0 on success; 2 for a bad flag combination (argparse); 1 for
`sys.exit("...")` messages (no input, no healthy file, stale parts, part
made from a different file list) and for uncaught exceptions from the tray.

#### `build_tray(files)` (nested in `main`)

Builds one fresh `I3Tray` for the given L3 files and returns
`(tray, counter)`. `counter` is a new dict `{"physics": 0, "stream": 0,
"n": 0}` that the counting modules fill. The modules are listed under
"What the tray contains". It reads the output paths from its own attributes
`build_tray.output_i3` / `build_tray.output_hdf5` (set by `_run_tray` to the
`_incomplete_` names), falling back to the command-line paths.

The small functions defined inside it:

- **`_count_physics(frame)`**: adds 1 to `counter["physics"]`; returns
  `True`. Physics frames only.
- **stream filter** (a `lambda`, module name `stream_filter`): returns
  `True` if the frame's `I3EventHeader.sub_event_stream` equals
  `--sub-event-stream`. Physics frames only.
- **`_count_stream(frame)`**: adds 1 to `counter["stream"]`; returns `True`.
- **`count(frame)`** (module name `counter`): adds 1 to `counter["n"]`
  after the `oscNext_L4` segment, so it counts the events that are booked;
  returns `True`.

---

## `oscnext_l4/runner.py`

### What this file is for

This module drives `scripts/process_L4.py` from the notebook. It starts
`process_L4.py` as one or more subprocesses, reads their output line by line,
and turns the `[CHUNK]` / `[PROGRESS]` lines into live progress bars. It has
three ways to run a sample: serially in one process, split across N parallel
workers, and one run at a time with a separate GCD per run (for detector
data). The logic lives in a module rather than in the notebook so it stays
under version control.

### Where it sits

- **Who imports it:** `notebooks/oscNext_L4.ipynb`, section 1:
  `from oscnext_l4.runner import configure_runner, run_process, run_all,
  run_process_per_run`, then `configure_runner(SAMPLES, PROCESS_PY, GCD)`,
  a smoke test `run_process("nue", n_frames=200)`, and
  `run_all(jobs=8, chunk_files=10)`.
- **What it calls:** `scripts/process_L4.py`, always as
  `[sys.executable, "-u", PROCESS_PY, ...]`. The `-u` flag makes Python's
  stdout unbuffered, so the lines arrive while the job runs. It uses only
  the standard library, plus `IPython.display` and `ipywidgets` when they
  are available.
- **What it expects from the notebook:** `SAMPLES`, a dict from sample name
  to a config dict. The keys this module reads are:
  - `l3`: one glob string, or a list of globs;
  - `hdf5`: the output path, for example `<HDF_BASE>/nue/L4_nue.hdf5`;
  - `flags`: the list of `process_L4.py` flags for this sample (for example
    `["--mc", "--genie"]`; the notebook appends `--cleaned-pulses ...` when
    the production names a cleaned series);
  - `gcd` (optional): a **list** of GCD globs, one per `l3` pattern, for
    detector data.

  These come from `config/productions.json` via the notebook.

### Output naming by driver

| Driver | Output passed to `process_L4.py` | With `chunk_files=10` the files are |
|---|---|---|
| `run_process` | `L4_nue.hdf5` (smoke test: `L4_nue_smoke.hdf5`) | `L4_nue_part000.hdf5`, ... |
| `run_process_parallel` | `L4_nue_job<J>.hdf5` per worker | `L4_nue_job0_part000.hdf5`, ... |
| `run_process_per_run` | `L4_data_<RunLabel>.hdf5` per run | `L4_data_Run00120200_part000.hdf5`, ... |

All of them match the loader's glob `L4_<sample>*.hdf5` in
`oscnext_l4/data.py`. Every part has its own `.meta.json`, so the
`n_l3_files` values add up correctly.

The parallel and per-run drivers also write the file list of each worker or
run to `<hdf5 dir>/_filelists/<sample>_job<J>.txt` or
`<sample>_<RunLabel>.txt`, and pass it with `--input-list`.

### Module-level objects

- `display`: `IPython.display.display`, or a do-nothing fallback (below)
  when IPython is not installed.
- `_HAS_W`: `True` if `ipywidgets` imports.
- `_CFG`: the dict that holds `SAMPLES`, `PROCESS_PY` and `GCD` after
  `configure_runner`.
- `_CHUNK_RE`, `_PROG_RE`: regular expressions that parse the `[CHUNK]` and
  `[PROGRESS]` lines of `process_L4.py`. They must stay in step with the
  format strings there.
- `threading` is imported halfway down the file, just before the parallel
  section.

### Functions and classes

#### `display(*a, **k)` (fallback)

Only defined when `IPython.display` cannot be imported. Does nothing, so the
module also works in a plain terminal.

#### `configure_runner(SAMPLES, PROCESS_PY, GCD)`

Stores the notebook's `SAMPLES`, the path of `process_L4.py` and the default
GCD path in `_CFG`. Call it once before anything else. It then checks three
things and prints the problems it finds: `PROCESS_PY` does not exist, `GCD`
does not exist, or the working directory contains `Trash` (a Jupyter server
started from a directory that was later moved to the trash). If all is well
it prints `configure_runner OK (cwd: ...)`. It never raises.

#### `_cfg(key)`

Returns `_CFG[key]`. Raises `RuntimeError` if `configure_runner` has not been
called.

#### `_fmt_eta(sec)`

Formats a number of seconds as `45s`, `3min07s` or `2h05min`. Returns `"?"`
for `None`, NaN or a negative number.

#### `class _Bar`

One progress bar. With `ipywidgets` it is a `FloatProgress` widget plus an
HTML text box next to it; without it, a single text line redrawn with `\r`,
which also works in a terminal.

- **`__init__(self, label, total=None)`**: stores the label, the total
  (which may be set later) and the start time. With widgets, it creates and
  displays the widget (description is the first 12 characters of the label).
- **`update(self, done, note="")`**: with a total, it computes the fraction
  done and an ETA from the elapsed time, and shows
  `done/total  NN%  ETA ...  note`. Without a total it shows only `note`
  (the widget is shown half full, as "unknown"). The widget's maximum is
  updated when `total` was set after construction. In text mode it draws a
  30-character bar of `#` and `-`.
- **`done(self, note="")`**: marks the bar finished (green widget, full), or
  ends the text line with a newline.
- **`fail(self, note="")`**: marks the bar failed (red widget), or prints
  `[ERROR] note` in text mode.

#### `l3_patterns(cfg)`

Returns a sample's `l3` entry as a list. A pass3 sample has one glob string;
a pass2 sample may have several (NuE is datasets 121122 and 121291), and
detector data has one per run.

#### `l3_files(cfg, max_files=0, fraction=0.0)`

Returns every L3 file of a sample: each pattern is globbed and sorted, the
results are joined, and duplicates are removed (order kept).

- `fraction` (0 < f <= 1) keeps a share of **each pattern**, by taking every
  k-th file (`files[::k]` with `k = round(1/f)`). A stride is used rather
  than the first files because files within a run are in time order; the
  stride spreads the slice over the whole run.
- `max_files` > 0 then keeps the first N of the **whole** list.

Raises `ValueError` if `fraction` is outside (0, 1].

Why both exist: for one pattern, `max_files` is fine. For detector data with
18 run patterns ordered by year, `max_files` would take the early years only,
while `fraction` keeps every run represented.

#### `run_process(name, n_frames=0, chunk_files=10, log_tail=15, bar=True, run_optional=False, extra_args=None, max_files=0, fraction=0.0)`

Runs `process_L4.py` for one sample, in one subprocess, and shows a live bar.

1. Output path: `cfg["hdf5"]`, or `..._smoke.hdf5` when `n_frames` > 0.
   Creates the output directory.
2. If `n_frames` > 0, sets `chunk_files` to 0 (they cannot be combined).
3. Inputs: with `max_files` or `fraction`, the explicit file list from
   `l3_files` (it prints how many of how many files are used); otherwise the
   raw `l3` patterns, which `process_L4.py` globs itself.
4. Builds the command: `python -u process_L4.py --gcd <GCD> --input ...
   --output-hdf5 <out> <sample flags>`, plus `--n <n_frames> --scan off` for
   a smoke test, `--chunk-files N` if set, `--run-optional` if set, and
   `extra_args` verbatim. Prints the command.
5. Starts it with stdout and stderr merged, and reads it line by line. It
   keeps the recent lines (up to 400, trimmed to the last 200). A `[CHUNK]`
   line sets the bar's total to the number of parts and moves it. A
   `[PROGRESS]` line moves the note only while the bar has no total (that
   is, without chunking). Lines starting with `Pre-scan`, `  scanned:`,
   `  Files to process`, `Parts:` or `  [!]` are shown in the bar's note.
6. On a non-zero exit code: marks the bar failed, prints the last 40 lines,
   and returns `None`.
7. On success: sums the size of all files matching `L4_<sample>*.hdf5`,
   marks the bar done, prints the last `log_tail` lines and
   `-> <out> (size, file count, seconds)`, and returns the output path.

*Watch out:*
- With chunking, the bar moves only when a whole part finishes.
- The size and file count in the final message come from the glob
  `L4_<sample>*.hdf5`, so they also include `_smoke` files and any
  `_job...` outputs of the same sample in that directory.
- With `max_files` or `fraction`, every file path is put on the command
  line. For a very large slice this is a long argument list; the parallel
  driver uses `--input-list` instead.

#### `run_all(samples=None, chunk_files=10, jobs=1, run_optional=False, extra_args=None, max_files=None, fraction=None)`

Processes several samples one after the other, with one bar per sample and
an overall `TOTAL` bar.

1. `names` is `samples`, or every sample in `SAMPLES`.
2. `max_files` and `fraction` may each be one number for all samples, or a
   dict `{sample: value}`. A dict that names an unknown sample raises
   `ValueError`. Samples not in the dict get no limit.
3. For each sample:
   - if its config has a `gcd` **list**, it goes to `run_process_per_run`
     (with `jobs`, `chunk_files`, `run_optional`, `extra_args`, `fraction`);
   - else if `jobs` > 1, to `run_process_parallel`;
   - else to `run_process`.
4. Counts the samples whose result is not `None`, finishes the overall bar,
   and prints the names of the failed samples.

**Returns** a dict `{sample: output path or None}`.

The default `jobs=1` is serial: parallel running must be asked for.

*Watch out:* `max_files` is not passed to `run_process_per_run`; for a
per-run sample only `fraction` limits the input.

#### `_split(seq, n)`

Splits a list into `n` consecutive groups whose sizes differ by at most one
(the first groups get the extra item). `n` is limited to between 1 and
`len(seq)`. Empty groups are dropped.

#### `run_process_parallel(name, jobs=4, chunk_files=10, log_tail=10, bar=True, run_optional=False, extra_args=None, max_files=0, fraction=0.0)`

Processes one sample with `jobs` parallel `process_L4.py` workers. Each
worker gets a disjoint, consecutive share of the L3 files and writes its own
output, so there is no shared state.

1. Gets the file list with `l3_files(cfg, max_files, fraction)`. Returns
   `None` if it is empty.
2. Splits it into groups with `_split` and prints the group sizes.
3. **Guard against a changed split.** For each worker, if
   `_filelists/<sample>_job<J>.txt` exists from an earlier run, holds a
   different file list, and output `L4_<sample>_job<J>*.hdf5` exists, that
   worker is "stale". If any worker is stale, it raises `RuntimeError`. The
   message says how many parts and L3 files are already on disk, suggests
   leaving a finished sample out of `run_all`, and gives the `rm -rf`
   command to start over. (Changing `jobs` changes which files each worker
   number gets, so resuming would mix old and new splits.)
4. For each worker: writes its file list to
   `_filelists/<sample>_job<J>.txt` and starts `python -u process_L4.py
   --gcd <GCD> --input-list <list> --output-hdf5 L4_<sample>_job<J>.hdf5
   <sample flags>`, plus `--chunk-files`, `--run-optional` and `extra_args`
   as given. Each worker pre-scans its own share (the default `--scan
   quick`).
5. If every worker holds only one part, prints a note that the file count
   will stay at 0 until workers finish, and that the event count shows the
   run is alive.
6. Starts one reader thread per worker (nested `reader`), waits for all of
   them, then collects the exit codes.
7. If any worker failed: marks the bar failed, prints the last `log_tail`
   lines of each failed worker, and returns `None`. Otherwise marks the bar
   done, prints the total size, and returns `cfg["hdf5"]`.

Nested functions:

- **`redraw()`**: sums the finished files and the events (finished parts
  plus the part in flight) over all workers, and updates the bar with
  `<N> workers  events <M>`.
- **`reader(j, p)`**: reads worker `j`'s output line by line and keeps its
  last lines (up to 60, trimmed to 30). A `[CHUNK]` line sets the worker's
  finished file count and cumulative booked count, and resets its in-flight
  count. A `[PROGRESS]` line sets the in-flight booked count. Both redraw the
  bar under a lock. Waits for the process at the end.

*Watch out:*
- With `chunk_files=0` no `[CHUNK]` lines are printed, so the file count on
  the bar never moves; only the event count does.
- Raising `max_files` or changing `fraction` also changes the split, so the
  guard refuses it, just as it refuses a change of `jobs`.
- The message `limited to the first N of M L3 files` is printed for
  `fraction` too, where the files are a stride and not "the first".

#### `run_gcd_pairs(cfg, fraction=0.0)`

For a sample with one GCD per run, pairs each L3 pattern with its GCD.
Returns a list of `(label, gcd_path, files)`.

1. Raises `ValueError` if `cfg["gcd"]` is not a list of the same length as
   the `l3` patterns.
2. For each (L3 pattern, GCD glob) pair: globs and sorts the L3 files, applies
   the `fraction` stride, and globs the GCD. A run is **skipped** if it has
   no L3 file, or if its GCD glob does not match exactly one file (two
   matches would mean the glob catches something else, and taking the first
   could use the wrong detector state).
3. The label is `_run_label(pattern)`, for example `Run00120200`.
4. Prints every skipped run with its reason and both patterns. When the GCD
   glob matched nothing, it also lists up to three files matching `*GCD*` in
   that directory, because the usual cause is a different suffix (for
   example `.i3.gz` instead of `.i3.zst`).

#### `_run_label(pattern)`

Returns the `Run<digits>` part of a path (for example `Run00120200`). If
there is none, it returns the last 24 characters of the pattern with every
run of non-word characters replaced by `_`.

#### `run_process_per_run(name, jobs=4, chunk_files=10, log_tail=10, bar=True, run_optional=False, extra_args=None, fraction=0.0, runs=None)`

Processes a sample **one run at a time, each with its own GCD**. This is for
detector data: the dead DOMs and the calibration change from run to run, the
L3 data files carry no GCD frames of their own, and `process_L4.py` takes
only one `--gcd`. Because the output is named after the run and not after a
worker number, `jobs` can change between calls, and an interrupted production
resumes by simply calling it again. One run's output also keeps
`(Run, Event, SubEvent)` unique, which the pass2 cross-check needs.

1. Creates the output directory and `_filelists/`.
2. Gets the `(label, gcd, files)` list from `run_gcd_pairs`. With `runs`,
   keeps only those labels. Returns `None` if nothing is left.
3. **Guard against a changed file list.** For each run, if
   `_filelists/<sample>_<label>.txt` exists with a different list and
   output `L4_<sample>_<label>*.hdf5` exists, the run is stale. If any run is
   stale it raises `RuntimeError` listing each such run, and gives the
   `rm -rf` command. (This is what changing `fraction` between calls does.)
4. Prints every run with its file count and GCD name.
5. Runs a pool of at most `jobs` processes at once: while runs are waiting
   and fewer than `jobs` are still running, it starts the next one (nested
   `launch`) and a reader thread for it; it checks again every second. When
   everything has finished, it joins every reader thread before looking at
   the results, so no failure is missed.
6. Sums the size of `L4_<sample>_Run*.hdf5`. If any run failed: marks the
   bar failed, prints the last lines of each failed run, prints the call
   that reruns just those runs, and returns `None`. Otherwise returns
   `cfg["hdf5"]`.

Nested functions:

- **`launch(label, gcd, files)`**: writes the run's file list to
  `_filelists/<sample>_<label>.txt`, starts `python -u process_L4.py --gcd
  <this run's GCD> --input-list <list> --output-hdf5 L4_<sample>_<label>.hdf5
  <sample flags>` (plus `--chunk-files`, `--run-optional`, `extra_args`),
  sets up the run's state, and returns the `Popen` object.
- **`reader(label, p)`**: reads the run's output, keeps its last lines, and
  on each `[CHUNK]` line updates the run's finished files and booked events
  and redraws the bar (`<done>/<total> runs  events <N>`). When the process
  ends it counts the run as done and, if the exit code is not 0, adds it to
  `failed`.

*Watch out:*
- The bar is fed by `[CHUNK]` lines only, so with `chunk_files=0` it does
  not move until the end.
- The final size and part count use the glob `L4_<sample>_Run*.hdf5`. If a
  label came from the fallback of `_run_label` (no `Run<digits>` in the
  pattern), those files are not counted in the message (they are still
  written and still loaded).

### A trap shared by all three drivers

Each driver guards only against changes **within its own naming scheme**.
If you process the same sample once with `run_process` (files
`L4_nue_partNNN.hdf5`) and again with `run_process_parallel` (files
`L4_nue_jobJ_partNNN.hdf5`), both sets stay in the directory, both match the
loader's `L4_nue*.hdf5` glob. The same happens if you switch between
`chunk_files=0` (`L4_nue.hdf5`) and `chunk_files>0`. The runners do not
detect this; the LOADER does: every output's `.meta.json` lists its L3 files,
and `data.load_sample` refuses to load two parts that share one
(`_refuse_double_booking`, Part C). Outputs made before that list was
recorded cannot be checked, and the loader says so. Either way, clear the
sample's directory when you change the way it is driven.

---

## `scripts/scan_files.py`

### What this file is for

A standalone scan for corrupt `.i3` files. It runs the same check as
`process_L4.py --scan`, but once, before any processing. The intended use is
to scan a set once, keep the list of healthy files, and give that list to
every job with `--input-list ... --scan off`, so the jobs do not repeat the
scan.

### Where it sits

Run by hand. It imports `oscnext_l4.tray_io.validate_files` (which needs
`icecube.dataio`, so it must run inside the IceTray environment). Its
`--good-list` output feeds `process_L4.py --input-list`.

### Command-line flags

| Flag | Default | What it does |
|---|---|---|
| `input` (positional, one or more) | *(required)* | File paths or glob patterns. Each pattern is globbed and sorted. |
| `--full` | off | Read every frame of every file (slow but certain). |
| `--frames N` | `25` | Frames read per file in quick mode. |
| `--good-list FILE` | none | Write the healthy files here, one path per line. |
| `--bad-list FILE` | `scan_bad.txt` | Write the corrupt files here, as `path<TAB>reason`. |

### Example command lines

```bash
# quick scan of the pass3 numu set, write the healthy list
python scripts/scan_files.py --good-list good_23799.txt \
    '/data/ana/LE/oscNext/pass3/genie/level3/23799/*.i3.zst'

# then process with the list, without scanning again
python scripts/process_L4.py --gcd <GCD> --input-list good_23799.txt --scan off \
    --output-hdf5 L4_output/hdf5/numu/L4_numu.hdf5 --mc --genie --chunk-files 10

# full scan of a pass2 set
python scripts/scan_files.py --full \
    '/data/ana/LE/oscNext/pass2/genie/level3/121122/*.i3.zst'
```

### Functions

#### `main()`

1. Parses the flags.
2. Expands each input: a glob with matches adds them (sorted); a path that
   exists but does not glob is added as it is; anything else is "unmatched".
   Each unmatched pattern is printed as `[!] matches nothing: ...` (it is not
   counted as corrupt). Exits if no file is found.
3. Exits if the directory for `--good-list` or `--bad-list` does not exist.
4. Prints the number of files and the mode, then calls
   `validate_files(files, n_frames=0 if --full else --frames)`. That
   function marks a file bad if it is empty (0 bytes), cannot be stat'ed, or
   raises while its frames are read; it prints `  scanned: i/N` every 100
   files.
5. Prints the healthy and corrupt counts, and every corrupt file with its
   reason.
6. If any file is corrupt, writes the bad list. If none is, and a bad list
   from an earlier run exists, **deletes** it, so an old list does not look
   current.
7. With `--good-list`, writes the healthy list and prints the
   `process_L4.py --input-list ... --scan off` command to use it.
8. **Returns** exit code 1 if any file was corrupt or any pattern matched
   nothing, else 0 (so a wrapper script can check it).

*Watch out:* the default `--bad-list` is `scan_bad.txt` in the **current
directory**, and it is overwritten (or deleted) by every scan. Scanning two
sets from the same directory keeps only the last list unless you name them.

---

## `scripts/diagnose_env.py`

### What this file is for

A report of what is and is not available in the current Python / IceTray
environment. Run it inside the environment and share the whole output when
something does not import. Most of it is top-level code that runs from top to
bottom; there is one helper function.

### Where it sits

Run by hand, normally as `./setup_env.sh run python
scripts/diagnose_env.py`. It imports nothing from `oscnext_l4` (it only adds
the repository root to `sys.path`). It takes no command-line arguments.

### What it prints, in order

1. **PYTHON**: the Python version and `sys.executable`.
2. **ENVIRONMENT**: the values of `I3_BUILD`, `I3_SRC`, `SROOT`.
3. **CORE ICETRAY**: imports `icecube`, `icecube.icetray`,
   `icecube.dataclasses`, `icecube.dataio`, `icecube.phys_services`, and
   `from icecube.icetray import I3Tray`.
4. If `icecube` cannot be found at all, it prints the two commands that load
   the cvmfs `py3-v4.4.2` toolset and run this script inside the
   `icetray/v1.17.0` env-shell, and **exits with code 1**. Nothing below can
   be judged without IceTray.
5. **BOOKING / TABLE WRITING**: `icecube.tableio`, `icecube.hdfwriter`,
   `icecube.rootwriter`, `tables` (pytables, needed to read the HDF5 files)
   and `h5py` (informational). If `hdfwriter` is missing it says booking has
   no fallback; if present, it notes that it is deprecated but still needed.
6. **PROJECTS REQUIRED BY THE L4 VARIABLES**: `icecube.DomTools`,
   `icecube.linefit`, `icecube.tensor_of_inertia` (only for
   `--run-optional`), `icecube.fill_ratio`, `icecube.common_variables`,
   `icecube.DeepCore_Filter`. Then it tries `icetray.load(...)` for the C++
   libraries `static-twc` and `slc-veto` (the latter is only for
   `--run-optional`).
7. **LIGHTGBM**: imports `lightgbm`. If it is missing, it says the Python is
   most likely not the env-shell one (lightgbm ships inside the cvmfs
   metaproject).
8. **oscNext PROJECT**: tries `icecube.oscNext` and three submodules. Their
   absence is **expected** in v1.17.0; it prints which files of this
   repository stand in for them.
9. **L3 OUTPUT**: prints a Python snippet that dumps the keys of the first
   Physics frame of an L3 file, and the keys to look for
   (`IC2018_LE_L3_Vars`, `IC2018_LE_L3_bools`, `SplitInIcePulses`, the
   cleaned series, `I3MCWeightDict`).
10. **PYTHON ML PACKAGES**: `numpy`, `sklearn`, `lightgbm`, `joblib`,
    `pandas`, with versions. Only `numpy` and `lightgbm` are needed.

### Functions

#### `try_import(name, note="")`

Imports module `name` with `importlib.import_module`. On success prints
`[OK] <name> <note>` and returns the module. On any exception prints
`[MISSING] <name> <exception type>: <message>` and returns `None`. It never
raises.

---

## `scripts/collect_meta.sh`

### What this file is for

Copies the text sources of the pass2 production (the `oscNext_meta
V01-00-07` build on cvmfs, and the L4 training scripts in the "fridge"
directory) into the repository, so they can be read next to this code. It
copies only text sources, no binaries, models or data.

### Where it sits

Run by hand, with no arguments: `bash scripts/collect_meta.sh`. It changes
to the repository root first, so the working directory does not matter. It
needs read access to
`/cvmfs/icecube.opensciencegrid.org/users/Oscillation/software/oscNext_meta/releases/V01-00-07`
and to
`/data/sim/DeepCore/2018/workspace/fridge/processing/samples/oscNext/selection/level4`.
Nothing else in the repository calls it.

### What it does

It writes into **`reference/oscNext_meta/`** (created if missing) with four
subdirectories:

1. `oscNext/`: every `.py` in the production's `oscNext/python/selection/`
   and `oscNext/python/tools/` (the live L3/L4/L5 selection scripts).
2. `tau_bdt/`: the `src/tau-bdt/python` directory (where `I3CutL7Module.py`,
   the VICH module, lives).
3. `event_selection/`: the private and public `analysis/.../event_selection`
   directories (`CalculateVariables.cxx`, `Variables.h`: the source of
   `accumulated_time` and `separation_in_cogs`).
4. `fridge_level4/`: every `.py` of the fridge's level-4 training recipe.

Then it deletes every `*.pyc` file and every file larger than 2 MB under
`reference/oscNext_meta/` (printing the names of the large ones), and prints
the total size and file count. If any source path was missing it prints
`[!] some sources were not found`. The script uses `set -u` (unset variables
are errors) but not `set -e`, so a failed copy does not stop it.

*Watch out:* the project rule is that files under `reference/` are never
edited. This script writes new files there and overwrites earlier copies in
`reference/oscNext_meta/` when you rerun it.

### Shell functions

#### `copy <dest> <source>...`

For each source: if it exists, copies it recursively into `<dest>/`;
otherwise prints `[!] not found: <source>` and sets `missing=1`. If a glob
such as `"$F"/*.py` matches nothing, bash passes it through literally, so it
is reported as not found.

---

## `setup_env.sh`

### What this file is for

Finds an IceTray environment, checks it, and runs things inside it. It
exists because `env-shell.sh` (the IceTray script that sets up the
environment) opens a **new shell**: in a script, a line after
`env-shell.sh` runs after that shell has closed, so without IceTray. For a
single command the right form is `env-shell.sh -- <command>`, which is what
this script uses.

### Where it sits

Run by hand from the repository root. The README and the notebook's error
messages point to it (for example `./setup_env.sh run python -c 'import
lightgbm'`). It calls `env-shell.sh`, the cvmfs toolset's `setup.sh`,
`python`, and for the `kernel` subcommand `python -m ipykernel install`.

### Subcommands

| Command | What it does |
|---|---|
| `./setup_env.sh` or `./setup_env.sh report` or `./setup_env.sh find` | Prints a report and changes nothing: the repository path, `SROOT`, `I3_BUILD`, whether IceTray is already loaded, every self-built `env-shell.sh` found, up to 5 cvmfs `env-shell.sh` found (newest version first), and then an import test (`icecube`, `lightgbm`, `I3Tray`). If IceTray is already loaded the test runs in the current environment; otherwise it runs inside the environment `find_env_shell` picks, and also prints the toolset a self-built build was compiled against (from its `CMakeCache.txt`). |
| `./setup_env.sh shell` | Opens an IceTray shell. If IceTray is already loaded it prints which one and exits with 0 instead of nesting a second environment. |
| `./setup_env.sh run <cmd> [args...]` | Runs one command inside IceTray. If IceTray is already loaded it runs the command directly (`exec`); otherwise `exec env-shell.sh -- <cmd>`. Exits 2 if no command is given. |
| `./setup_env.sh kernel` | Registers a Jupyter kernel named `icetray`, shown as "IceTray (oscNext L4)", with `python -m ipykernel install --user`. It uses the current Python if IceTray is already loaded, otherwise the Python inside the chosen `env-shell.sh`. |
| `./setup_env.sh lab [PORT]` | Starts `jupyter lab --no-browser --port PORT` (default 8888) inside IceTray, the same way as `run`. |
| anything else | Prints the usage comment at the top of the file and exits with 1. |

In every subcommand that needs an environment, "IceTray is already loaded"
(the test `python -c "import icecube"` succeeds) takes priority over any
search. This avoids env-shell's `I3_BUILD CHANGED` refusal when you are
already inside a different build.

### The search order for an IceTray environment (`find_env_shell`)

The first match wins. A match means an executable `env-shell.sh`.

1. `$OSCNEXT_I3_BUILD/env-shell.sh`, if the variable is set. If it is set
   but holds no executable `env-shell.sh`, a warning goes to stderr and the
   search continues.
2. `$I3_BUILD/env-shell.sh`, if `I3_BUILD` is set. (In a shell opened from
   the cvmfs metaproject, `I3_BUILD` points at a directory without
   `env-shell.sh`, so this step does not match there.)
3. Self-compiled builds, in this order:
   `/data/user/<you>/icetray_build/build/`,
   `/data/user/<you>/*/build/`,
   `/data/user/<you>/build/`,
   `~/icetray/build/`,
   `~/*/build/`,
   `~/*/*/build/`,
   `~/build/`.
4. The cvmfs metaproject the pipeline was verified on:
   `/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/${OS_ARCH:-RHEL_9_x86_64_v2}/metaprojects/icetray/v1.17.0/env-shell.sh`.
5. Any other cvmfs metaproject,
   `/cvmfs/icecube.opensciencegrid.org/py3-v*/*/metaprojects/icetray/*/env-shell.sh`,
   sorted by version (`sort -rV`), newest first.

If nothing is found, the function returns 1 and the caller prints
`env-shell.sh not found`.

Note that step 3 comes before cvmfs: if you have a self-built IceTray under
`/data/user/<you>/`, it is chosen over cvmfs unless IceTray is already
loaded in the shell you start from.

### Shell functions

#### `find_env_shell`

Prints the path of the `env-shell.sh` to use, following the search order
above, and returns 0; returns 1 if none is found.

#### `load_toolset_for <env-shell path>`

A cvmfs metaproject's `env-shell.sh` expects its toolset to be loaded first
(`eval $(.../py3-vX/setup.sh)`, which sets `SROOT` and the matching Python).
This function does that when needed: if `SROOT` is already set it does
nothing; if the path is not under `/cvmfs/icecube.opensciencegrid.org/py3-v<N>`
it does nothing; otherwise it runs `eval "$(<toolset>/setup.sh)"` and says
so on stderr.

*Watch out:* for a self-compiled build it loads nothing, even though the
build needs the toolset it was compiled against. `report` shows that
toolset (via `detect_toolset`), but loading it is up to you.

#### `detect_toolset <build dir>`

Reads `<build dir>/CMakeCache.txt` and prints the first
`/cvmfs/icecube.opensciencegrid.org/py3-v<N>/<platform>` path in it, which is
the toolset the build was compiled against. Returns 1 if there is no cache
file. Used only by `report`.

#### `report`

Prints the report described in the subcommand table.

#### `already_inside`

Returns success if `python -c "import icecube"` works, meaning an IceTray
environment is already loaded.

#### `run_here_or_in_shell <cmd> [args...]`

The body of `run` and `lab`. Exits 2 if no command is given. If IceTray is
already loaded, prints a note on stderr and `exec`s the command directly.
Otherwise finds `env-shell.sh`, loads the toolset if needed, and `exec`s
`env-shell.sh -- <cmd>`. Exits 1 if no `env-shell.sh` is found.

*Watch out (the `kernel` subcommand):* `ipykernel install` records the path
of the Python in the kernel's `kernel.json`. By default it does not record
environment variables such as `PYTHONPATH` or `LD_LIBRARY_PATH` that
`env-shell.sh` sets. So whether `import icecube` works in that kernel may
depend on how the Jupyter server itself was started. Starting Jupyter from
inside the environment (`./setup_env.sh lab`, or `jupyter lab` in an
`env-shell`) avoids the question. Check with `import sys; sys.executable` and
`import icecube` in the first cell.


---

# Part C — the analysis side: booked HDF5 → training sets → models → checks

This part covers everything that happens **after** `process_L4.py` has booked
the L4 variables to HDF5 files.  None of this code needs IceTray.  It needs
`numpy` and `pytables`.  Training also needs `lightgbm`, and the plots need
`matplotlib`.

```
config/variables.json ──► oscnext_l4/varmap.py ──► oscnext_l4/data.py ──┐
config/productions.json ─────────────────────────────────────────────────┤
                                                                          ▼
<hdf-base>/<sample>/L4_<sample>*.hdf5  (+ .meta.json)                     │
        │                                                                 │
        ▼   data.load_sample, data.add_weights                            │
   {sample: {variable: array, w_phys: array, ...}}  ◄────────────────────┘
        │
        ▼   oscnext_l4/dataset.py  (stack, split, noise cut, build_dataset)
        │   driven by scripts/make_dataset.py (or notebook section 6)
   L4_noise_dataset.npz,  L4_muon_dataset.npz
        │
        ▼   scripts/train_L4_classifier.py
   L4_<tag>_model.txt  +  L4_<tag>_model.json  +  <tag>_*.png
        │
        ├──► scripts/check_leakage.py       (duplicated rows across the split)
        ├──► scripts/plot_inputs.py         (input distributions, per-variable AUC)
        └──► scripts/check_application.py   (tray score == training-side score)

scripts/inspect_production_table.py   (read-only dump of the production's own
                                       training tables; stands apart)
```

The order of this part:

1. [The two config files](#1-the-two-config-files)
2. [The HDF5 layout the code expects](#2-the-hdf5-layout-the-code-expects)
3. [The methods, in prose](#3-the-methods-in-prose): weights, the train/test
   split, the noise cut, class balancing, training, the application check
4. [File by file](#4-file-by-file): every function, in file order

Two words used throughout:

- **sample**: one named entry in `productions.json`, for example `nue`,
  `corsika` or `data`.  One sample is booked into one directory of HDF5
  files.
- **part**: one HDF5 file of a sample.  A sample is usually booked in many
  parts (`L4_nue_part000.hdf5`, `L4_nue_job3_part002.hdf5`, ...).  One L3
  file always goes into exactly one part.

---

## 1. The two config files

Both files are **pure data**.  Reading either one is `json.load()` and nothing
else.  JSON has no comments, so the reasons behind the values are written in
`config/README.md`.  A key that starts and ends with `__` (for example
`__README__`, `__runs__`) is a note for humans, and every reader ignores it.

### 1.1 `config/variables.json` — one row per variable, both sides of it

#### Why "one row, both sides"

A trained model is used in two places, and they read the same quantity in two
different ways:

- **Training** reads a **column of an HDF5 table**.
- **Application** (the `L4Classifier` tray module in `oscnext_l4/classifier.py`)
  reads a **field of an object in an I3 frame**.

The names differ between the two.  The hdfwriter converter renames things.
For example, the fill ratio is the field `fill_ratio_from_mean` on the frame
object, but the column `fillratio_from_mean` in the HDF5 file.  If the two
sides ever pointed at **different** quantities, the model would be trained on
one number and applied to another.  It would not raise an error.  It would
just give wrong scores.

So each variable is **one row** that holds both sides.  You cannot edit one
side without seeing the other.  `varmap.check()` (called as
`data.check_feature_map()`) then checks that the two sides of each row still
overlap.

#### Top-level fields

| field | meaning |
|---|---|
| `__README__` | a note for humans; ignored |
| `bdt_features` | `{"noise": [5 names], "muon": [10 names]}`.  The input list of each classifier, **in the model's feature order** (see below). |
| `kind_to_scheme` | `{"muon_bg": "corsika", "noise_bg": "noise", "signal": "genie"}`.  A fallback only: it gives a weight scheme to a sample spec that has a `kind` but no `weight` field. |
| `schemes_without_aux` | `["data"]`.  Weight schemes that have **no** per-event weight columns on purpose.  Detector data is weighted by 1/livetime, which is not a column.  This lets `check_registry` tell "nothing to check, by design" from "the caller built an empty list by mistake". |
| `variables` | the rows, `name → row`.  46 rows today. |

#### Fields of one row

| field | meaning |
|---|---|
| `hdf5` | `[table, column]` in a booked HDF5 file.  This is what training reads. |
| `hdf5_alts` | optional list of further `[table, column]` pairs that hold the **same** physical quantity under another spelling (older productions, other converter versions).  When loading, the first pair that exists in the file wins, starting with `hdf5`. |
| `frame` | `[frame key, field]` of the same quantity in an I3 frame.  This is what the tray reads.  A dotted field walks the Python binding: `cog.z` means `frame[key].cog.z`. |
| `frame_alts` | optional list of other **field** names on the same frame key, tried in order. |
| `schemes` | present **only on weight columns**.  It lists the weight schemes (`genie`, `noise`, `corsika`, `muongun`) that need this column.  Its presence is what marks a row as a weight column and not a physics variable. |

A row with `schemes` has no `frame` side: weight columns are never read from a
frame.

#### The 15 BDT input slots (14 distinct variables)

The noise BDT has 5 inputs and the muon BDT has 10.  `NchCleaned` is an input
to both, so there are 14 distinct variables.  The tables are in **feature
order**: position 0 is the first column of the model's input vector.

**Noise BDT** (`bdt_features.noise`):

| pos | name | HDF5 (table, column) | HDF5 alternatives | frame (key, field) | frame alternatives |
|---|---|---|---|---|---|
| 0 | `NchCleaned` | `IC2018_LE_L3_Vars`, `NchCleaned` | – | `IC2018_LE_L3_Vars`, `NchCleaned` | – |
| 1 | `micro_count` | `L4_micro_count`, `STW_m3500p4000_DTW200` | `L4_micro_count`, `STW7500_DTW200` | `L4_micro_count`, `STW_m3500p4000_DTW200` | – |
| 2 | `iLineFit_speed` | `L4_iLineFitParams`, `lf_vel` | `L4_iLineFitParams`, `LFVel`; `L4_iLineFit`, `speed` | `L4_iLineFitParams`, `lf_vel` | `LFVel`, `speed` |
| 3 | `fill_ratio` | `L4_fill_ratio`, `fillratio_from_mean` | `…`, `fill_ratio_from_mean`; `…`, `fillRatioFromMean`; `…`, `FillRatioFromMean` | `L4_fill_ratio`, `fill_ratio_from_mean` | `fillratio_from_mean` |
| 4 | `FullTimeLengthRatio` | `L4_FullTimeLengthRatio`, `value` | – | `L4_FullTimeLengthRatio`, `value` | – |

**Muon BDT** (`bdt_features.muon`):

| pos | name | HDF5 (table, column) | frame (key, field) | frame alternatives |
|---|---|---|---|---|
| 0 | `ICVetoHits` | `IC2018_LE_L3_Vars`, `ICVetoHits` | same | – |
| 1 | `RTVeto250Hits` | `IC2018_LE_L3_Vars`, `RTVeto250Hits` | same | – |
| 2 | `NchCleaned` | `IC2018_LE_L3_Vars`, `NchCleaned` | same | – |
| 3 | `NAbove200Hits` | `IC2018_LE_L3_Vars`, `NAbove200Hits` | same | – |
| 4 | `VICH_nch` | `L4_VICH_nch`, `value` | same | – |
| 5 | `accumulated_time` | `L4_accumulated_time`, `value` | same | – |
| 6 | `first_hlc_rho` | `L4_first_hlc_rho`, `value` | same | – |
| 7 | `cog_z` | `SRTTWSplitInIcePulsesDCHitStatistics`, `cog_z` | `SRTTWSplitInIcePulsesDCHitStatistics`, `cog.z` | `cog_z` |
| 8 | `z_sigma` | `SRTTWSplitInIcePulsesDCHitStatistics`, `z_sigma` | same | – |
| 9 | `z_travel` | `SRTTWSplitInIcePulsesDCHitStatistics`, `z_travel` | same | – |

None of the muon inputs has an HDF5 alternative.

Two spellings worth knowing:

- `cog_z` is a **column** name (the tableio converter flattens the position).
  On the frame object, `I3HitStatisticsValues.cog` is an `I3Position`, so the
  frame side must be `cog.z`.
- The hit-statistics key keeps the **pass3** spelling
  (`SRTTWSplitInIcePulsesDC…`) even on a pass2 run.  The values come from
  whatever series was passed; only the key name is fixed.  Booking and reading
  use the same literal, so they agree.

#### Why the order in `bdt_features` is written out

A LightGBM model takes a plain vector of numbers.  Position `i` of the vector
must be the variable the model saw at position `i` during training.  So the
order of `bdt_features.muon` **is** the muon model's input order.

It is written out by hand rather than derived from the order of the
`variables` block, because the two orders differ.  `NchCleaned` sits in the
noise part of `variables` but is **third** in the muon list.  Deriving the
list from the block would hand the muon model a vector in the wrong order,
with no error.

The order is carried all the way through: `build_dataset` stores it in the
`.npz` as `features`, the trainer passes it to LightGBM as `feature_name`, and
the model file stores it.  The trainer refuses a `.npz` whose `features` do not
equal `bdt_features` (unless `--features` is given on purpose).  The tray and
`check_application.py` read the order back from the booster itself.

#### The other rows

- **19 candidate variables**: booked and readable, but not BDT inputs today:
  `C2HR6`, `CausalVetoHits`, `CleanedFullTimeLength`, `DCFiducialHits`,
  `STW9000_DTW300Hits`, `UncleanedFullTimeLength`, `VICH_npulses`,
  `VICH_qtot`, `VertexGuessZ`, `VetoFiducialRatioHits`, `cog_x`, `cog_y`,
  `first_hlc_x/y/z`, `n_hit_doms`, `separation_in_cogs`, `z_min`, `z_max`.
- **13 weight columns** (rows with `schemes`):

| name | HDF5 (table, column) | scheme |
|---|---|---|
| `true_energy` | `I3MCWeightDict`, `PrimaryNeutrinoEnergy` | genie |
| `OneWeight` | `I3MCWeightDict`, `OneWeight` | genie |
| `NEvents` | `I3MCWeightDict`, `NEvents` | genie |
| `pdg` | `I3MCWeightDict`, `PrimaryNeutrinoType`; alt. `MCInIcePrimary`, `pdg_encoding` | genie |
| `n_flux_events` | `L4_n_flux_events`, `value` | genie |
| `gen_ratio` | `I3MCWeightDict`, `gen_ratio` | genie |
| `noise_weight` | `noise_weight`, `weight`; alt. `noise_weight`, `value` | noise |
| `cwm_Weight` | `CorsikaWeightMap`, `Weight` | corsika |
| `cwm_NEvents` | `CorsikaWeightMap`, `NEvents` | corsika |
| `cwm_OverSampling` | `CorsikaWeightMap`, `OverSampling` | corsika |
| `MuonWeight` | `MuonWeight`, `value` | muongun |
| `MuonWeight_GaisserH4a` | `MuonWeight_GaisserH4a`, `value` | muongun |
| `MuonGunWeight` | `MuonGunWeight`, `value` | muongun |

**Watch out:** an entry in `hdf5_alts` must be the **same** physical quantity.
The fill-ratio table also has `from_rms` and `from_nch` columns; putting one
of those in the alternatives would silently train on a different variable.

### 1.2 `config/productions.json` — the per-production facts

This file is the one place that knows a fact that differs between pass2 and
pass3.  Everything downstream keys on **roles** and **weight schemes**, never
on sample names.

#### Top level

| field | meaning |
|---|---|
| `__README__` | a note; ignored |
| `bdt_roles` | what each classifier loads, as roles: `noise: [signal, noise_bg]`, `muon: [signal, muon_bg]` |
| `productions` | `pass3` and `pass2`, each described below |

#### Per production

| field | pass3 | pass2 | meaning |
|---|---|---|---|
| `gcd` | the Pass3 `GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz` on cvmfs | an averaged pass2 GCD under `/data/ana/LE/oscNext/pass2/genie/level4/121122/` | the GCD for every MC sample |
| `cleaned_pulses` | `SRTTWSplitInIcePulsesDC` | `SRTTWOfflinePulsesDC` | the cleaned pulse series; appended to the booking flags as `--cleaned-pulses <name>`.  The uncleaned series is `SplitInIcePulses` in both, so it is not configured. |
| `noise_weight_unit` | `per_ns` | `hz` | the unit of the vuvuzela weight.  `per_ns` → factor 1e9, `hz` → factor 1.  **Wrong value = every noise rate off by 1e9, with no error.** `make_dataset.py` refuses a production that does not declare it. |
| `samples` | `nue`, `numu`, `corsika`, `noise` | `nue`, `numu`, `noise`, `nutau`, `muongun`, `data` | name → sample spec; order is kept |

#### Per sample

| field | meaning |
|---|---|
| `l3` | a glob, or a list of globs (a sample made of several datasets, e.g. pass2 `nue` = 121122 + 121291) |
| `flags` | passed verbatim to `process_L4.py` (`--mc`, `--genie`, `--corsika`, `--noise`, `--muongun`).  Detector data has `[]`. |
| `kind` | the **role**: `signal`, `noise_bg` or `muon_bg` |
| `weight` | the **weight scheme**: `genie`, `noise`, `corsika`, `muongun` or `data`.  Picks the weighting function in `data.WEIGHTERS`. |
| `gcd` | optional; only `pass2.data` has it.  A list parallel to `l3`: one GCD glob per run directory, because detector data needs the run's own GCD. |
| `__runs__` | a note on `pass2.data` |

The samples in use:

| production | sample | kind | weight | L3 dataset(s) |
|---|---|---|---|---|
| pass3 | nue | signal | genie | 23800 |
| pass3 | numu | signal | genie | 23799 |
| pass3 | corsika | muon_bg | corsika | 23694 |
| pass3 | noise | noise_bg | noise | 23813 |
| pass2 | nue | signal | genie | 121122, 121291 |
| pass2 | numu | signal | genie | 141154, 141292 |
| pass2 | noise | noise_bg | noise | 888003 |
| pass2 | nutau | signal | genie | 160511 |
| pass2 | muongun | muon_bg | muongun | 139008 |
| pass2 | data | muon_bg | data | 18 detector runs, three per year 2012–2017 |

`pass2` has two samples with `kind: muon_bg`.  They must never be stacked
together (a model would learn simulation vs. measurement).
`make_dataset.py` picks one, preferring detector data.

`OSCNEXT_L4_CONFIG` points `make_dataset.py` (and the notebook) at a different
productions file; `OSCNEXT_L4_VARIABLES` points `varmap` at a different
variables file.

---

## 2. The HDF5 layout the code expects

The files are written by hdfwriter (`tray_io.add_booker`, Part B) with the
`InIceSplit` sub-event stream.  The reading code in `data.py` assumes this
layout.

### 2.1 Directory and file names

```
<hdf-base>/<sample>/L4_<sample>.hdf5                    one-shot run
<hdf-base>/<sample>/L4_<sample>_part000.hdf5            serial run, --chunk-files
<hdf-base>/<sample>/L4_<sample>_job3_part002.hdf5       parallel run
<hdf-base>/<sample>/L4_<sample>_Run00120200_part000.hdf5  detector data, per run
<hdf-base>/<sample>/L4_<sample>_smoke.hdf5              smoke test (--n)
```

Each sample's spec gets `hdf5 = <hdf-base>/<sample>/L4_<sample>.hdf5`, and
`sample_files()` globs `L4_<sample>*.hdf5` in that directory.  Smoke files are
ignored whenever a real file exists.

### 2.2 Inside one file

- **One table per booked frame key**, at the root: `/I3EventHeader`,
  `/IC2018_LE_L3_Vars`, `/L4_VICH_nch`, `/SRTTWSplitInIcePulsesDCHitStatistics`, ...
- Each data table starts with the **index columns** `Run`, `Event`,
  `SubEvent`, `SubEventStream`, `exists`, followed by the object's own
  columns.  An `I3Double` gives one column `value`.  An `I3MapStringDouble`
  such as `IC2018_LE_L3_Vars` gives one column per map key.  An
  `I3HitStatisticsValues` gives `cog_x`, `cog_y`, `cog_z`, `z_sigma`, ...
- **`/__I3Index__/<key>`**: for every data table, hdfwriter also writes an
  index table with **one row per booked frame**.  Its columns include
  `exists` (was the key in this frame?) and `start`/`stop` (which rows of the
  data table belong to this frame).
- **`/I3EventHeader`** is the reference table: one row per booked event.  Its
  `time_start_mjd` column (fractional MJD, float64) is used to estimate the
  livetime of detector data.

The code never uses `h5.walk_nodes()` keyed by name directly: the data table
`/X` and the index table `/__I3Index__/X` have the same leaf name `X`.
`_table_nodes()` skips everything under `__I3Index__`.

### 2.3 The `.meta.json` beside each part

`process_L4.py` writes `<part>.hdf5.meta.json` **after** the part is complete.
Fields:

| field | meaning | read by |
|---|---|---|
| `n_l3_files` | L3 files actually processed into this part.  **The weight divisor.**  `null` for a `--n` smoke run. | `_n_l3_files_why` |
| `n_l3_files_unreliable` | `true` for a `--n` run (single-output mode only) | `_n_l3_files_why` |
| `l3_files` | the L3 files actually processed | `_refuse_double_booking` |
| `n_l3_files_given`, `l3_files_given` | what the part was asked to process | (Part B) |
| `physics_frames`, `sub_event_stream`, `after_stream_filter`, `booked` | frame counts | (Part B) |
| `chunk_index`, `n_chunks` (chunked) / `n_frames_limit` (single) | bookkeeping | (Part B) |
| `elapsed_s` | wall time | – |

### 2.4 How `load_sample` lines tables up with events

The problem: a table for key `X` has one row per frame **that had `X`**.  If
`X` was missing in some frames, row `i` of `/X` is not event `i`.  And
`(Run, Event, SubEvent)` is **not unique** inside one part: in MC, `Run` is
the dataset number and `Event` restarts at 0 in every L3 file, and a part
holds several L3 files.

So `_read_resolved()` builds, for every table, an index array `idx` of length
*n* (the number of `I3EventHeader` rows): `idx[i]` is the data-table row of
event `i`, or `-1` if event `i` has no entry.  It tries, in order:

1. **The index table** (preferred, and cheap).  If `/__I3Index__/X` exists,
   has exactly *n* rows, and has `exists` and `start` columns:
   `idx = where(exists, start, -1)`.  Any `start` past the end of the data
   table is set to `-1`.
2. **No index, identical identifiers.**  If the table's own
   `(Run, Event, SubEvent)` columns equal the header's, row by row, the table
   is aligned and is used directly.  If it has an `exists` column with some
   `False` entries, those rows are masked to `-1` (they are padding, not
   measurements).
3. **No index, and the header's triples repeat.**  Matching is impossible.
   Every value is left `NaN` and a warning is printed once per sample and
   table.  Nothing is guessed.
4. **No index, unique triples.**  A dictionary `(Run, Event, SubEvent) → row`
   is built from the header, and each data row (with `exists` true) is placed
   at its event.

Then every wanted column is read as float64 and scattered: `out[i] =
column[idx[i]]` where `idx[i] ≥ 0`, else `NaN`.  A variable that is not in the
file at all comes back as an array of `NaN`, never as a missing key.

`load_sample` does this for each part, concatenates the parts in sorted
file-name order, and records which part each event came from (`_part`).

---

## 3. The methods, in prose

### 3.1 Weights: `w_phys`

`data.add_weights(data, SAMPLES)` gives every event of every sample a
**physical weight** `w_phys`, meant as a rate in Hz.  The formula is chosen by
the sample's **weight scheme** (`productions.json` → `weight`), through the
table `WEIGHTERS`:

| scheme | function |
|---|---|
| `genie` | `genie_weight` |
| `noise` | `noise_weight` |
| `corsika` | `corsika_weight` |
| `muongun` | `muongun_weight` |
| `data` | `data_weight` |

In every MC formula below, **n_files** is `d["_n_files"]`: the **total number
of L3 files** behind the whole sample, summed from the `n_l3_files` field of
every part's `.meta.json`.  It is not the number of HDF5 parts (one part can
hold many L3 files).

After the formula, `add_weights` sets every non-finite weight to 0, prints the
total rate and the largest single weight as a share of the total (a warning
above 5%), and compares the total with Table 13 of the technical note for the
sample names it knows (`nue` 0.95 mHz, `numu` 3.77 mHz, `corsika` 505 mHz,
`noise` 36.6 mHz).  A ratio outside 0.1–10 is flagged as an order-of-magnitude
error.

#### GENIE neutrinos (`genie`)

```
w = OneWeight · Φ(E) / N_gen / n_files
Φ(E) = 2·10⁻² · E^(−3)            (NORM = 2e-2, GAMMA = −3.0; E = PrimaryNeutrinoEnergy)
```

`Φ` is a single power law.  The same `norm` and `spectral_index` are what the
official oscNext `weighting.py` uses for machine-learning training samples.
It is a training weight, not a physical flux model.  For `E ≤ 0` the flux is
`NaN` (and the weight becomes 0).

`N_gen` is the number of generated events of this neutrino type, per file.
It is found per event, best source first:

1. `n_flux_events` (column `L4_n_flux_events.value`, written at booking from
   `I3GenieInfo` when `--genie` is passed).  Used wherever it is finite.
2. Otherwise `NEvents · gen_ratio`, where `gen_ratio` is
   a. **read** from the `I3MCWeightDict.gen_ratio` column, if that column
      has any finite value (pass2 stores it); otherwise
   b. **derived** from the sign of the neutrino type `pdg`: 0.7 for
      neutrinos (`pdg > 0`), 0.3 for antineutrinos (`pdg < 0`) (`NU_FRAC`,
      `NUBAR_FRAC`).  pass3 needs this route.
   c. If neither is available, or `pdg` is `NaN` for any event that needs
      it, the function **raises `KeyError`**.  The reason: `NaN < 0` is
      `False`, so a missing `pdg` would silently give every antineutrino the
      neutrino ratio and make it 2.33× too heavy.

Route 2 is the official production formula (`neutrinos.py`:
`OneWeight · flux / (NEvents · gen_ratio)`).  Route 1 is this repository's
own; the two agree only if `n_flux_events == NEvents · gen_ratio`.  pass2
GENIE L3 has no `I3GenieInfo`, so a pass2 run always takes route 2.

#### Vuvuzela noise (`noise`)

```
w = noise_weight · k / n_files
k = 1e9  if noise_weight_unit == "per_ns"   (pass3: the weight is stored in 1/ns)
k = 1    if noise_weight_unit == "hz"       (pass2: already in Hz)
```

`k` is a module-level setting (`NOISE_NS_SCALE`), set by
`set_noise_weight_unit()` from the production's `noise_weight_unit`.  It is
printed every time the weight is computed.  In the vuvuzela samples the
weights are uniform, so for noise the weighted and unweighted efficiencies are
the same.

#### CORSIKA air showers (`corsika`)

The main route uses the `simweights` package with the **GaisserH3a** cosmic-ray
flux model.  simweights reads the generation information straight from the
HDF5 file (`CorsikaWeightMap`, `I3CorsikaInfo`, ...), so each part is
reopened:

```
for each part:
    w_part = simweights.CorsikaWeighter(part_file, nfiles = N_total)
                       .get_weights(simweights.GaisserH3a())
```

`N_total` is the **total** L3 file count of the whole sample, passed to
**every** part.  simweights builds the generation surface as
`nfiles × (one file's NEvents × OverSampling)`.  Each part must therefore be
told how many files the whole sample came from.  Giving a part only its own
count would make each part estimate the full rate by itself, and the sum
would be too high by the number of parts.

simweights returns one weight per row of the `CorsikaWeightMap` table.  These
are placed on events **by position** through `/__I3Index__/CorsikaWeightMap`
(`exists`, `start`), the same way `load_sample` does it.  If there is no
usable index but the counts match, they are used row for row; otherwise that
part's events get `NaN` (later 0).

The **fallback** (no simweights, no file list, a part without a usable
`.meta.json`, or a total count that does not match the loaded events):

```
w = CorsikaWeightMap.Weight / (CorsikaWeightMap.NEvents · OverSampling · n_files)
```

where `OverSampling` is replaced by 1 if it is not finite or ≤ 0.  It prints
that the absolute rate is unreliable; the spectrum shape is right.

#### MuonGun (`muongun`, pass2's simulated muon background)

MuonGun stores a per-event rate weight directly.  The code takes the **first**
of these columns that exists and has any finite value, and prints which:

```
MuonWeight  →  MuonWeight_GaisserH4a  →  MuonGunWeight
w = <that column> / n_files
```

If none is found, every event gets weight 0 and a warning is printed.  This
has not been checked against a pass2 absolute rate (the production also divides
by a KDE passing probability that is not booked here); the shape is what
training uses.

#### Detector data (`data`)

```
w = 1 / T_live          (the same for every event)
```

`T_live` is the total livetime, in seconds, of the runs that were booked.  It
is not in any file, so it is declared with `set_data_livetime(seconds)`.
`make_dataset.py` finds it like this:

1. It estimates it from the event times (`livetime_from_headers`): for each
   run, last event time − first event time, summed over runs.  This misses
   the few seconds before the first and after the last event, and includes
   any dead time in between.
2. It reads the exact livetime from the **Good Run Lists**
   (`livetime_from_grl`), where dead time is already subtracted.  Default
   files, tried per year from 2011 to 2023, first that exists wins:
   `/data/exp/IceCube/<y>/filtered/level2pass2/IC86_<y>_GoodRunInfo.txt`,
   then `/data/exp/IceCube/<y>/filtered/level2/IC86_<y>_GoodRunInfo.txt`.
3. If the Good Run Lists contain **every** booked run, their total is used.
   Otherwise the event-time estimate is used.  Which one is recorded in the
   training set's provenance.

If the livetime is never set, every data event gets weight 1.0 and a warning
says the printed rate is not in Hz.  **Training is not affected**:
`build_dataset` normalises each class to unit sum, and all data events have the
same weight, so any constant gives identical training weights.

### 3.2 The train/test split

Constants in `oscnext_l4/dataset.py`:

| constant | value | meaning |
|---|---|---|
| `RNG_SEED` | 12345 | seed of the random draws |
| `TRAIN_FRAC` | 0.5 | probability that an event (or a group) goes to training |
| `NOISE_CUT` | 0.70 | P_noise threshold before the muon set |

#### The shared signal split

The final L4 cut is the **AND** of the two classifiers.  To measure it
honestly, you need one held-out signal set that neither classifier trained on.
So the **signal** events get **one** train/test split, used by both
classifiers.

The noise set and the muon set are built by two separate runs of
`make_dataset.py`.  They still get the same split because both runs do the
same thing:

1. Stack the signal samples (every loaded sample with `kind == "signal"`, in
   `productions.json` order) into one array.
2. Make `rng = np.random.default_rng(12345)`.
3. **First draw**: `sig_istrain = rng.random(N_signal) < 0.5`.

Same events in the same order + same seed + same first draw = same split.

#### The noise background

In the noise stage, the **second** draw of the same generator splits the
vuvuzela background event by event:
`bg_istrain = rng.random(N_noise) < 0.5`.  Vuvuzela events are independent
(no oversampling), so an event-level split is fine.

#### The muon background: grouped by (part, Run)

In CORSIKA one air shower is reused `OverSampling` times.  If copies of one
shower land on both sides of the split, the test set contains near-copies of
training events, and the test efficiency looks better than it is.

So the muon background is split by **group**, not by event
(`split_by_shower`), with its **own** generator `default_rng(12346)` (seed + 1).
Using a separate generator means this draw does not depend on whether the
noise stage ran.

- The group of an event is the pair **(part, Run)**.  `stack()` builds a
  `_group` number per event as `(sample index << 32) + part index`.  All
  copies of one shower are written into one L3 file, and one L3 file is never
  split across parts, so a (part, Run) pair never cuts a shower in two.  This
  holds even when `Run` is the dataset number (as oscNext's simulation headers
  set it), where `Run` alone would be one value for the whole sample.
- Each distinct group goes to training with probability 0.5; all its events
  follow.
- With **fewer than 20** distinct groups there is nothing sensible to split
  on.  For CORSIKA this is an error (`make_dataset.py` exits and suggests
  booking in more parts).  For MuonGun and detector data it falls back to an
  event-level split with a warning.

#### The signal-identity check

The shared split works only if the noise run and the muon run loaded **exactly
the same signal events in the same order**.  One part missing at one stage
shifts every later event, and the two splits silently differ.

So `make_dataset.py` computes a **signal identity**: a SHA-256 hash over
`[RNG_SEED, TRAIN_FRAC, signal sample names]` and then every signal event's
`(Run, Event, SubEvent)` as int64 bytes, sample by sample.  It is stored in the
provenance of the `.npz`, and the trainer copies it into the model's `.json`.

In the muon stage, `check_shared_split` reads the noise model's `.json`
(`L4_noise_model.txt` → `L4_noise_model.json`) and compares hashes:

- different hash → **exit** with both event counts;
- no identity recorded (a model from the notebook or an older run) → warning,
  continue;
- same hash → "the split is shared".

### 3.3 The noise cut before the muon set

The production trains the muon BDT only on events that already pass the noise
classifier (note sec. 3.6.3).  `make_dataset.py --stage muon --noise-model
L4_noise_model.txt` does the same:

1. Load the trained noise model (`load_noise_prob`).  Its inputs are read in
   the **booster's own** feature order, taken from the model file.
2. Compute P(neutrino) for every stacked signal event and every muon-background
   event.
3. Keep only events with **P ≥ 0.70**, in both classes.
4. The signal's `istrain` flags are masked **together** with the events, so a
   surviving signal event stays on the side it was already on.  That keeps the
   split shared with the noise set.
5. Record in the provenance: threshold, model path and SHA-256, and kept/total
   counts per class.

It exits if the cut removes every background event.  `--no-noise-cut` builds
the muon set without the cut; this is a real deviation from the production and
is recorded as `noise_cut.applied = false`.

### 3.4 Class balancing in `build_dataset`

The classes differ hugely in size and in total rate.  The training weight
`weight` is built from `w_phys` so that both classes carry the **same total
weight**, while the shape inside each class follows `w_phys`:

1. Copy `w_phys` for signal (`ws`) and background (`wb`).  Set any non-finite
   or negative weight to 0.
2. Divide each class by its own sum: now `Σ ws = Σ wb = 1`.
3. Divide both by the single largest weight of either class,
   `scale = max(max ws, max wb)`.  Now the largest weight is 1 and both class
   sums are still equal (both `1/scale`).

So inside the signal the GENIE E⁻³ weights still decide which neutrinos count
more; inside CORSIKA the GaisserH3a weights do.  The unbalanced `w_phys` is
stored next to it for the rate cross-check.  LightGBM's own `is_unbalance` is
off.

**Watch out:** an event with `w_phys = 0` (for example an unmatched CORSIKA
weight, or `E ≤ 0`) gets training weight 0.  It does not influence training,
but it is still counted in the event-count efficiency on the test set.

The `.npz` then holds:

| key | content |
|---|---|
| one key per feature | float64 values, signal first, then background |
| `label` | 1 = signal, 0 = background |
| `weight` | the balanced training weight |
| `w_phys` | the physical weight (Hz) |
| `istrain` | `True` = training set |
| `features` | the feature names, in model order |
| `provenance` | optional; a JSON string (see `make_dataset.provenance`) |

### 3.5 Training (`train_L4_classifier.py`)

#### Hyperparameters (technical note Table 10)

| parameter | noise | muon |
|---|---|---|
| `objective` / `metric` | `binary` / `auc` | same |
| `max_depth` | 6 | 6 |
| `num_leaves` | 25 | 25 |
| `max_bin` | 32 | 32 |
| `min_data_in_leaf` | 500 | 500 |
| `feature_fraction` | **0.8** | **0.7** |
| `lambda_l1` | 2.0 | 2.0 |
| `lambda_l2` | 1.0 | 1.0 |
| `min_gain_to_split` | 2.0 | 2.0 |
| `is_unbalance` | False (balancing is in `weight`) | False |
| `learning_rate` | 0.05 | 0.05 |
| `num_boost_round` | 2000 (upper limit) | 2000 |
| `seed`, `deterministic` | 12345, True | same |

`learning_rate`, `min_data_in_leaf`, `num_boost_round` and `seed` can be
overridden from the command line.  The native LightGBM API is used, so these
names are the Table 10 names.

#### Early stopping on a slice of the TRAINING set

The test set is never looked at during training.  Instead:

1. For each class separately, `--valid-frac` (default 20%) of the **training**
   events is picked at random (`default_rng(--seed)`) as a validation slice.
   Per class, so the scarce background is not thinned by chance.
2. The model is fitted on the rest of the training set, with `weight` as
   sample weights.
3. After every boosting round the weighted AUC on the validation slice is
   computed.  If it has not improved for **100 rounds** (`EARLY_STOPPING`),
   training stops, and the model keeps the best round (`best_iteration`).
4. The model is saved with that number of trees.

`--no-early-stopping` (or `--valid-frac 0`) fits on the whole training set for
exactly `num_boost_round` rounds.  If fewer than `10 × min_data_in_leaf`
background events are fitted, a warning suggests lowering `--min-data-in-leaf`.

#### The efficiency/rejection table

Evaluation uses the **test set only** and **counts events** (no `w_phys`),
because the rate weights depend on the invented E⁻³ flux.

For a cut `c` on the score P(signal):

```
efficiency(c) = (# test signal with score ≥ c) / (# test signal)
rejection(c)  = 1 − (# test background with score ≥ c) / (# test background)
```

The candidate cuts are the distinct test-background scores, plus one below
the lowest and one above the highest.  For a target rejection *R*, the cut is
the one with the **highest efficiency among all cuts with rejection ≥ R**.

The table has rows for R = 90%, 95%, 99%, 99.5%, 99.9%.  Each row shows the
efficiency, the cut, and the raw counts `sig kept / sig total` and
`bg kept / bg total`.  A row with fewer than 10 background events left is
marked `<-- NOT A MEASUREMENT`.

Then:

- **Headline**: the efficiency at `--target-rejection` (default 0.99 for
  noise, 0.94 for muon).
- **Rate-weighted cross-check**: the cut is placed where the cumulative
  `w_phys` of the test background (sorted by score) reaches the target
  fraction; the weighted signal efficiency above it is printed.
- **At the note's cut**: efficiency and rejection at P = 0.70 (noise) or
  0.65 (muon).
- **Feature importance**: both `gain` (total loss reduction; skewed) and
  `split` (number of times used; the note's Figures 14/21 show split counts).

#### The train/test gap (the overtraining test)

At `--gap-at` rejection (default 90%), the efficiency is computed twice: on
the training set, with a cut from the training background scores, and on the
test set, with a cut from the test background scores.

```
gap = efficiency_train − efficiency_test
```

A gap above 5 points is flagged `<-- MEMORISING`.  90% is the default because
about 100 background test events sit above that cut; at 99% it would be about
10, mostly noise.  A KS test on the score distributions is deliberately not
used: on large samples it flags differences that are statistically
significant but physically irrelevant.

Note: the "training set" here is the whole `istrain` set, including the
early-stopping validation slice.

#### The sidecar `L4_<tag>_model.json`

| field | content |
|---|---|
| `tag` | `noise` or `muon` |
| `features` | input names in model order |
| `params` | the LightGBM parameters actually used (after overrides; `num_threads` only if set) |
| `num_boost_round_limit` | the round limit |
| `n_trees` | trees kept (best iteration) |
| `model_sha256` | hash of the `.txt` it describes |
| `trained_at`, `lightgbm_version` | when and with what |
| `dataset` | absolute path of the `.npz` |
| `n_train_sig`, `n_train_bg`, `n_test_sig`, `n_test_bg` | set sizes |
| `target_rejection` | the headline target |
| `metrics.eff_at_target`, `cut_at_target`, `bg_kept_at_target` | the headline |
| `metrics.gap_at`, `metrics.gap` | the overtraining gap |
| `metrics.eff_weighted_at_target` | the `w_phys` cross-check |
| `metrics.table` | the rows of the efficiency table (rejection, cut, eff, rej, sig/bg kept and total) |
| `metrics.at_default_cut` | efficiency/rejection/counts at 0.70 or 0.65 |
| `importance_gain`, `importance_split` | per feature |
| `default_cut` | 0.70 or 0.65 |
| `plots` | paths of the PNGs written |
| `provenance` | copied verbatim from the `.npz`, or `null` |

### 3.6 What `check_application.py` compares

`varmap.check()` only compares **names**.  It cannot tell whether the frame
object really has the named field.  (It passed while the frame side of `cog_z`
pointed at a field the binding does not have; every event was scored with
that input missing, and nothing raised.)

`check_application.py` tests the real thing.  You first run `process_L4.py`
with both `--apply-cut` and `--output-hdf5`.  That HDF5 then contains, for
every event:

- every model input, as **training** would read it (HDF5 columns), and
- the score the **tray** computed from the frame
  (`L4_NoiseClassifier_ProbNu`, `L4_MuonClassifier_Data_ProbNu`).

For each model the script:

1. reads the inputs from the HDF5 through `data.load_one_file` — the same
   column resolution (`REGISTRY`/`ALTS`) and the same event matching as
   training;
2. recomputes the score with the same booster;
3. compares it event by event with the booked score: `|recomputed − booked|`.

The same float64 inputs and the same booster give **bitwise identical**
scores, so the default tolerance is 1e-9.  A frame read that found nothing
shows up as a difference wherever the missing value takes another branch of a
tree.  Exit status 0 only if each model was compared on at least one event
and no event differs by more than `--atol`.

---

## 4. File by file

---

## `oscnext_l4/varmap.py`

**What it is for.**  It loads `config/variables.json` and turns the one table
of rows into the dictionaries ("views") the rest of the code uses.  It also
holds the consistency check between the HDF5 side and the frame side.  It
imports only the standard library on purpose: `classifier.py` (runs inside
IceTray) and `data.py` (needs pytables) both read it, and neither can import
the other.

**Where it sits.**  Imported by `data.py` (HDF5 views), `classifier.py`
(`FEATURE_MAP`, `COLUMN_ALTS`), `train_L4_classifier.py` (the two input
lists) and `make_release.py`.  It calls nothing of ours.  The file is read
once, at import.

**Module-level names** (all built at import):

| name | type | content |
|---|---|---|
| `CONFIG_PATH` | str | `$OSCNEXT_L4_VARIABLES`, else `<repo>/config/variables.json` |
| `CONFIG` | dict | the parsed JSON |
| `VARIABLES` | dict | `CONFIG["variables"]` (46 rows) |
| `NOISE_FEATURES` | list | `bdt_features.noise`, 5 names, model order |
| `MUON_FEATURES` | list | `bdt_features.muon`, 10 names, model order |
| `KIND_TO_SCHEME` | dict | `kind_to_scheme` |
| `SCHEMES_WITHOUT_AUX` | tuple | `("data",)` |
| `REGISTRY` | dict | name → `(table, column)`, for every row **without** `schemes` (33 rows) |
| `AUX` | dict | name → `(table, column)`, for every row **with** `schemes` (13 weight columns) |
| `ALTS` | dict | name → `[primary pair] + hdf5_alts`, only for rows with `hdf5_alts` (5 rows: `micro_count`, `iLineFit_speed`, `fill_ratio`, `pdg`, `noise_weight`) |
| `AUX_SCHEMES` | dict | weight column → tuple of schemes |
| `FEATURE_MAP` | dict | name → `(frame key, field)`, for every row with a `frame` side (33) |
| `COLUMN_ALTS` | dict | `(frame key, field)` → list of `frame_alts` (5 entries) |

### `_is_aux(entry)`

Returns `True` if the row has a `schemes` field, i.e. it is a weight column,
not a physics variable.

### `check(verbose=True)`

Checks that the two sides of each variable still agree.  It reads only the
config.

1. For every row that has a `frame` side and is not a weight column: build the
   HDF5 accept-set `{hdf5} ∪ hdf5_alts` and the frame accept-set
   `{(key, field)} ∪ {(key, alt) for alt in frame_alts}`.  If the two sets
   share **no** pair, that row is a conflict.
2. For every name in `NOISE_FEATURES ∪ MUON_FEATURES`: a conflict if the name
   has no row, or its row has no `frame` side.
3. If `verbose`: print the conflicts, or "no conflict (33 variables with both
   sides, 13 weight columns)".

- **Returns:** the list of conflicts, as `(name, hdf5 side, frame side)`
  tuples.  Empty means consistent.
- **Watch out:** it compares names only.  It cannot see whether the frame
  object's Python binding has that field.  `check_application.py` is the test
  for that.

---

## `oscnext_l4/data.py`

**What it is for.**  It turns booked L4 HDF5 files into numpy arrays and gives
every event a physical weight.  It lists and checks tables, resolves which
column holds each variable, lines tables up with events, loads a whole sample,
and holds the five weighting schemes plus the detector-data livetime tools.

**Where it sits.**  Called by the notebook (sections 2, 4, 5),
`scripts/make_dataset.py` (loading and weights), `scripts/check_application.py`
(`load_one_file`), and `verification/pass2.py` (`_table_nodes`, `_index_node`,
`_ids`, so the cross-check reads files exactly as loading does).  It calls
`varmap` for the variable table, `pytables` for reading, and optionally
`simweights` and `h5py`.

**Constants and module state:**

| name | value | meaning |
|---|---|---|
| `IDX` | `("Run", "Event", "SubEvent")` | the event identifier columns (defined, not used in this file) |
| `REGISTRY`, `ALTS`, `AUX`, `AUX_SCHEMES`, `NOISE_FEATURES`, `MUON_FEATURES`, `SCHEMES_WITHOUT_AUX` | from `varmap` | re-exported under their old names |
| `WANTED` | sorted union of the two input lists and all of `AUX` | the default list `load_sample` is asked for |
| `NORM`, `GAMMA` | `2e-2`, `-3.0` | the GENIE training power law |
| `NU_FRAC`, `NUBAR_FRAC` | `0.7`, `0.3` | GENIE `gen_ratio` for ν and ν̄ |
| `NOISE_UNITS` | `{"per_ns": 1e9, "hz": 1.0}` | vuvuzela unit factors |
| `NOISE_WEIGHT_UNIT`, `NOISE_NS_SCALE` | `"per_ns"`, `1e9` | current setting (pass3 default); changed by `set_noise_weight_unit` |
| `DATA_LIVETIME_S` | `None` | detector-data livetime; set by `set_data_livetime` |
| `_MJD_COL` | `"time_start_mjd"` | the event-time column hdfwriter writes |
| `_MJD_COLS` | `("time_start_mjd_day", "..._sec", "..._ns")` | an older split layout, still accepted |
| `GRL_PATTERNS` | the `level2pass2` then `level2` GoodRunInfo paths, `%(y)d` = year | where Good Run Lists are looked for |
| `WEIGHTERS` | scheme → function | see §3.1 |
| `L3_RATES_MHZ` | `nue 0.95, numu 3.77, corsika 505.0, noise 36.6` | Table 13 L3 rates for the magnitude check |
| `_warned_unresolved` | set | remembers which warnings were already printed |

### `scheme_of(cfg)`

The weight scheme of one sample spec: `cfg["weight"]` if present, else
`KIND_TO_SCHEME[cfg["kind"]]`, else `None`.

### `aux_for(scheme)`

The weight columns expected under one scheme: every `AUX` name whose
`AUX_SCHEMES` entry contains `scheme`.  `aux_for("genie")` gives the six GENIE
columns; `aux_for("data")` gives `[]`.

- **Watch out:** it takes a **scheme** (`"genie"`), not a role (`"signal"`).
  A role returns an empty list.

### `_table_nodes(h5)`

Returns `{table name: node}` for the real data tables in an open pytables
file.

1. Walk every `Table` in the file, including subgroups.
2. Skip any whose path contains `__I3Index__`.
3. If a name still appears twice, keep the one closest to the root.

Why: hdfwriter writes `/X` and `/__I3Index__/X`.  Keyed by leaf name, the
index would overwrite the data, and every variable would come out `NaN`.

### `_index_node(h5, name)`

Returns the node `/__I3Index__/<name>`, or `None` if it does not exist.  Its
rows are one per frame, with `exists` and `start`/`stop`.  See §2.4.

### `dump_tables(h5path, only=None, max_cols=40)`

Lists the tables and columns of one HDF5 file.

1. If the file does not exist, print `MISSING:` and return `{}`.
2. For each real data table: record its row count and its columns, without
   `Run`, `Event`, `SubEvent`, `SubEventStream`, `exists`.
3. Print each table (sorted by name) and up to `max_cols` columns.  If
   `only` (a list of substrings) is given, print only tables whose name
   contains one of them; the return value still has all tables.

- **Returns:** `{table: (nrows, [columns])}`.  This is the `found` argument
  of `check_registry`.
- **Side effects:** prints.

### `check_registry(found, names, verbose=True, scheme=None)`

Checks that the columns for some variables really exist in a file.

1. For each name, `resolve_one` finds the first existing pair among its
   candidates (`ALTS`, else `REGISTRY`/`AUX`).
2. Not found → reported with the reason (no such table, or no such column,
   plus the first 12 columns that the table does have).
3. Found through an alternative → reported with `[i]`, naming the pair used
   and the first candidate that was absent.
4. If `names` is empty: when `scheme` is in `SCHEMES_WITHOUT_AUX`, it says
   "0/0, no weight columns by design"; otherwise it warns that nothing was
   checked and the caller's list is probably wrong.

- **Inputs:** `found` from `dump_tables`; `names` a list of variable names;
  `scheme` optional, when the names came from `aux_for`.
- **Returns:** the list of `(name, pair)` not found.

### `_ids(tbl)`

Returns the `Run`, `Event`, `SubEvent` columns of a table as three int64
arrays.

### `_has_duplicate_ids(r, e, s)`

Returns `True` if any `(Run, Event, SubEvent)` triple occurs more than once.
Empty input → `False`.

### `_sample_tag(path)`

Turns a part's file name into a sample tag, so warnings print once per sample
and not once per part.  `L4_nue_job3_part002.hdf5` → `L4_nue`.  It cuts the
base name at the first `_job` or `_part`.

### `resolve_one(name, available)`

Picks the `(table, column)` pair that really exists.

- **Inputs:** a variable name; `available = {table: set(columns)}`.
- The candidates are `ALTS[name]` if it exists, else the single pair from
  `REGISTRY` or `AUX`.  The first candidate present in `available` is
  returned.
- **Returns:** `(table, column)` or `None`.

### `load_one_file(path, wanted, extra=None)`

Reads the wanted variables from one HDF5 file.

1. Open the file once.
2. Build the node map (`_table_nodes`) and `available` columns.
3. Hand over to `_read_resolved`; close the file afterwards.

- **Inputs:** `wanted`, a list of variable names from the variable table;
  `extra`, optional `{name: (table, column)}` for columns **not** in the
  variable table (used by `check_application.py` for the booked scores).
- **Returns:** `{name: float64 array}` plus `Run`, `Event`, `SubEvent` (int64),
  one entry per `I3EventHeader` row.

### `_read_resolved(h5, path, nodes, available, wanted, extra=None)`

The body of `load_one_file`, with the file already open.

1. For each `extra` name: use its pair if present; otherwise mark it
   unresolved and print a warning (for every file).
2. For each `wanted` name: resolve it with `resolve_one`, and group the
   needed columns by table.  An unresolved name is warned about once per
   sample (no such table, or no such column plus the columns that exist).
3. Require an `I3EventHeader` table (else `RuntimeError`).  Its rows define
   the events; `Run`/`Event`/`SubEvent` come from it.
4. Fill every unresolved name with `NaN`.
5. Check whether the triples repeat inside the part; if so, print once that
   matching goes through `__I3Index__`.
6. For each table, build the row index `idx` as in §2.4 (index table →
   aligned identifiers → refuse on repeated triples → dictionary match).
7. Read each wanted column as float64 and place it by `idx`; events without
   an entry get `NaN`.

- **Returns:** the dict described under `load_one_file`.
- **Side effects:** prints warnings; remembers them in `_warned_unresolved`.

### `n_l3_files(h5path)`

Returns how many L3 files one HDF5 part was made from, from its `.meta.json`,
or `None`.  A thin wrapper around `_n_l3_files_why`.  (Nothing in the
repository calls it today; `load_sample` calls `_n_l3_files_why` directly.)

### `_n_l3_files_why(h5path)`

Returns `(count, reason)` for one part.

- `(n, "ok")` if `<h5path>.meta.json` exists and has a usable `n_l3_files`.
- `(None, "smoke")` if `n_l3_files` is `null` or `n_l3_files_unreliable` is
  true (a `--n` smoke run: expected, nothing is broken).
- `(None, "no_meta")` if there is no readable meta file (made by an older
  `process_L4.py`, or the job died): the weights may be wrong.

### `sample_files(name, SAMPLES, include_smoke=False)`

Finds the HDF5 parts of a sample: globs `SAMPLES[name]["hdf5"]` with
`.hdf5` replaced by `*.hdf5`, sorted.  Files with `_smoke` in the name are
dropped, unless nothing else exists (or `include_smoke=True`).

- **Returns:** a sorted list of paths.

### `find_hdf5(name, SAMPLES, verbose=True)`

Returns the first part of a sample (for `dump_tables`), or `None` with a hint
to run the booking first.  Prints how many parts were found.

### `_refuse_double_booking(name, files)`

Stops if one L3 file went into two of the parts about to be loaded.  That
happens when an old serial run's `L4_x_part000` sits beside a parallel run's
`L4_x_job0_part000`: both match the glob, so the same events would be loaded
twice (double weight, copies on both sides of the split).

1. For each part, read `l3_files` from its `.meta.json`.
2. If an L3 file has been seen in an earlier part, raise `RuntimeError`
   naming both parts.
3. Parts with no `l3_files` list cannot be checked; if there is more than one
   part, say how many.

### `load_sample(name, SAMPLES, wanted, max_files=None)`

Reads and concatenates every part of one sample.

1. Find the sample's scheme.  Drop from `wanted` every `AUX` column that this
   scheme does not use (so a CORSIKA file is not asked for `OneWeight`), and
   say which were skipped.
2. Find the parts (`sample_files`); keep the first `max_files` if given.  No
   parts → print and return `None`.
3. `_refuse_double_booking`.
4. `load_one_file` for each part; concatenate every key.
5. Add `_part`: the index of the part each event came from (0, 1, 2, ... in
   sorted file order).  Used for the (part, Run) grouping of the split.
6. The weight divisor: if every part has a usable meta count, `n_l3 = sum` and
   the source is `meta.json`.  Otherwise fall back to
   `SAMPLES[name]["n_l3_files"]` (the L3 glob count) or the number of parts,
   and print either a smoke-run note or a "weights may be WRONG" warning.
7. Add bookkeeping keys and print one summary line.

- **Returns:** `{variable: array, "Run", "Event", "SubEvent", "_part",
  "_n_files" (float), "_n_files_src" ("meta.json" or "fallback"), "_n_hdf5",
  "_files" (list of part paths), "_n_l3_per_file" (list, may hold None)}`,
  or `None`.
- **Watch out:** `make_dataset.py` refuses to build a set if any MC sample
  has `_n_files_src == "fallback"`.

### `set_noise_weight_unit(unit)`

Sets the vuvuzela weight unit: `"per_ns"` (factor 1e9) or `"hz"` (factor 1).
Changes the module globals `NOISE_WEIGHT_UNIT` and `NOISE_NS_SCALE`, prints
the choice, returns the factor.  Any other value raises `ValueError`.  Call it
before `add_weights`.

### `set_data_livetime(seconds)`

Declares the total livetime in seconds of the detector-data runs that were
booked.  Sets `DATA_LIVETIME_S`, prints it in seconds and days, and returns
it.  `None` unsets it.  Zero or negative raises `ValueError`.

- **Watch out:** it must cover exactly the runs whose events were loaded.

### `_event_times_s(tbl)`

Returns the event start times in seconds since MJD 0 from an `I3EventHeader`
table: `time_start_mjd × 86400` if that column exists, else the split
`day×86400 + sec + ns×1e-9` layout, else `None`.

### `livetime_from_headers(paths, verbose=True)`

Estimates detector livetime from the booked event times.

1. For each file: open it, read `/I3EventHeader`, the event times and `Run`.
   Files that cannot be opened or have no header/time column are listed as
   skipped.
2. For each run: keep the earliest and latest event time over all files, and
   count events.
3. Span per run = latest − earliest.  Total = sum of the spans (runs are
   summed, not merged, because they are far apart in time).
4. If `verbose`: print a table of run, events, span, rate; then the median
   rate, and every run whose rate is more than 30% away from the median (a sign
   of missing subruns).

- **Returns:** `(total seconds, {run: span})`.
- **Watch out:** an estimate.  It misses the head and tail of each run and
  includes dead time.  The Good Run List is the exact source.

### `_parse_grl(path)`

Reads one Good Run List file.

1. Find the header line: the first line with a `RunNum` token and a token
   starting with `LiveTime`.  Columns are found **by name**, not by
   position.
2. For every other line: read run number and livetime; read `Good_i3` if
   that column exists.  Lines that do not parse are skipped.

- **Returns:** `({run: (livetime_s, good_i3 or None)}, None)`, or
  `({}, reason)` if no header was found.

### `livetime_from_grl(runs, years=None, patterns=GRL_PATTERNS, verbose=True)`

The exact livetime of given runs from the Good Run Lists.

1. For each year (default 2011–2023), try each pattern in order; the first
   file that exists and parses is used for that year.
2. Each run is looked up.  A run in no list is **reported and excluded**,
   never assumed.  A run with `Good_i3 != 1` is reported but kept.
3. If `verbose`: print the lists read, a per-run table, the total in seconds
   and days, and warnings for runs not flagged good and runs missing.

- **Returns:** `(total seconds over found runs, {run: seconds})`.
- **Watch out:** if runs are missing, their events are still loaded, so a
  rate computed from this total is too high.  `make_dataset.py` then uses the
  event-time estimate instead.

### `data_weight(d)`

Detector data: `1 / DATA_LIVETIME_S` for every event.  If the livetime is
unset, every event gets 1.0 and a warning explains that training is
unaffected but the rate is not in Hz.

### `genie_weight(d)`

GENIE weight `OneWeight · 2e-2·E⁻³ / N_gen / n_files`, with `N_gen` from
`n_flux_events`, else `NEvents · gen_ratio` (read, or derived from the sign
of `pdg`).  Full description in §3.1.

- **Inputs:** `d` with `true_energy`, `OneWeight`, `NEvents`, `_n_files`, and
  optionally `n_flux_events`, `gen_ratio`, `pdg`.
- **Returns:** a float array (may contain `NaN`, which `add_weights` sets to
  0).
- **Raises:** `KeyError` if `N_gen` is needed and neither `gen_ratio` nor a
  finite `pdg` is available, or if the route it takes is missing for some of
  the events that need it (a partly NaN `gen_ratio` or `pdg`).  Those
  events would otherwise get a NaN weight, silently set to 0.
- **Side effects:** prints which route was taken and for how many events.

### `noise_weight(d)`

`noise_weight · NOISE_NS_SCALE / n_files`.  Prints the unit and the factor
every time.

### `_corsika_weight_manual(d)`

The CORSIKA fallback without simweights:
`Weight / (NEvents · OverSampling · n_files)`, with `OverSampling` set to 1
where it is not finite or ≤ 0.  Division warnings are suppressed.  Prints that
the absolute rate is unreliable.

### `_open_for_simweights(path)`

Opens a file in a form simweights accepts: pytables first, else h5py.
Returns `(file object, "tables" or "h5py")`; raises `RuntimeError` if both fail.

### `corsika_weight(d)`

CORSIKA weights with simweights and GaisserH3a (§3.1).

1. If simweights cannot be imported → `_corsika_weight_manual`.
2. If `d["_files"]` is empty, or any part has no L3 count → manual fallback.
3. `total` = sum of the per-part L3 counts.
4. For each part: read the event count and `/__I3Index__/CorsikaWeightMap`;
   compute simweights weights with `nfiles=total`; place them by the index
   (or row for row if counts match); otherwise leave `NaN`.  A failing part is
   reported and left `NaN`.
5. Concatenate in the same part order as `load_sample`.  If the length does
   not match the loaded events → manual fallback.
6. Print the share of events that got a weight; warn if under 99%.

- **Returns:** a float array, one per loaded event.

### `muongun_weight(d)`

The first of `MuonWeight`, `MuonWeight_GaisserH4a`, `MuonGunWeight` present
with any finite value, divided by `n_files`.  Prints which.  None found →
zeros and a warning.

### `add_weights(data, SAMPLES=None, weighters=None)`

Adds `w_phys` to every loaded sample (§3.1).

1. For each sample: find its scheme (`scheme_of(SAMPLES[name])`; without
   `SAMPLES`, the sample name is tried as a scheme).
2. No weighter for that scheme → `w_phys = NaN` everywhere and a hint to set
   `weight` in `productions.json`.
3. Otherwise compute the weights, set non-finite ones to 0, store as
   `d["w_phys"]`.
4. Print the total rate (Hz and mHz), max/sum (warning above 5%), and the
   Table 13 comparison for known sample names.

- **Returns:** `data`, changed in place.
- `weighters`, if given, replaces the `WEIGHTERS` table (nothing in the repository passes it).

### `check_feature_map(verbose=True)`

Calls `varmap.check(verbose)` and returns its list of conflicts.  Kept under
this name because the notebook calls it.

---

## `oscnext_l4/dataset.py`

**What it is for.**  It turns loaded, weighted samples into the `.npz`
training set of one classifier.  It stacks samples, reports missing inputs,
draws the grouped background split, balances the classes and writes the file.
It also loads a trained noise model as a function, for the noise cut.

**Where it sits.**  Called by `scripts/make_dataset.py` and by notebook
section 6, so both use the same code and the same constants.  It imports only
numpy at load time; `lightgbm` is imported inside `load_noise_prob`.

**Constants:**

| name | value | meaning |
|---|---|---|
| `RNG_SEED` | 12345 | signal split = first draw of `default_rng(12345)`; noise background = second draw; muon background uses `default_rng(12346)` |
| `TRAIN_FRAC` | 0.5 | probability of going to training |
| `NOISE_CUT` | 0.70 | P_noise threshold applied before the muon set |

### `stack(data, samples, features)`

Concatenates several samples into one dict.

- Each name in `features`, plus `w_phys` and `Run`, is concatenated across
  `samples` in the given order.
- If every sample has `_part`, adds `_group = (sample index << 32) + part` per
  event: a unique number per (sample, part).
- **Returns:** `{name: array}`.
- **Watch out:** `add_weights` must have run (it needs `w_phys`).  `Event` and
  `SubEvent` are not stacked.

### `report_missing_inputs(d, features, label)`

Counts events with at least one non-finite input.  If any, prints the total
and the count per feature.  It does **not** remove anything: LightGBM handles
missing values itself.  Returns the boolean mask of fully finite events.

### `split_by_shower(runs, frac, rng_, groups=None, allow_event_fallback=True)`

The grouped train/test split (§3.2).

1. If `groups` is given, each event's unit is its `(group, Run)` pair
   (numbered with `np.unique(..., return_inverse=True)`); otherwise its `Run`.
2. With fewer than 20 distinct units: raise `ValueError` if
   `allow_event_fallback` is `False`; otherwise print a warning and return an
   event-level split `rng_.random(n) < frac`.
3. Otherwise: one random number per unit; units with value `< frac` go to
   training; every event follows its unit.

- **Returns:** a boolean `istrain` array, one per event.

### `build_dataset(tag, sig, bg, features, sig_istrain, bg_istrain, outdir, provenance=None)`

Writes `<outdir>/L4_<tag>_dataset.npz`.

1. Build the balanced training weights from `w_phys` (§3.4).
2. Concatenate each feature (signal then background) as float64.
3. Add `label`, `weight`, `w_phys`, `istrain`, `features`, and, if given,
   `provenance` as a JSON string (keys sorted).
4. Save with `np.savez_compressed`; print train/test counts per class and the
   path.

- **Returns:** the path.

### `load_noise_prob(model_file)`

Loads a trained noise model as a function.

1. Load the LightGBM booster from the `.txt`.
2. Read its feature names from the booster itself (not from
   `NOISE_FEATURES`), so the columns are always fed in the model's order.

- **Returns:** `(noise_prob, features)`.

#### `noise_prob(d)` (inner function)

Builds the input matrix from `d[f]` for each feature in booster order and
returns `booster.predict(X)`: P(neutrino) per event.

### `sha256_of(path)`

Hex SHA-256 of a file, read in 1 MiB blocks.  Used to identify the config and
the noise model in the provenance.

---

## `scripts/make_dataset.py`

**What it is for.**  It builds the `.npz` training set of **one** classifier
from booked HDF5, without the notebook.  It does what notebook sections 1, 4,
5 and 6 do: choose the production and its samples, load, weight, stack, split,
and write.  It follows the production's order: build and train the noise set
first; the muon set is then built only from events that pass the noise model.

**Where it sits.**  After `process_L4.py --output-hdf5`, before
`train_L4_classifier.py`.  It calls `oscnext_l4.data` (loading, weights,
livetime) and `oscnext_l4.dataset` (stack, split, build, noise model).  Needs
numpy and pytables; lightgbm for `--stage muon`.  No IceTray.

**Constants:** `ROOT` = repository root (added to `sys.path`).  The seed,
fraction and noise cut come from `dataset.py`.

### `_code_version()`

Returns `git describe --always --dirty` of the checkout, or `None` outside
git or on any error.  Recorded in the provenance.

### `select_samples(config, production, stage, hdf_base)`

Builds `SAMPLES` for this stage, the way notebook section 1 does.

1. Exit if the production is not in the config.
2. Keep only samples whose `kind` is in `bdt_roles[stage]`.
3. For each: drop `__…__` keys, append `--cleaned-pulses <name>` to `flags`,
   set `hdf5 = <hdf_base>/<name>/L4_<name>.hdf5`.
4. Count the L3 files matching the `l3` glob(s) → `n_l3_files` (only used as
   a fallback divisor).
5. Print the production, the stage, and one line per sample.

- **Returns:** `(production dict, SAMPLES)`.

### `set_livetime(data, samples, grl_patterns)`

The detector-data livetime (§3.1).  For every loaded sample with scheme
`data`:

1. `livetime_from_headers` over its parts.
2. `livetime_from_grl` over its distinct runs.
3. If the GRL covers every run: print the GRL/span ratio and
   `set_data_livetime(GRL total)`.  Otherwise `set_data_livetime(span total)`.

- **Returns:** `{"source": "good_run_list" or "event_time_span", "seconds",
  "patterns"}`, or `None` if no data sample is loaded (always the case in the
  noise stage).
- **Exits** with a message if the GRL misses a run **and** the booked files
  give no event-time span: there is then no livetime to weight by.

### `_sample_record(name, cfg, d)`

One sample's provenance entry: name, kind, weight scheme, `l3` patterns (as a
list), event count, number of parts, and the L3 file count used as divisor.

### `provenance(args, config_path, prod, samples, data, signal, background, livetime, noise_cut)`

Builds the record of what made the training set.

| field | content |
|---|---|
| `schema` | 1 |
| `made_by`, `made_at`, `code_version` | script, time, `git describe` |
| `config`, `config_sha256` | path and hash of `productions.json` |
| `production`, `stage` | from the command line |
| `features` | the stage's input list |
| `cleaned_pulses` | from the production |
| `rng_seed`, `train_frac` | 12345, 0.5 |
| `signal`, `background` | lists of `_sample_record` |
| `signal_weighting` | the signal schemes; for `genie` also `norm`, `gamma`, `nu_frac`, `nubar_frac` as used |
| `noise_weight_unit` | the unit in force |
| `data_livetime` | the `set_livetime` record or `None` |
| `noise_cut` | the noise-cut record (muon stage) or `None` |

`main()` adds `signal_identity` afterwards.

### `signal_identity(data, signal)`

The hash that the shared split depends on (§3.2): SHA-256 over
`json([RNG_SEED, TRAIN_FRAC, signal names])` and then the int64 bytes of
`Run`, `Event`, `SubEvent` of each signal sample in order.

- **Returns:** `{"sha256": hex, "n_events": {sample: count}}`.

### `check_shared_split(noise_model, sig_id)`

Refuses a muon set whose signal differs from the noise set's.

1. Read `<noise_model without extension>.json` → `provenance.signal_identity`.
2. None recorded → warning, return.
3. Different hash → exit, printing both per-sample event counts.
4. Same → print that the split is shared.

### `refuse_dead_inputs(sig, bg, features)`

Exits if any input is non-finite in **every** event of either class.  LightGBM
would train on it happily, learning "missing" as the separator, and the tray
would read the same nothing, so no later check would fail.

### `main()`

1. Parse and check the flags (see table).  For `--stage muon`, exactly one of
   `--noise-model` and `--no-noise-cut` is required, and the model file must
   exist.  For `--stage noise`, neither is allowed.
2. Load `productions.json` (`--config`, else `$OSCNEXT_L4_CONFIG`, else the
   repository's).
3. `select_samples`; exit if the production declares no `noise_weight_unit`;
   `set_noise_weight_unit`.
4. `load_sample` for each sample (with `WANTED`); drop samples with 0
   events; list samples not loaded.  Exit if any non-data sample's divisor is
   not from `meta.json`.
5. `set_livetime`, then `add_weights`.
6. Stack the signal samples (all 14 BDT variables); compute
   `signal_identity`; draw the signal split (first draw of
   `default_rng(12345)`); create `--outdir`.
7. **Noise stage:** stack the `noise_bg` samples; report missing inputs;
   `refuse_dead_inputs`; split the background event by event (second draw);
   build the provenance plus `signal_identity`; `build_dataset("noise", ...)`.
8. **Muon stage:** collect `muon_bg` samples; if more than one, keep **one**,
   preferring scheme `data`.  Stack it.  If `--noise-model`:
   `check_shared_split`, load the model, apply `P ≥ 0.70` to signal (with its
   `istrain`) and background, print kept fractions, record the cut.  Report
   missing inputs; `refuse_dead_inputs`; split the background with
   `split_by_shower(..., default_rng(12346), groups=_group,
   allow_event_fallback = scheme != "corsika")`; build the provenance;
   `build_dataset("muon", ...)`.

**Output:** `<outdir>/L4_<stage>_dataset.npz`.

**Flags:**

| flag | default | meaning |
|---|---|---|
| `--stage` | required | `noise` or `muon` |
| `--production` | required | a production in the config: `pass2` or `pass3` |
| `--hdf-base` | required | root of the HDF5 tree, `<hdf-base>/<sample>/L4_<sample>*.hdf5` |
| `--outdir` | required | where `L4_<stage>_dataset.npz` is written |
| `--config` | `None` | productions file; else `$OSCNEXT_L4_CONFIG`, else `config/productions.json` |
| `--noise-model` | `None` | muon stage: the trained `L4_noise_model.txt`; only events with P_noise ≥ 0.70 enter |
| `--no-noise-cut` | off | muon stage without the noise cut (a recorded deviation) |
| `--grl-pattern` | `None` → `GRL_PATTERNS` | a Good Run List path pattern with `%(y)d` for the year; repeatable, tried in order |

**Example:**

```bash
python scripts/make_dataset.py --stage noise --production pass2 \
    --hdf-base /data/user/$USER/L4_output/hdf5_pass2 \
    --outdir   /data/user/$USER/L4_output/ds_pass2
python scripts/train_L4_classifier.py --tag noise \
    --dataset /data/user/$USER/L4_output/ds_pass2/L4_noise_dataset.npz \
    --outdir  models
python scripts/make_dataset.py --stage muon --production pass2 \
    --hdf-base /data/user/$USER/L4_output/hdf5_pass2 \
    --outdir   /data/user/$USER/L4_output/ds_pass2 \
    --noise-model models/L4_noise_model.txt
```

---

## `scripts/train_L4_classifier.py`

**What it is for.**  It trains one L4 classifier with LightGBM on a `.npz`
training set, using the Table 10 hyperparameters.  It evaluates on the test
set by event counts, measures the train/test gap, writes the model in
LightGBM's text format with a JSON sidecar, and draws three plots.  It is the
only training implementation; the notebook calls it as a subprocess.

**Where it sits.**  After `make_dataset.py` (or notebook section 6).  Its
outputs are read by `classifier.py` in the tray, by `make_dataset.py --stage
muon` (noise model and its `.json`), by `check_application.py` and by
`make_release.py`.  Needs numpy and lightgbm (+ matplotlib).  It imports
`NOISE_FEATURES`/`MUON_FEATURES` from `varmap` to check the dataset's input
list.

**Constants:**

| name | value | meaning |
|---|---|---|
| `BDT_FEATURES` | `{"noise": NOISE_FEATURES, "muon": MUON_FEATURES}` | the expected input lists |
| `PARAMS` | see §3.5 | hyperparameters per tag |
| `EARLY_STOPPING` | 100 | rounds without improvement before stopping |
| `DEFAULT_THREADS` | 0 | 0 = LightGBM uses all cores |
| `DEFAULT_CUT` | `{"noise": 0.70, "muon": 0.65}` | the note's cut values |
| `REJ_LEVELS` | `[0.90, 0.95, 0.99, 0.995, 0.999]` | rows of the results table |
| `RESERVED_COLS` | `w_phys, weight, istrain, features, livetime, Run, Event, SubEvent, provenance` | `.npz` keys that are never model inputs |

`deterministic=True` makes runs reproducible **for a given thread count**.
Different thread counts can differ in the last bits.

### `_sha256(path)`

Hex SHA-256 of a file (1 MiB blocks).  Used for `model_sha256`.

### `load_dataset(path, features=None)`

Reads the `.npz`.

1. Exit if the file is missing.
2. The features are, in order of preference: the `features` argument, the
   `features` array in the file, or every non-reserved key except `label`,
   sorted.
3. Exit if a feature is missing or is a reserved name.
4. Build `X` (float), and read `provenance` (JSON, or `None`), `label`,
   `weight` (NaN → 0), `w_phys` (NaN → 0; a copy of `weight` if absent),
   `istrain`.
5. Print how many events have a non-finite input (not fatal).

- **Returns:** `(X, y, w, wp, tr, features, provenance)`.

### `_kept_at(sorted_scores, cuts)`

How many scores are `≥` each cut, for many cuts at once:
`n − searchsorted(sorted_scores, cuts, side="left")`.  `sorted_scores` must be
sorted ascending.

### `curve(s, b)`

The cut scan on event counts.

1. Cuts = distinct background scores, plus `min(b) − 1` and `max(b) + 1`.
2. `eff` = fraction of signal scores ≥ cut; `rej` = 1 − fraction of
   background scores ≥ cut.

- **Returns:** `(cuts, eff, rej)`, sorted by cut.  O(n log n).

### `kept_counts(s, b, target)`

The best cut for a target rejection: among cuts with `rej ≥ target`, the one
with the highest efficiency.

- **Returns:** `{cut, eff, rej, sig_kept, sig_total, bg_kept, bg_total}`, or
  `None` if no cut reaches the target.  The counts are recovered from the
  ratios by rounding.

### `eff_at_rejection_weighted(s, ws, b, wb, target)`

The weighted version, for the `w_phys` cross-check.  Sort the background by
score, take the cumulative weight fraction, place the cut at the first
background score where it reaches `target`.

- **Returns:** `(cut, weighted signal efficiency, weighted background
  rejection)`; all `NaN` if either class has zero total weight.

### `report_table(s, b, levels=REJ_LEVELS)`

Prints the efficiency-vs-rejection table (§3.5) and returns its rows as dicts
(`rejection` plus the `kept_counts` fields).  Unreachable levels print
`cut unreachable` and are left out of the returned rows.

### `_setup_matplotlib()`

Imports matplotlib with the non-interactive `Agg` back end and returns
`pyplot`.

### `make_plots(scores, weights, tag, outdir, weight_name, n_bg_te)`

Draws three PNGs in `outdir`:

- `<tag>_overtrain.png`: the four score distributions (train/test ×
  signal/background), each normalised to unit area, log y.  Train as filled
  histograms, test as points.
- `<tag>_dist.png`: the same four, not normalised, weighted by
  `weight_name`, in linear and log y.
- `<tag>_cuts.png`: left, efficiency and rejection vs. the cut, with the note's
  cut as a vertical line; right, efficiency vs. rejection (80–100%), with
  vertical lines where 100, 10 and 1 test background events are left, and for
  `noise` the Table 13 target (99.2%, 96%) as a star.

- **Returns:** the list of paths written.

#### `save(fig, name, suptitle=False)` (inner function)

Lays out the figure (leaving room for a super-title if asked), saves it at
130 dpi into `outdir`, closes it, records and prints the path.

### `main()`

1. Parse the flags.  The target rejection defaults to 0.99 (noise) or 0.94
   (muon).  `--features` is split on commas.
2. `load_dataset`.  Without `--features`, exit unless the stored features
   equal `bdt_features[tag]` **including order**.  Warn if there is no
   provenance; exit if the provenance says the set was built for the other
   stage.
3. Build the parameters: copy `PARAMS[tag]`, add `num_threads` if set, take
   `num_boost_round` out (overridable), apply `--learning-rate`,
   `--min-data-in-leaf`, `--seed`.
4. Print set sizes, features and parameters.
5. Early stopping: carve a per-class validation slice out of the training set
   (§3.5).  Warn if fitted background < 10 × `min_data_in_leaf`.
6. `lgb.train` with `weight` as sample weights, `log_evaluation(--log-every)`,
   and `early_stopping(100)` if there is a validation slice.  Report the
   number of trees used; note if the round limit was reached.
7. Score train and test, signal and background.
8. Print the table, the headline, the gap, the weighted cross-check, the
   numbers at the note's cut, and the feature importances (gain and split, at
   the best iteration).
9. Save `L4_<tag>_model.txt` (best iteration only), the plots (a plotting
   error is caught and printed; the model and numbers are already safe), and
   `L4_<tag>_model.json` (§3.5).

#### `score(mask)` (inner function)

`booster.predict(X[mask])`.  `predict` uses the best iteration after early
stopping.

**Outputs:** `<outdir>/L4_<tag>_model.txt`, `<outdir>/L4_<tag>_model.json`,
`<outdir>/<tag>_overtrain.png`, `<tag>_dist.png`, `<tag>_cuts.png`.

**Flags:**

| flag | default | meaning |
|---|---|---|
| `--tag` | required | `noise` or `muon` |
| `--dataset` | required | the `.npz` from `make_dataset.py` or the notebook |
| `--outdir` | required | where the model, sidecar and plots go |
| `--features` | `None` | comma-separated input list; default is the `.npz`'s `features`.  The one deliberate way to train on a non-standard list. |
| `--target-rejection` | `None` → 0.99 noise, 0.94 muon | rejection of the headline number |
| `--gap-at` | 0.90 | rejection at which the train/test gap is measured |
| `--valid-frac` | 0.2 | share of the training set used for early stopping |
| `--no-early-stopping` | off | train exactly `num_boost_round` rounds |
| `--num-boost-round` | `None` → 2000 | round limit |
| `--learning-rate` | `None` → 0.05 | override |
| `--min-data-in-leaf` | `None` → 500 | override Table 10 (for a small background sample) |
| `--plot-weight` | `weight` | `weight` or `w_phys`, the weight used in `<tag>_dist.png` and `<tag>_overtrain.png` |
| `--no-plots` | off | skip the plots |
| `--seed` | 12345 | LightGBM seed and the validation-slice draw (not the train/test split, which is in the `.npz`) |
| `--num-threads` | 0 | LightGBM threads; 0 = all cores |
| `--log-every` | 50 | print the validation metric every N rounds; 0 = silent |

**Example:**

```bash
python scripts/train_L4_classifier.py --tag noise \
    --dataset L4_output/ds/L4_noise_dataset.npz \
    --outdir  L4_output/models
```

---

## `scripts/check_application.py`

**What it is for.**  It checks that the tray scores an event exactly as the
training side would.  It recomputes each model's score from the booked HDF5
columns and compares it, event by event, with the score the tray wrote into
the frame (§3.6).  It is the only test that reaches the level of the frame
object's Python binding.

**Where it sits.**  Run after `process_L4.py --apply-cut --output-hdf5 ...
--model-dir models`.  It calls `data.load_one_file` for the columns and
lightgbm for the scores.  Needs numpy, pytables, lightgbm; no IceTray.  It is
shipped with the release (`make_release.py`).

**Constants:**

| name | value | meaning |
|---|---|---|
| `ROOT` | repository root | for imports and to find `classifier.py` |
| `SCORE_KEYS` | `noise → L4_NoiseClassifier_ProbNu`, `muon → L4_MuonClassifier_Data_ProbNu` | the frame keys the tray writes the scores to |

### `_check_score_keys()`

Reads `oscnext_l4/classifier.py` as text and exits if either score key no
longer appears there as a quoted string.  This keeps `SCORE_KEYS` in step with
the tray without importing IceTray.

### `load_model(model_dir, tag)`

Loads `L4_<tag>_model.txt` and `L4_<tag>_model.json` from `model_dir` (exits
if either is missing).  Exits if the `.json`'s `features` differ from the
booster's own feature names.  Returns `(booster, features)`.

### `main()`

1. Parse the flags; check the score keys; expand glob patterns; exit if a
   file is missing.
2. Load **both** models (both must exist).  `wanted` = the union of their
   features.  `extra` = the two booked scores, as `(key, "value")`.
3. For each file: `load_one_file(path, wanted, extra)`.  For each model:
   build `X` in the model's feature order; take the booked score; count
   events, events without a booked score, and events with a `NaN` input; for
   events with a score, compute `|predict(X) − booked|`, count those above
   `--atol`, track the maximum, and keep the worst `--show` events.
4. Print one summary row per model (events, compared, no score, NaN inputs,
   max \|diff\|, over tolerance).  A model with no compared event, or with
   any event over tolerance, makes the run **FAIL**; the worst events are
   listed by file, Run, Event, SubEvent.
5. Exit 0 on PASS, 1 on FAIL.

**Flags:**

| flag | default | meaning |
|---|---|---|
| `hdf5` (positional, one or more) | required | HDF5 file(s) from `process_L4.py --apply-cut --output-hdf5`; glob patterns accepted |
| `--model-dir` | required | the models the tray ran with |
| `--atol` | 1e-9 | largest acceptable \|recomputed − booked\| |
| `--show` | 5 | how many of the worst events to print per model |

**Example:**

```bash
python scripts/process_L4.py --gcd GCD --input L3.i3.zst \
    --output-hdf5 check.hdf5 --apply-cut --model-dir models [flags]
python scripts/check_application.py check.hdf5 --model-dir models
```

---

## `scripts/check_leakage.py`

**What it is for.**  It looks for train/test leakage in a finished `.npz`:
rows that are exact copies of each other and sit on **both** sides of the
split.  The usual cause is a stale HDF5 part that loads the same L3 file
twice.  It also says which leakage mechanisms it cannot see from the `.npz`
alone (sub-events of one DAQ event, CORSIKA oversampling), since
`Run`/`Event`/`SubEvent` are not stored there.

**Where it sits.**  Run by hand on the output of `make_dataset.py`.  numpy
only, so it also runs in the IceTray environment.

### `bytes_key(cols)`

Stacks the columns into a float64 matrix and views each row as one structured
record, so identical rows compare equal byte for byte (and a `NaN` matches
the same `NaN` bit pattern).  Returns one key per row.

### `straddle_report(key, istrain, label, title)`

Groups identical keys and reports:

- rows, distinct rows, rows in a duplicate group;
- rows in a duplicate group that **crosses** the split (some copies in
  train, some in test) — the leakage;
- for crossing groups: how many, their sizes, and how many of the first 2000
  are signal.

Returns the number of rows in crossing groups.

### `main()`

1. Needs exactly one argument (the `.npz`); otherwise prints the docstring
   and returns 2.
2. Prints the dataset's features and train/test and signal/background counts.
3. `straddle_report` on all features **plus the label** (a signal and a
   background row with equal values are not the same event).
4. If some features are continuous (not all integer-valued) and some are
   not, repeats the report on the continuous ones only.  Integer features
   collide by chance; if both reports agree, the duplicates are real.
5. Prints a verdict, and the mechanisms it cannot check.  Returns 0.

- **Watch out:** a feature is counted as "continuous" whenever
  `array_equal(x, round(x))` is false, and that is also false for any column
  that contains `NaN`.

**Arguments:** one positional argument, the `.npz` path.  No flags.

**Example:**

```bash
python scripts/check_leakage.py /path/to/L4_noise_dataset.npz
```

---

## `scripts/plot_inputs.py`

**What it is for.**  It plots the BDT input distributions, signal against
background, from a `.npz` — the counterpart of Figure 13 of the technical
note.  Each panel is normalised to unit area and titled with the variable's
single-variable AUC and, if any, its `NaN` fraction.  It needs neither the
model nor the HDF5.

**Where it sits.**  Run by hand, or by `presentation/make_figures.py`.  numpy
and matplotlib.

**Constants:**

| name | value | meaning |
|---|---|---|
| `C_SIG`, `C_BG` | `tab:blue`, `tab:red` | signal and background colours |
| `AXES` | per-variable axis hints | `iLineFit_speed`: log x, 1e-3–1e3; `fill_ratio`, `FullTimeLengthRatio`: 0–1; `NchCleaned`, `micro_count`, `ICVetoHits`, `NAbove200Hits`, `RTVeto250Hits`, `VICH_nch`: integer bins.  Others use the 0.5–99.5 percentile range. |

### `single_auc(s, b, rng, cap=200000)`

The AUC of one variable on its own, by ranks (Mann–Whitney).  Non-finite
values are dropped; each class is subsampled to at most `cap` events; tied
values get their average rank.  Returns `max(auc, 1 − auc)`, so the direction
does not matter.  `NaN` if a class is empty.

### `edges_for(name, vals, nbins)`

Chooses the bin edges for one variable.

1. No finite values → `linspace(0, 1)`.
2. Range from `AXES`, else the 0.5 and 99.5 percentiles.
3. Log axis: log-spaced edges from `max(lo, smallest positive value)` to `hi`
   (falls back to linear if there are no positive values).
4. Integer variable: unit-width bins centred on integers, starting at
   `floor(lo)`, at most `nbins` of them.
5. Otherwise `nbins` linear bins.

- **Returns:** `(edges, logx)`.
- **Watch out:** values outside the chosen range are left out of the
  histogram (and of its normalisation).  For an integer variable whose range
  is wider than `nbins`, only the first `nbins` integers from `lo` are drawn.

### `main()`

1. Parse the flags; load the `.npz`; pick the weight (`w_phys`, `weight`, or
   ones for `none`; non-finite or negative → 0).
2. One panel per feature (3 columns if more than 4 features): signal as a
   filled step, background as a line, each normalised to unit area, log y.
3. Print and title the AUC and the `NaN` percentage per class.
4. Save `<outdir>/<tag>_inputs.png`; the tag defaults to the file name with
   `L4_` and `_dataset.npz` removed.

**Flags:**

| flag | default | meaning |
|---|---|---|
| `dataset` (positional) | required | the `.npz` |
| `--outdir` | required | output directory |
| `--tag` | from the file name | name used for the output file |
| `--weight` | `w_phys` | `w_phys` (rate-weighted shapes, as the note's figures), `weight` (balanced training weight), or `none` (raw counts) |
| `--bins` | 60 | number of bins |
| `--seed` | 12345 | seed for the AUC subsampling |

**Example:**

```bash
python scripts/plot_inputs.py L4_output/ds/L4_noise_dataset.npz --outdir plots
python scripts/plot_inputs.py L4_output/ds/L4_noise_dataset.npz --outdir plots --weight none
```

---

## `scripts/inspect_production_table.py`

**What it is for.**  It dumps the production's own classifier training tables
(for example
`/data/ana/LE/oscNext/pass2/resources/classifier_models/level4_muon/L4_muon_model_data.hdf5`),
which hold the actual train/test events the released models were fitted on.
It reads the file and prints; it changes nothing.  It stands apart from the
pipeline: nothing calls it.

**Where it sits.**  Run by hand.  pytables and numpy.

### `walk(h5)`

Returns `(path, class name, node)` for every `Table`, `Array`, `CArray` or
`EArray` in the file.  The layout is not assumed.

### `main()`

For each leaf:

1. Print its path, kind and row count.
2. For an array: print shape, dtype and the first `--rows` entries.
3. For a table: print its columns (up to `--max-cols`); for columns whose
   lower-case name looks like a class or split label (`class`, `label`, `y`,
   `subsample`, `istrain`, `train`, `type`, `dataset`, `category`), print the
   distinct values and their counts (first 20); for run columns (`run`,
   `run_id`, `runid`), print the number of distinct runs and the first 40;
   then the first `--rows` rows.

Returns 0.

**Flags:**

| flag | default | meaning |
|---|---|---|
| `path` (positional) | required | the HDF5 file |
| `--rows` | 5 | example rows printed per table |
| `--max-cols` | 200 | most column names printed per table |

**Example:**

```bash
python scripts/inspect_production_table.py \
    /data/ana/LE/oscNext/pass2/resources/classifier_models/level4_muon/L4_muon_model_data.hdf5 --rows 5
```


---

# Part D — Interfaces and checks

This part covers the files a person actually drives, and the checks that sit
around the pipeline rather than inside it:

1. **`notebooks/oscNext_L4.ipynb`**: the main interface.  It runs the whole
   chain from L3 files to trained classifiers, section by section.
2. **`verification/`**: the pass2 cross-check.  It runs our pipeline on the
   pass2 L3 files and compares every variable, event by event, with the
   values the official pass2 L4 files already contain.
3. **`scripts/make_release.py`** and the templates in **`release/`**: how the
   self-contained package for a collaborator is assembled.
4. **`presentation/make_figures.py`** and **`presentation/make_update_pdf.py`**:
   two small helpers for talks.

None of these files computes a physics variable.  The notebook calls the
package (`oscnext_l4/`) and the scripts in `scripts/`; Parts B and C describe
those functions in detail.  Here each call is named, with what it produces,
so you can follow the chain without leaving this part.

A few words used throughout:

- **Sample**: one simulated or measured dataset, for example `nue` (GENIE
  electron neutrinos) or `noise` (vuvuzela pure-noise simulation).  Each
  sample is one entry in `config/productions.json`.
- **Role** (`kind` in the config): what a sample is used as.  `signal`,
  `noise_bg` (the noise classifier's background) or `muon_bg` (the muon
  classifier's background).
- **Weight scheme** (`weight` in the config): which formula weights the
  sample: `genie`, `noise`, `corsika`, `muongun` or `data`.
- **Part**: one HDF5 output file holding a slice of a sample's L3 files
  (`L4_nue_part000.hdf5`, ...).  Each part has a `<part>.hdf5.meta.json`
  beside it that records how many L3 files went in.
- **Sidecar**: the `.json` file written next to a trained LightGBM model
  (`L4_noise_model.json`).  It holds the feature order, the training record
  and, when known, the provenance (which samples made the model).
- **Booking**: writing frame objects into HDF5 tables with hdfwriter.  Every
  booked key gets a data table `/<key>` and an index table
  `/__I3Index__/<key>`.

---

## notebooks/oscNext_L4.ipynb

**What it is for.**  The notebook is the one place where a person runs the
whole L3 → L4 chain: processing, checking the booked files, loading, weighting,
building training sets, training, and reading the results.  It is a thin
interface: every real step is a call into `oscnext_l4/` or a subprocess
running a script in `scripts/`.  It defines only two functions of its own
(`_default_output_root` in cell 3 and `run_train` in cell 29).

**Where it sits.**  It needs the IceTray environment only for section 1
(processing), because that section runs `scripts/process_L4.py` in a
subprocess.  Sections 2-9 need `numpy`, `tables` (pytables), `lightgbm` and
`matplotlib`.  It does not use `pandas`.

**Cell numbers.**  Cells are numbered here by their position in the notebook
file, counting from 0 and including Markdown cells.  That is the numbering
`CLAUDE.md` uses ("cell 3 carries `PRODUCTION`").  Jupyter does not display
these numbers; the section headings are the landmarks.

### The settings a user changes

There are only a few, and all but one sit in the first cells.

| setting | cell | values | what it controls |
|---|---|---|---|
| `PRODUCTION` | 3 | `"pass3"` or `"pass2"` (the file ships with `"pass2"`) | Which production the whole notebook works on: the GCD, the sample paths, the cleaned pulse series and the noise weight unit (all read in cell 5), and the output directory names. |
| `OSCNEXT_OUT_ROOT` (environment variable) | 3 | a directory | Where all outputs go.  Without it: `/data/user/$USER/L4_output` if that exists and is writable, else `<repo>/L4_output`. |
| `OSCNEXT_L4_ROOT` (environment variable) | 3 | the repository path | Only needed when the notebook cannot find the repository by itself. |
| `OSCNEXT_L4_CONFIG` (environment variable) | 5 | a JSON file | Use a different production table than `config/productions.json`. |
| `STAGE` | 5 | `"noise"`, `"muon"` or `"all"` (ships with `"noise"`) | Which samples are loaded and which classifier is trained.  See "The stages" below. |
| `RELOAD` | 17 | `True` / `False` | Force section 4 to re-read every HDF5 file instead of keeping what is already in memory. |
| arguments of `run_all(...)` | 10 | `jobs`, `chunk_files`, `max_files`, `fraction`, `run_optional`, `extra_args` | How the processing runs (parallel workers, part size, taking a slice). |

Output directories depend on `PRODUCTION`.  pass3 uses the bare names; any
other production gets a suffix, so the two never overwrite each other:

| variable | pass3 | pass2 | holds |
|---|---|---|---|
| `HDF_BASE` | `<out>/hdf5` | `<out>/hdf5_pass2` | `process_L4.py` output, one subdirectory per sample |
| `DS_BASE` | `<out>/ds` | `<out>/ds_pass2` | the training sets `L4_<tag>_dataset.npz` |
| `MODEL_DIR` | `<out>/models` | `<out>/models_pass2` | `L4_<tag>_model.txt`, `.json`, and the plots |

### The stages

The production trains the two classifiers in order: first the noise
classifier, then it applies that model, and trains the muon classifier only
on the events that pass the noise cut (P(ν) ≥ 0.70).  `STAGE` follows that
order.

- `STAGE = "noise"` loads the signal and noise-background samples, builds the
  noise training set, and trains the noise model.
- `STAGE = "muon"` loads the signal and muon-background samples, applies the
  noise model already in `MODEL_DIR` to both classes, builds the muon training
  set from the survivors, and trains the muon model.  The noise model is not
  retrained.
- `STAGE = "all"` loads every sample, for inspection.  It builds the noise
  set but stops at cell 25: the muon set has to be cut with the noise model
  that section 7 trains, which does not exist yet at that point.

The normal sequence is: run the notebook with `STAGE = "noise"`, then set
`STAGE = "muon"` and run it again from the top.

> **Why `"all"` stops at cell 25.**  Cell 25 would load the noise model
> that is *already on disk*, before cell 29 trains the new one.  The muon set
> would then be cut with an old noise model (or with none), and the new muon
> model would ship beside a noise model it was never cut with.  Cell 25
> therefore raises under `"all"` and names the two-stage sequence.

### Resuming after a kernel restart

A kernel restart loses every variable, but not the working directory (that
comes from the Jupyter server).  What to re-run depends on what you want:

| goal | cells to run |
|---|---|
| anything at all | 2 and 3 (imports, paths, `PRODUCTION`), then 5 (config, `STAGE`, `SAMPLES`, noise weight unit) |
| process more files | also 6 (`configure_runner`), then section 1 |
| train from `.npz` files already on disk | 28, then 29 and/or 30.  Sections 2-6 are not needed; cell 28 finds the `.npz` files in `DS_BASE`. |
| look at trained models | sections 8 and 9 only need cells 2 and 3 (`MODEL_DIR`).  Cell 38 also needs cell 15, which imports `check_feature_map`. |
| rebuild the training sets | sections 4, 5 and 6, **with cell 3 run first** (see the watch-out in section 6) |

Section 4 is slow on pass2 (hours for all samples).  Its cell keeps what is
already loaded, but a restart empties memory, so after a restart it reads
everything again.

---

### Section 0 — Configuration and environment check

**Cell 0 (Markdown).**  Title and a short statement that the notebook is a
thin interface and that the kernel must be the Python inside IceTray's
`env-shell.sh`.

**Cell 1 (Markdown).**  Says `lightgbm` and `tables` are required and
`pandas` is not used.

**Cell 2 (code): imports and dependency check.**

1. Turns on IPython's `autoreload` (mode 2), so edits to `oscnext_l4/*.py`
   are picked up without a restart.  If that fails (for example outside
   IPython), it is silently skipped.
2. Imports `os, sys, glob, json, shlex, subprocess, time` and `numpy as np`.
3. Imports `lightgbm`.  If that fails it stops the cell with `SystemExit` and
   a hint to test `import lightgbm` inside the environment.
4. Imports `tables` (pytables).  Stops with `SystemExit` if missing.
5. Tries `matplotlib`, `simweights` and `ipywidgets` and prints `OK` or
   `MISSING` with what is lost: no plots; CORSIKA weights fall back to the
   manual formula; the progress bar falls back to ASCII.
6. Imports `matplotlib.pyplot as plt` and sets `figure.dpi = 110`,
   `font.size = 9`.

*Check:* the `lightgbm` and `tables` lines say `OK`.

> **Watch out.**  Step 6 imports matplotlib unconditionally.  The cell
> reports matplotlib as "optional", but without it the cell ends in an
> `ImportError`.

**Cell 3 (code): paths and production.**

1. Finds the repository root: `OSCNEXT_L4_ROOT` if set, else the current
   directory or its parent, whichever contains `oscnext_l4/`.  Stops with a
   clear message if neither does.  Puts the root first on `sys.path`.
2. Defines and calls `_default_output_root()` (below) to set `OUTPUT_ROOT`.
3. Sets `PRODUCTION` (edit this line) and the suffix `_SUF` (`""` for pass3,
   `"_pass2"` for pass2).
4. Sets `HDF_BASE`, `DS_BASE`, `MODEL_DIR` and creates the three directories.
5. Sets `PROCESS_PY` and `TRAIN_PY`, the absolute paths of
   `scripts/process_L4.py` and `scripts/train_L4_classifier.py`.
6. Imports `RNG_SEED` (12345) from `oscnext_l4.dataset` and creates
   `rng = np.random.default_rng(RNG_SEED)`.  This generator draws the
   train/test splits in section 6.
7. Prints the repository root, the output root, the free disk space (marked
   `TIGHT` below 2000 GB) and whether the two scripts were found.

*Produces:* `REPO_ROOT`, `OUTPUT_ROOT`, `PRODUCTION`, `HDF_BASE`, `DS_BASE`,
`MODEL_DIR`, `PROCESS_PY`, `TRAIN_PY`, `rng`.

*Check:* the output root is where you expect (not the repository for a real
run), and both scripts say `found`.

#### `_default_output_root()` (defined in cell 3)

- **What it does:** decides where outputs go when nothing was said.
  1. If `OSCNEXT_OUT_ROOT` is set, returns it.
  2. Else, if `/data/user/$USER` exists and is writable, returns
     `/data/user/$USER/L4_output`.
  3. Else prints a warning (fine for a smoke test, not for production) and
     returns `<repo>/L4_output`.
- **Inputs → outputs:** none → a directory path (string).
- **Why:** the HDF5 files run to tens of GB, and `/home` on cobalt is a
  shared, nearly full filesystem, while `/data/user` has room.

---

### Section 1 — L3 to L4 processing

**Cell 4 (Markdown).**  Explains that this section runs
`scripts/process_L4.py` per sample, takes hours, and is done once.
`--apply-cut` is not used here: before the models exist, every event must be
booked.

**Cell 5 (code): which production and what to load.**

1. Imports `set_noise_weight_unit` from `oscnext_l4.data`.
2. Reads `config/productions.json` (or `OSCNEXT_L4_CONFIG`) with
   `json.load` into `CONFIG`.
3. Sets `STAGE` (edit this line) and turns it into `ROLES`:
   `CONFIG["bdt_roles"]["noise"]` (`signal`, `noise_bg`),
   `CONFIG["bdt_roles"]["muon"]` (`signal`, `muon_bg`), or `None` for `"all"`.
4. Picks the production block `CONFIG["productions"][PRODUCTION]` and reads
   `GCD` and `CLEANED_PULSES` from it.
5. Calls `set_noise_weight_unit(_prod["noise_weight_unit"])`.  This sets the
   unit of the vuvuzela weight column: `"per_ns"` (pass3, factor 1e9) or
   `"hz"` (pass2, factor 1).  It is set here, beside the production switch,
   because a wrong unit raises nothing and scales every noise rate by 1e9.
6. Builds `SAMPLES`: for every sample of the production whose `kind` is in
   `ROLES` (all samples if `ROLES` is `None`):
   - copies its spec, dropping keys that start with `__` (comments);
   - copies `flags` and, if the production names a cleaned pulse series,
     appends `--cleaned-pulses <name>` (this is the whole pass2 difference);
   - sets `hdf5` to `HDF_BASE/<sample>/L4_<sample>.hdf5`.
7. Prints the production, stage, GCD, cleaned pulses and HDF5 base.
8. For each sample, globs its `l3` pattern(s), stores the count in
   `SAMPLES[name]["n_l3_files"]`, and prints `name, kind, count`.  Section 4
   uses this count only as a fallback divisor when `.meta.json` files are
   missing.
9. Warns about samples with zero L3 files.

*Produces:* `CONFIG`, `STAGE`, `ROLES`, `NOISE_BDT_ROLES`, `MUON_BDT_ROLES`,
`GCD`, `CLEANED_PULSES`, `SAMPLES` (a dict: sample name → spec with `l3`,
`flags`, `kind`, `weight`, `hdf5`, `n_l3_files`, and `gcd` for detector data).

*Check:* the `noise_weight unit` line matches the production, every sample
has a non-zero L3 file count, and the list of samples is what the stage
should load.

**Cell 6 (code): register the runner.**  Imports `configure_runner`,
`run_process`, `run_all` and `run_process_per_run` from `oscnext_l4.runner`
and calls `configure_runner(SAMPLES, PROCESS_PY, GCD)`.  That stores the three
values inside the runner module and checks that `process_L4.py` and the GCD
exist and that the working directory is not in the trash.

*Check:* it prints `configure_runner OK`.  Any `[!]` line here means later
subprocesses would die with an unhelpful `returncode=2`.

**Cell 7 (Markdown).**  Smoke test first.  Explains that `--n` counts frames,
not events: 200 frames giving about 60 booked events is normal.

**Cell 8 (code): smoke test.**  `smoke = run_process("nue", n_frames=200)`.
This runs `process_L4.py` on the `nue` sample with `--n 200 --scan off` and
no parts, writing `HDF_BASE/nue/L4_nue_smoke.hdf5`.  A progress bar is shown
while it runs; at the end it prints the last lines of the log.

*Check:* the log shows each stage (Physics frames, `InIceSplit`, after the L3
cut, events booked).  An `InIceSplit` count of 0 means the sub-event stream
is wrong; an L3 line of 0 means the input is not L3 output.

**Cell 9 (Markdown).**  Full production: `chunk_files=10` makes one part per
10 L3 files, which gives a real percentage and ETA and makes the run
resumable.  Corrupt files are dropped by `--scan quick` and `--retries`.

**Cell 10 (code): full production.**  `results = run_all(jobs=8,
chunk_files=10)`.  For every sample in `SAMPLES`, `run_all`:

- sends detector data (a sample whose spec has a `gcd` **list**, one GCD per
  run) to `run_process_per_run`: one `process_L4.py` per run, each with its
  own GCD, output named after the run;
- otherwise, with `jobs > 1`, runs `run_process_parallel`: the L3 files are
  split into `jobs` groups, each processed by its own `process_L4.py`
  (outputs `L4_<sample>_job<j>_part<k>.hdf5`);
- with `jobs = 1`, runs `run_process` (outputs `L4_<sample>_part<k>.hdf5`).

The comments in the cell show the two ways to take a slice:
`max_files={"noise": 1000}` (the first N of the sorted file list) and
`fraction={"data": 0.5}` (a share of *each* pattern; use this for detector
data, whose `l3` is one pattern per run ordered by year).

*Produces:* HDF5 parts under `HDF_BASE/<sample>/`, each with a `.meta.json`
(and a `.badfiles.txt` if files were dropped).  `results` maps sample →
output path, or `None` for a failed sample.

*Check:* the final line lists no failed sample.  Finished parts are skipped
on a re-run, so a crash only costs the part in flight.

---

### Section 2 — Booking check

**Cell 11 (Markdown).**  The first thing to do after processing: if a tray
module failed silently, its table is never written.

**Cell 12 (code).**

1. Imports `dump_tables` and `find_hdf5` from `oscnext_l4.data`.
2. `H5_SAMPLE` = the first sample whose `kind` is `signal` (by role, so it
   does not depend on the name `nue`).
3. `H5 = find_hdf5(H5_SAMPLE, SAMPLES)`: the first existing HDF5 part of that
   sample (production files are preferred over `_smoke` files).  Prints how
   many parts exist and which one is inspected.
4. `TABLES = dump_tables(H5)`: prints every data table in that file (the
   `__I3Index__` tables are skipped), its row count and its columns.  Returns
   `{table: (nrows, [columns])}`.

*Produces:* `H5_SAMPLE`, `H5`, `TABLES` (`{}` if no file was found).

*Check:* every `L4_*` key you expect has a table with a plausible row count.

**Cell 13 (code).**  Only a commented example:
`dump_tables(H5, only=["iLineFit"])` prints just the tables whose name
contains one of the given strings.

---

### Section 3 — Feature registry and consistency

**Cell 14 (Markdown).**  Explains that every variable is one row in
`config/variables.json` carrying both sides: the `(HDF5 table, column)` that
training reads and the `(frame key, field)` that the tray reads.
`check_registry` asks whether the columns are really in this file;
`check_feature_map` asks whether the two sides of each row still name the
same quantity.

**Cell 15 (code).**

1. Imports `data as l4data` and, from `oscnext_l4.data`: `REGISTRY`, `ALTS`,
   `AUX`, `NOISE_FEATURES`, `MUON_FEATURES`, `WANTED`, `aux_for`,
   `scheme_of`, `check_registry`, `check_feature_map`.
2. `check_registry(TABLES, NOISE_FEATURES)` and
   `check_registry(TABLES, MUON_FEATURES)`: for each BDT input, looks for its
   column in the dumped file, trying the alternative spellings in `ALTS` too.
   Prints `found: n/N`, an `[i]` line when an alternative spelling was used,
   and an `[!]` line (with the table's real columns) for each one not found.
3. For the weight columns: `_scheme = scheme_of(SAMPLES[H5_SAMPLE])`, then
   `check_registry(TABLES, aux_for(_scheme), scheme=_scheme)`.  `aux_for`
   returns the auxiliary (weight) columns expected under that weight scheme.
   Passing `scheme` lets the check tell detector data (no weight columns by
   design) from a wrongly built empty list.
4. `check_feature_map()`: compares the two sides of every row of the variable
   table and checks that every BDT input has both.  Prints `no conflict` or a
   `[!] CONFLICT` list.  It reads only the config file.

*Check:* `found: 5/5` for the noise inputs, `found: 10/10` for the muon
inputs, the weight columns all found, and `no conflict`.

---

### Section 4 — HDF5 to numpy

**Cell 16 (Markdown).**  Tables are matched per event (through hdfwriter's
index), never by row order.  The weight divisor is the number of L3 files,
read from each part's `.meta.json`.

**Cell 17 (code): load the samples.**

1. Imports `load_sample`.
2. `RELOAD = False` (edit to force a full re-read).
3. Keeps the existing `data` dict if there is one; creates an empty one
   otherwise, or when `RELOAD` is `True`.
4. For each sample in `SAMPLES` not already in `data`:
   `load_sample(name, SAMPLES, WANTED)`.  That reads every HDF5 part of the
   sample, resolves every wanted column (the BDT inputs plus the weight
   columns this sample's scheme needs), aligns the tables per event, and
   concatenates the parts.  It refuses to load if one L3 file appears in two
   parts (double booking), and it records bookkeeping keys: `_part` (which
   part each event came from), `_n_files` (the L3 file count, the weight
   divisor), `_n_files_src`, `_n_hdf5`, `_files`, `_n_l3_per_file`.
5. Prints the event count per sample and lists samples that did not load.

*Produces:* `data`: sample name → dict of numpy arrays (`Run`, `Event`,
`SubEvent`, each variable by its registry name, and the bookkeeping keys).

*Check:* no `[!] ... have no meta.json` warning (the weights would be wrong),
and no sample in the "not loaded" list.  To reload one sample only, delete it
from `data` (`del data["nue"]`) and re-run the cell.

**Cell 18 (Markdown).**  A variable that is entirely NaN in a sample was
never booked.

**Cell 19 (code): NaN table.**  For every noise and muon input and every
loaded sample, prints the percentage of non-finite values.  A variable above
99.9% in any sample is marked `<-- ALWAYS MISSING`.

*Check:* no `ALWAYS MISSING` marker.  Some NaN is normal (for example
`accumulated_time` is not written for events with four or fewer DOMs).

---

### Section 5 — Weights

**Cell 20 (Markdown).**  Three separate notions: `w_phys` (the physical rate
in Hz, used for distributions and cut performance), `weight` (the training
weight, made in section 6) and unweighted counts.  The signal flux is a plain
power law, not a physical atmospheric flux.

**Cell 21 (code): livetime and weights.**

1. Imports `add_weights`, `set_data_livetime`, `livetime_from_headers`,
   `livetime_from_grl`, `scheme_of`.
2. For each loaded sample whose scheme is `data` (detector data only):
   - `livetime_from_headers(data[n]["_files"])`: an *estimate*: the span from
     the first to the last event of each run, from the booked event times,
     summed over runs.  Prints per-run spans and rates.
   - `livetime_from_grl(unique runs)`: the livetime from the Good Run Lists
     under `/data/exp/IceCube/<year>/filtered/...`, with dead time already
     subtracted.  Prints each run's livetime and any run not found.
   - If the Good Run Lists cover every run, prints the ratio GRL / estimate
     and calls `set_data_livetime(GRL total)`.  Otherwise it warns and uses
     the estimate.
3. `add_weights(data, SAMPLES)`: for every sample, picks the weight function
   by its scheme (`genie`, `noise`, `corsika`, `muongun`, `data`), stores
   `w_phys` (Hz) in the sample's dict, and prints the total rate, the
   largest single weight as a share of the total (`max/sum`), and, for
   `nue`, `numu`, `corsika` and `noise`, the ratio to the L3 rate in Table 13
   of the technical note (flagged if outside 0.1-10).

*Produces:* `data[name]["w_phys"]` for every sample.

*Check:* no `ORDER OF MAGNITUDE OFF`, and `max/sum` below 5%.  For detector
data the livetime only changes whether rates are in Hz: section 6 normalises
each class to unit sum, so training weights do not depend on it.

**Cell 22 (code): weight histograms.**  One panel per loaded sample:
histogram of `log10(w_phys)` over the finite, positive weights, titled with
`max/sum` in percent.

*Check:* above 5% means one event dominates the rate.

---

### Section 6 — Training sets (`.npz`) and the train/test split

**Cell 23 (Markdown).**  Describes the `.npz` content and the two splitting
rules: the signal split is drawn once and shared by both classifiers (the L4
cut is the AND of both, so they need a common held-out set), and CORSIKA is
split by shower, not by event (one shower is reused `OverSampling` times).

**Cell 24 (code): stack the signal and draw its split once.**

1. Imports `TRAIN_FRAC` (0.5), `stack`, `report_missing_inputs`,
   `split_by_shower`, `build_dataset` from `oscnext_l4.dataset`.
2. `ALL_FEATURES` = the sorted union of noise and muon inputs.
3. `SIGNAL` = the loaded samples whose `kind` is `signal` (so pass2's NuTau
   is included).
4. `SIG = stack(data, SIGNAL, ALL_FEATURES)`: concatenates those samples into
   one dict with every feature, `w_phys`, `Run`, and `_group` (sample and
   part index, used by the shower split).
5. `rng = np.random.default_rng(RNG_SEED)`, then
   `SIG_ISTRAIN = rng.random(n) < TRAIN_FRAC`: the one signal split, as
   the first draw of a fresh generator (the same draw `make_dataset.py`
   makes).  The noise background split in cell 26 is its second draw.

*Produces:* `ALL_FEATURES`, `SIGNAL`, `SIG`, `SIG_ISTRAIN`.

> **Watch out.**  Re-running the cell gives the same split, because it starts
> a fresh generator.  The split is the same in the noise and the muon stage
> only if `SIG` is stacked from the same samples, in the same order, with
> the same parts on disk.  The notebook does not check that;
> `scripts/make_dataset.py` (Part C) refuses a muon set whose signal does not
> match the noise set's, event for event.

**Cell 25 (code): load the noise model for the cut.**

1. Imports `NOISE_CUT` (0.70) and `load_noise_prob`.
2. If `MODEL_DIR/L4_noise_model.txt` exists:
   `noise_prob, _nfeat = load_noise_prob(path)`.  `noise_prob(d)` returns
   P(ν) for a stacked dict, reading the columns in the booster's own feature
   order.  Prints the model path, its features and the cut.
3. Otherwise prints a warning that the muon set will be built on the uncut
   population.

*Produces:* `noise_prob` (a function, or `None`).

**Cell 26 (code): build the training sets.**

*Noise half.*

1. `NOISE_BG` = loaded samples with `kind == "noise_bg"`.
2. If there are none (normal at `STAGE = "muon"`), sets `DS_NOISE = None`
   and prints `SKIPPED`.
3. Otherwise: `BG_NOISE = stack(data, NOISE_BG, ALL_FEATURES)`; reports NaN
   inputs for both classes with `report_missing_inputs`; draws
   `BG_NOISE_ISTRAIN` from `rng` (per event: vuvuzela has no
   oversampling); and calls `build_dataset("noise", SIG, BG_NOISE,
   NOISE_FEATURES, SIG_ISTRAIN, BG_NOISE_ISTRAIN, DS_BASE)`.

*Muon half.*

4. `MUON_BG` = loaded samples with `kind == "muon_bg"`.
5. If more than one is loaded (pass2 has `muongun` and `data`), they are
   **not** stacked: the detector-data one is kept (the production's choice),
   else the first.
6. If none: `DS_MUON = None`, prints `SKIPPED`.
7. Otherwise: `BG_MUON = stack(data, MUON_BG, ALL_FEATURES)`.
8. If `noise_prob` exists, applies the noise cut to both classes: keeps
   events with P(ν) ≥ 0.70, masks the signal *together with* its split flag
   (a surviving event keeps its side), and prints how many events of each
   class survive.  If no noise model exists, prints a loud warning and uses
   the uncut population.
9. Reports NaN inputs for both classes.
10. Draws the background split with `split_by_shower(BG_MUON["Run"],
    TRAIN_FRAC, np.random.default_rng(RNG_SEED + 1),
    groups=BG_MUON.get("_group"), allow_event_fallback=(weight != "corsika"))`.
    Events sharing (part, `Run`) move together.  It has its own generator so
    the muon set does not depend on whether the noise half ran.  With fewer
    than 20 groups it falls back to a per-event split with a warning, except
    for CORSIKA, where it raises instead.
11. Prints the number of distinct `Run` values and calls
    `build_dataset("muon", SIG_M, BG_MUON, MUON_FEATURES, SIG_M_ISTRAIN,
    BG_MUON_ISTRAIN, DS_BASE)`.

`build_dataset` sets non-finite or negative physical weights to 0, divides
each class by its sum (so both classes weigh the same), rescales all weights
so the largest is 1, and writes `DS_BASE/L4_<tag>_dataset.npz` with the
feature columns, `label` (1 signal, 0 background), `weight`, `w_phys`,
`istrain` and `features` (the name order).  It returns the path.

*Produces:* `DS_NOISE`, `DS_MUON` (paths or `None`), and the `.npz` files.

*Check:* the train/test counts printed by `build_dataset`; the noise-cut
survival percentages; no `NO NOISE CUT APPLIED` warning at the muon stage;
no `split_by_shower` fallback warning for a simulated background with
oversampling.

> **Watch out.**  The notebook calls `build_dataset` without `provenance`,
> so its `.npz` files, and the model sidecars trained from them, record no
> provenance.  `make_release.py` then writes "not recorded" for the signal
> and muon background and cannot check that the muon model was cut with the
> shipped noise model.  `scripts/make_dataset.py` builds the same sets with
> provenance.

---

### Section 7 — Training

**Cell 27 (Markdown).**  Training runs `scripts/train_L4_classifier.py`; the
hyperparameters are the technical note's Table 10 and are fixed in the
script.  Watch the `min_data_in_leaf` warning the script prints.

**Cell 28 (code): find the datasets.**  For each tag (`noise`, `muon`):

- if `DS_NOISE` / `DS_MUON` is already set (built in section 6), keeps it;
- else, if `DS_BASE/L4_<tag>_dataset.npz` exists, sets the variable to that
  path and prints its size;
- else sets it to `None` and prints `MISSING`.

This is the cell that lets you train after a kernel restart without redoing
sections 4-6.

**Cell 29 (code): train the noise model.**  Defines `run_train` (below).
Trains the noise model only if `DS_NOISE` is set **and** `STAGE` is `"noise"`
or `"all"`.  At `STAGE = "muon"` it prints that the noise model is not
retrained, because the muon set was just cut with it.

#### `run_train(tag, dataset, extra=())` (defined in cell 29)

- **What it does:**
  1. Builds the command `python -u train_L4_classifier.py --tag <tag>
     --dataset <dataset> --outdir MODEL_DIR` plus any `extra` flags, and
     prints it.
  2. Starts it with `subprocess.Popen`, merging stderr into stdout.
  3. Prints each output line as it arrives (so a long training is visibly
     alive).
  4. Waits, then prints the exit code and the elapsed time.
- **Inputs → outputs:** tag (`"noise"`/`"muon"`), dataset path, extra flags
  (for example `["--min-data-in-leaf", "100"]`) → `True` if the exit code was 0.
- **Side effects:** the script writes into `MODEL_DIR`:
  `L4_<tag>_model.txt` (LightGBM text model), `L4_<tag>_model.json`
  (sidecar) and the plots `<tag>_overtrain.png`, `<tag>_dist.png`,
  `<tag>_cuts.png`.

**Cell 30 (code): train the muon model.**  `run_train("muon", DS_MUON)` if
`DS_MUON` is set; otherwise prints that it was skipped.

*Check (both):* `exit 0`, and read the script's own warnings.

---

### Section 8 — Validation

**Cell 31 (Markdown).**  Overtraining is judged by the gap between train and
test efficiency at the same rejection, not by a KS test (which flagged
irrelevant differences on this data).

**Cell 32 (code): summary per model.**  For each tag whose
`MODEL_DIR/L4_<tag>_model.json` exists, loads it into `META[tag]` and prints:
number of trees, efficiency at the target rejection, the target rejection,
background events left there, and the train-test gap.  Marks
`<-- MEMORISING` when the gap is above 0.05, and warns when fewer than 10
background events define the target point.

*Produces:* `META` (tag → sidecar dict).

*Check:* gap below 0.05; at least 10 background events at the target point.

**Cell 33 (code): plots.**  Displays `<tag>_cuts.png`, `<tag>_overtrain.png`
and `<tag>_dist.png` from `MODEL_DIR` for each trained model, or says which
file is missing.

**Cell 34 (code): feature importance.**  For each model, prints the
`importance_gain` from the sidecar, sorted, with each variable's share of the
total gain.

---

### Section 9 — Choosing the cut

**Cell 35 (Markdown).**  The model output is P(signal).  The thresholds
`noise ≥ 0.70` and `muon ≥ 0.65` are the production's, not an optimum for our
models.  Lists the Table 13 targets (noise: 36.6 mHz down to < 0.3 mHz,
keeping ~96% of neutrinos; muon: 94% of muons rejected, 87% of neutrinos
kept).

**Cell 36 (code): the table per model.**  For each model in `META`, prints
the sidecar's `metrics.table`: for each rejection level, the efficiency, the
cut value and the background events kept out of the total.  Rows with fewer
than 10 background events are marked `<-- not a measurement`.  Collects
`CUTS[tag] = default_cut` (0.70 / 0.65) and prints it.

*Produces:* `CUTS`.  To use your own cut, pick it from the table and set
`CUTS` by hand.

---

### Section 10 — Applying the model to frames

**Cell 37 (Markdown).**  Shows how to add both classifiers to an IceTray
tray:

```python
from oscnext_l4.classifier import add_L4_classifiers
add_L4_classifiers(tray, "L4_clf", model_dir=MODEL_DIR,
                   noise_cut=0.70, muon_cut=0.65)
```

That adds two modules writing `L4_NoiseClassifier_ProbNu` and
`L4_MuonClassifier_Data_ProbNu`, and a third writing the combined cut
`L4_oscNext_bool` (an `I3Bool`, noise AND muon).  In practice
`process_L4.py --apply-cut --model-dir <dir>` does this (Part B).

**Cell 38 (code).**  `check_feature_map()` again (same as in section 3).

**Cell 39 (Markdown): checklist.**  A tick list per stage (processing,
weights, training) and the items still unverified: run
`scripts/check_application.py` on an `--apply-cut` output to check the frame
reads on a real frame; `fill_ratio`'s `SphericalRadiusMean=1.6` was never
re-tuned for oscNext; the muon background differs from the production's;
the noise MC statistics are thin.

---

## verification/

**What it is for.**  The pass2 cross-check.  Several L4 variables are
pure-Python rewrites of IceTray modules this environment does not have.  The
official pass2 production is an answer key for them:

- the pass2 **L3** files are the input our code was built to read;
- the matching pass2 **L4** files already contain every variable, computed by
  the original code.

So the check runs our ordinary `process_L4.py` over a pass2 L3 file, and
compares its output, event by event and column by column, with the values
stored in the matching L4 file.

```
pass2 L3  --(process_L4.py, our segment)-->  ours.hdf5   ┐
                                                         ├-> match events, compare columns
pass2 L4  --(the production's own HDF5, or `book`)--> pass2.hdf5 ┘
```

**Where it sits.**  Outside the pipeline.  It is a *consumer* of
`oscnext_l4/`, like `scripts/` and `notebooks/`; nothing in the pipeline
imports it.  The one deliberate coupling: `pass2.py` uses three private
helpers of `oscnext_l4.data` (`_table_nodes`, `_index_node`, `_ids`), so that
it reads HDF5 files exactly the way training does.

### Key ideas

**How events are matched.**  An event is identified by the triple
`(Run, Event, SubEvent)` from `I3EventHeader`.  Within one file:

1. Each compared column is read from its table.  If the table has one row
   per event, in the same order as `I3EventHeader`, it is used as is.
2. Otherwise (a key missing in some frames), the row is found through
   hdfwriter's `/__I3Index__/<key>` table: one row per frame, with `exists`
   (is the key in this frame) and `start` (its row in the data table).
   Frames without the key get NaN.
3. The two files are then paired on the triple: each event of ours is looked
   up among theirs.

**Why one L3 file per HDF5.**  The triple is unique only within one L3 file.
In simulation `Run` is the dataset number and `Event` restarts at zero in
every file, so two L3 files in one HDF5 produce repeated triples, and
matching them would pair unrelated events.  `match()` therefore refuses
(raises) when it sees a repeated triple.  That is why the cross-check books
each L3 file into its own HDF5 (no `--chunk-files`), and why it cannot use
the notebook's runner, which names parts by job and part index.

**Control rows versus rewritten rows.**  The compared quantities fall into
three groups:

| group | rows | what a disagreement means |
|---|---|---|
| **rewritten** | `micro_count`, `FullTimeLengthRatio`, `VICH_nch`, `VICH_npulses`, `VICH_qtot`, `accumulated_time`, `first_hlc_rho` | our rewrite differs from the original; this is what the exercise tests |
| **derived** | `fill_ratio` | the original module, fed the vertex we compute; a difference is inherited from `first_hlc_rho` |
| **control** | `iLineFit_speed`, `cog_z`, `z_sigma`, `z_travel`, `n_hit_doms` (original modules on both sides), `NchCleaned`, `ICVetoHits` (read straight from L3) | the two sides did not see the same input (pulse series, geometry, event set), and nothing else in the table can be trusted |

Read the control rows first.

**Verdicts.**  For each row, only events where both sides are finite are
compared (`n` is their count).  An event "agrees" when:

- for integer-valued rows (`micro_count`, `VICH_nch`, `VICH_npulses`,
  `n_hit_doms`, `NchCleaned`, `ICVetoHits`): the values are exactly equal;
- for float rows: `|ours - pass2| ≤ 1e-9 + 1e-6·|pass2|`.

The verdict then comes from the *fraction* of agreeing events:

| verdict | condition |
|---|---|
| `identical` | the largest absolute difference is exactly 0 |
| `agrees` | at least 99.9% of events agree |
| `mostly` | at least 99% agree; the report says how many are left |
| `DIFFERS` | below 99% |
| `no overlap` | no event has a finite value on both sides |

The current result is recorded in `verification/README.md`: 14 of 15 rows
bitwise identical over 56,301 events; `accumulated_time` agrees in 99.84%.

## verification/\_\_init\_\_.py

**What it is for.**  Makes `verification` an importable package (so the
notebook and `compare_pass2.py` can write `from verification import pass2`).
It contains only a docstring: what the cross-check is, why it sits outside
`oscnext_l4/`, and the rule "delete this arm last" (it is the only place a
change that silently breaks a variable would be caught).  No code.

## verification/pass2.py

**What it is for.**  The comparison machinery: which keys to book from the
pass2 files, which columns to compare, how to read and match two HDF5 files,
the verdicts, and the text report.  Used by `compare_pass2.py` and by
`pass2_verification.ipynb`.  Imports only `os`, `json` and `numpy` at module
level (`tables` and `oscnext_l4.data` are imported inside `read_pairs`), so
it loads without IceTray.

**Module-level constants.**

| name | value / meaning |
|---|---|
| `_CONFIG_PATH` | `OSCNEXT_L4_CONFIG` or `<repo>/config/productions.json` |
| `_PASS2` | the `pass2` block of that file, read at import time |
| `PASS2_CLEANED_PULSES` | pass2's cleaned series, from the config (`SRTTWOfflinePulsesDC`) |
| `PASS2_SAMPLES_ALL` | every pass2 sample spec from the config |
| `PASS2_UNCLEANED_PULSES` | `"SplitInIcePulses"` (same as pass3) |
| `OURS_HITSTAT`, `OURS_HITMULT` | the hit-statistics key names our pipeline writes even on pass2 (`SRTTWSplitInIcePulsesDCHitStatistics` / `...HitMultiplicity`) |
| `L3_VARS_KEY` | `"IC2018_LE_L3_Vars"` |
| `DEPENDS_ON` | `{"fill_ratio": "first_hlc_rho (the vertex we pass it)"}`: rows that are the original module fed our input |
| `INTEGER_VARS` | rows held to exact equality |
| `EXPECTED_DEVIATION` | rows expected to differ, with the explanation printed when they do.  Only `micro_count` is listed, and its text says it is *no longer* expected to differ with the default chain (only with `--micro-count-cleaned`). |
| `FLOAT_RTOL`, `FLOAT_ATOL` | `1e-6`, `1e-9`: the float agreement tolerance |
| `AGREE_FRACTION`, `MOSTLY_FRACTION` | `0.999`, `0.99`: the verdict thresholds |
| `_SAMPLES` | `("NuE", "NuMu", "NuTau", "MuonGun", "Noise")`: the cross-check's sample names (capitalised; the config uses lower case).  Detector data is not among them. |
| `PASS2_L3` | sample → list of L3 glob patterns (from the config) |
| `PASS2_FLAGS` | sample → `process_L4.py` flags (from the config, e.g. `["--mc", "--genie"]`) |

### `pass2_hitstat_keys(cleaned_pulses=PASS2_CLEANED_PULSES)`

- **What it does:** builds pass2's hit-statistics key names from the cleaned
  series name.
- **Inputs → outputs:** series name → `(name + "HitStatistics",
  name + "HitMultiplicity")`.

### `pass2_book_keys(cleaned_pulses=PASS2_CLEANED_PULSES)`

- **What it does:** lists the frame keys to book out of a real pass2 L4 file:
  `I3EventHeader`, the two L3 maps, every L4 variable key (`L4_micro_count`,
  `L4_fill_ratio`, the three VICH keys, `L4_accumulated_time`,
  `L4_separation_in_cogs`, `L4_first_hlc(_rho)`, `L4_iLineFit(Params)`,
  `L4_ToI(Params)`), pass2's hit statistics, pass2's own verdicts
  (`L4_oscNext_bool`, `L4_NoiseStraightCuts_Bool`, the three classifier
  scores) and `L4_QR_Box`.
- **Inputs → outputs:** cleaned series name → list of key names.
- Nothing is computed on this side.  Keys absent from the file are simply
  not booked; the report later says "no table".

### `compare_table(cleaned_pulses=PASS2_CLEANED_PULSES)`

- **What it does:** defines what is compared.  Returns a list of 15 rows
  `(name, ours=(table, column), pass2=(table, column), rewritten)`.
- The two sides differ in only two ways: `FullTimeLengthRatio` is read from
  `L4_FullTimeLengthRatio.value` on our side and from
  `IC2018_LE_L3_Vars.FullTimeLengthRatio` on pass2's (pass2 stored it at L3;
  we divide at L4); and the hit statistics carry our pass3-style key name on
  our side and pass2's name on theirs.
- `rewritten` is `True` for the seven rewritten rows, `False` for the rest.
- **Watch out:** the `ours` pairs are written out by hand here.  They must
  match the HDF5 side of `config/variables.json`; nothing checks that
  automatically.  (They match today.)

### `read_pairs(h5path, pairs)`

- **What it does:** reads named columns from one HDF5 file, aligned to its
  `I3EventHeader` rows.
  1. Opens the file with pytables and lists the real data tables with
     `data._table_nodes` (index tables excluded).  Raises if there is no
     `I3EventHeader` table.
  2. Reads `Run`, `Event`, `SubEvent` from `I3EventHeader` with `data._ids`.
  3. For each `(name, table, column)`: if the table or column is missing,
     records it and fills NaN.  If the table's ids equal the header's row
     for row, uses the column directly.  Else uses the `/__I3Index__/<table>`
     table (`data._index_node`): rows where `exists` is true take the value
     at `start`; others get NaN.  If there is no usable index, records
     `"unaligned, no index"` and fills NaN.
- **Inputs → outputs:** file path, list of `(name, table, column)` →
  `(out, missing)`: `out` is `{name: float64 array}` plus `Run`, `Event`,
  `SubEvent`; `missing` is a list of `(name, table, column, reason)`.

### `_triples(d)`

- **What it does:** stacks `Run`, `Event`, `SubEvent` of a dict into an
  `(n, 3)` array.

### `match(ours, theirs, strict=True)`

- **What it does:** pairs the events present on both sides.
  1. Checks each side for repeated `(Run, Event, SubEvent)` triples.  If one
     repeats: with `strict=True` raises `RuntimeError` ("the file holds more
     than one L3 file"); with `strict=False` prints a warning and continues.
  2. Builds a lookup from their triples to row numbers, and walks our rows,
     keeping those whose triple exists on their side.
- **Inputs → outputs:** two dicts from `read_pairs` → `(idx_ours,
  idx_theirs)`, two integer arrays of equal length.
- **Watch out:** with `strict=False` and repeated triples, the last
  occurrence wins, so the match is not reliable.

### `load_comparison(ours_h5, pass2_h5, cleaned_pulses=PASS2_CLEANED_PULSES, strict=True)`

- **What it does:** reads both files with `read_pairs` (using the two sides
  of `compare_table`), matches them with `match`, and returns the matched
  values.
- **Inputs → outputs:** two paths → `(data, missing, counts)`:
  - `data`: `Run`, `Event`, `SubEvent` of the matched events, plus
    `data["ours"][name]` and `data["pass2"][name]` arrays in matched order;
  - `missing`: the columns not found, each tagged `"ours"` or `"pass2"`;
  - `counts`: events on each side and matched.

### `_verdict(name, ours, theirs)`

- **What it does:** judges one row (see "Verdicts" above).
  1. Keeps events finite on both sides; returns `"no overlap"` if none.
  2. Computes the absolute difference, the largest absolute difference and
     the largest relative difference (relative to pass2; where pass2 is
     exactly 0 and ours is not, the relative difference is infinite and is
     left out of the maximum).
  3. Computes the agreeing fraction (exact for integer rows, tolerance for
     floats).
  4. Returns `identical` / `agrees` / `mostly` / `DIFFERS`.
- **Inputs → outputs:** row name, two arrays →
  `(status, n, agree_fraction, max_abs_diff, max_rel_diff)`.
- **Watch out:** events where only one side is NaN are left out of `n`, not
  counted as disagreements.  Compare `n` with the matched count to see them.
  (The docstring still mentions `np.hypot` for `first_hlc_rho`; the code no
  longer uses it and the row is now bitwise identical.)

### `report(ours_h5, pass2_h5, cleaned_pulses=PASS2_CLEANED_PULSES, n_worst=5, strict=True, stream=None)`

- **What it does:** prints the full comparison for one pair of files.
  1. `load_comparison`; prints the file names, event counts and the matched
     count.  If nothing matched, says so and returns `[]`.
  2. Lists columns not found on either side.
  3. Prints one line per row: name, role (`rewritten`, `derived`,
     `control`), `n`, agreement in percent, max |diff|, max relative diff,
     verdict.
  4. For each `mostly` row, says how many events are left.
  5. For each `DIFFERS` or `mostly` row, prints up to `n_worst` events with
     the largest difference (only events that really differ).
  6. Prints the explanation for `DIFFERS` rows listed in
     `EXPECTED_DEVIATION` or `DEPENDS_ON`.
  7. If a control row `DIFFERS`, prints `CONTROL ROWS DISAGREE` with the
     advice to fix that first.
  8. Prints "Rewritten variables reproducing pass2: k/7" (`identical`,
     `agrees` and `mostly` count as reproducing) and lists rows with no
     overlap.
- **Inputs → outputs:** two paths, options → list of dicts, one per row, with
  `name`, `rewritten`, `status`, `n`, `agree`, `max_abs`, `max_rel`.
- **Side effects:** prints to `stream` (default `sys.stdout`).

### `_l3_patterns(spec)`

- **What it does:** returns a sample spec's `l3` as a list (a pass2 sample
  can span two datasets, e.g. NuE is 121122 and 121291).

### `l4_path(l3_path)`

- **What it does:** derives the pass2 L4 path from an L3 path by replacing
  `level3` with `level4` (in the directory and in the file name).
- **Inputs → outputs:** L3 path → L4 path.  Raises `ValueError` if the path
  has no `level3`, so a file is never compared with itself.

### `pair_files(l3_files)`

- **What it does:** for each L3 file, checks whether its L4 partner exists.
- **Inputs → outputs:** list of L3 paths → `(pairs, orphans)`: `pairs` is
  `[(l3, l4)]` for existing L4 files, `orphans` is `[(l3, expected_l4)]` for
  missing ones.

## verification/compare_pass2.py

**What it is for.**  The command-line front end to `pass2.py`, with four
subcommands.  Run from the repository root (it adds the root to `sys.path`
itself).

| subcommand | needs IceTray | what it does |
|---|---|---|
| `inspect` | yes | prints what is really in a pass2 frame; run it first |
| `plan` | no | prints the `process_L4.py`, `book` and `report` command lines for a sample |
| `book` | yes | books the L4 values already stored in pass2 L4 `.i3` files into an HDF5 |
| `report` | no | matches two HDF5 files and prints the comparison |

### Flags

`inspect`:

| flag | default | meaning |
|---|---|---|
| `--input FILE [FILE ...]` | required | `.i3` files; globs are expanded |
| `--n N` | 1 | how many P frames to show per file |
| `--sub-event-stream S` | `InIceSplit` | only show P frames of this stream (empty string: all) |
| `--all-keys` | off | also list every key in the frame |

`plan`:

| flag | default | meaning |
|---|---|---|
| `--sample NAME` | required | one of `MuonGun`, `Noise`, `NuE`, `NuMu`, `NuTau` |
| `--gcd FILE` | required | the GCD written into every printed command |
| `--outdir DIR` | `L4_output/pass2_check` | where the printed commands put their HDF5 files |
| `--n-files N` | 5 | how many L3 files (first N of the sorted list) |
| `--cleaned-pulses NAME` | `SRTTWOfflinePulsesDC` (from the config) | the pass2 cleaned series |

`book`:

| flag | default | meaning |
|---|---|---|
| `--gcd FILE` | required | GCD file, read before the inputs |
| `--input FILE [FILE ...]` | required | pass2 L4 `.i3` files; globs are expanded |
| `--output-hdf5 FILE` | required | the HDF5 to write |
| `--sub-event-stream S` | `InIceSplit` | P frames of other streams are dropped |
| `--n N` | 0 | stop after N frames (0 = all) |
| `--overwrite` | off | replace an existing output file |
| `--cleaned-pulses NAME` | from the config | decides pass2's hit-statistics key names |

`report`:

| flag | default | meaning |
|---|---|---|
| `--ours FILE` | required | our HDF5 (one L3 file) |
| `--pass2 FILE` | required | the answer-key HDF5 |
| `--n-worst N` | 5 | worst events printed per disagreeing row |
| `--allow-duplicates` | off | match even when triples repeat (`strict=False`); the match is then not reliable |
| `--fail-on-diff` | off | exit with code 1 when a rewritten row `DIFFERS` (for scripted use) |
| `--cleaned-pulses NAME` | from the config | pass2's series name, for the hit-statistics keys |

Example:

```bash
python verification/compare_pass2.py inspect --input <pass2 L4 file>
python verification/compare_pass2.py plan --sample NuE --gcd <GCD> --n-files 2
python verification/compare_pass2.py book --gcd <GCD> --input <L4 file> \
    --output-hdf5 L4_output/pass2_check/pass2_X.hdf5
python verification/compare_pass2.py report \
    --ours L4_output/pass2_check/ours_X.hdf5 --pass2 L4_output/pass2_check/pass2_X.hdf5
```

### `expand(patterns)`

- **What it does:** expands shell-style globs (only strings containing `*`,
  `?` or `[`), keeps plain paths as they are, prints `[!] no match` for a glob
  that matches nothing, and removes duplicates while keeping the order.
- **Inputs → outputs:** list of strings → list of paths.

### `_icetray()`

- **What it does:** imports IceTray only when a subcommand needs it:
  `I3Tray`, `icetray`, `dataio`, `dataclasses`, and calls
  `tray_io.load_deserialization_libs()` so frame objects of the other
  projects can be read.
- **Inputs → outputs:** none → `(icetray, dataio, dataclasses, I3Tray)`.

### `_typename(frame, key)`

- **What it does:** returns the Python type name of `frame[key]`, or
  `<unreadable: ...>` when reading the object raises (for example
  `L4_Dunkman_..._Variables`, whose class belongs to a project this build
  lacks).  This keeps `inspect` going past unreadable keys.

### `cmd_inspect(args)`

- **What it does:** for each input file:
  1. Opens it with `dataio.I3File` (a file that cannot be opened is reported
     and skipped).
  2. Reads frames until `--n` matching P frames were shown, skipping other
     sub-event streams.
  3. For each shown frame, prints the pulse series keys (names containing
     `Pulse`) with their types, the hit-statistics keys, and the `L4_*` keys
     with their types; with `--all-keys`, every key.
  4. Says so if no P frame matched.
- **Why run it first:** it shows what the pulse series and the L4 keys are
  really called in these files.

### `cmd_plan(args)`

- **What it does:** prints, for the first `--n-files` L3 files of a sample
  that have an L4 partner, three commands each: `process_L4.py` (our side,
  one L3 file per output `ours_<tag>.hdf5`, with the sample's flags),
  `compare_pass2.py book` (the answer key `pass2_<tag>.hdf5`) and
  `compare_pass2.py report`.  Reports orphan L3 files.  Exits with an error
  for an unknown sample.
- **Watch out:** every printed command uses the one `--gcd` you pass.  The
  notebook instead finds the GCD beside each file.

### `cmd_book(args)`

- **What it does:** books the values already in pass2 L4 files; computes
  nothing.
  1. Imports IceTray, `tray_io.add_booker`, and `icecube.common_variables`
     (this registers the HDF5 converters for the hit-statistics objects;
     without it the writer dies mid-run).
  2. Expands the inputs; exits if none.  Gets the key list from
     `pass2_book_keys`.
  3. Exits if the output exists and `--overwrite` was not given; creates the
     output directory.
  4. Builds a tray: `I3Reader` over `[GCD] + files`; a filter keeping P frames
     whose `I3EventHeader.sub_event_stream` equals `--sub-event-stream`; a
     counter (inner function `_count`, which adds one per P frame and keeps
     the frame); and the booker (`add_booker`) for those keys and that stream.
  5. Runs it (`--n` frames, or all) and prints the number of events booked.
- **Side effects:** writes the HDF5 (with its `__I3Index__` tables).

### `cmd_report(args)`

- **What it does:** calls `pass2.report(...)` with the flags.  With
  `--fail-on-diff`, exits with code 1 if any rewritten row `DIFFERS`.

### `main()`

- **What it does:** builds the argparse parser with the four subcommands
  (flags above).  An inner helper `common(sp)` adds `--cleaned-pulses` to
  `plan`, `book` and `report`.  Parses the command line and calls the chosen
  `cmd_*` function; with no subcommand, prints the help and exits with 1.

## verification/pass2_verification.ipynb

**What it is for.**  The cross-check as it is actually run: many files per
sample, several trays in parallel, and the comparison **pooled over all
files** (each variable is judged on all matched events at once).  It does
not use `book`: the pass2 production published an HDF5 beside every L4
`.i3.zst` (same file name stem), and that file is read in place as the answer
key.  Nothing writes to it.

### Section 0 — Setup

**Cell 2 (code).**  Enables autoreload; finds the repository root
(`OSCNEXT_L4_ROOT`, else the first parent directory holding `oscnext_l4/`);
**changes the working directory to the root** (`os.chdir`); puts the root on
`sys.path`; imports `verification.pass2 as P`; prints the root, the Python
executable and the sample names.

### Section 1 — Configuration

**Cell 4 (code).**  The settings:

| setting | default | meaning |
|---|---|---|
| `SAMPLES` | `["NuE", "NuMu", "NuTau", "MuonGun", "Noise"]` | which samples |
| `N_FILES` | 2 | L3 files per sample (0 = all).  One file is already ~8000 events. |
| `OUTDIR` | `L4_output/pass2_check` | where our HDF5 files go (relative to the repository root) |
| `EXTRA_FLAGS` | `[]` | extra `process_L4.py` flags.  Empty reproduces pass2.  `["--micro-count-cleaned"]` or `["--accumulated-time-note"]` test the technical note's readings instead. |

Creates `OUTDIR`.

### Section 2 — The file pairs

**Cell 6 (code).**  Defines two helpers and builds the job list.

#### `production_hdf5(l4_path)`

- **What it does:** replaces the `.i3.zst` / `.i3.bz2` / `.i3.gz` / `.i3`
  ending of an L4 path with `.hdf5`: the production's published HDF5.
- **Inputs → outputs:** path → path, or `None` if the ending is not
  recognised.

#### `find_gcd(*paths)`

- **What it does:** for each given file in turn, looks in its directory for
  `GeoCalibDetectorStatus*.i3*`.  Exactly one match is returned (and cached
  per directory); more than one raises `RuntimeError`; none moves on to the
  next path.  Raises if no path has one.
- **Why:** each sample uses the GCD that sits with its own files, never one
  passed by hand; every geometry-derived variable depends on it.

Then, for each sample: globs its L3 patterns (`P.PASS2_L3`), takes the first
`N_FILES`, pairs them with `P.pair_files`, and for each pair whose production
HDF5 exists appends a job `(sample, tag, l3, production_hdf5, gcd,
ours_hdf5)` to `JOBS`.  The GCD is looked for beside the L4 file first, then
the L3 file.  `ours_hdf5` is `OUTDIR/ours_<L3 file stem>.hdf5`, so each
output is named after its L3 file.  Prints per sample: L3 files, paired,
with a production HDF5, orphans; then the GCD used per sample and the total
number of jobs.

*Check:* the "with a production hdf5" count is non-zero for each sample, and
each sample shows exactly one GCD.

### Section 3 — Produce our side

**Cell 8 (code).**  Defines `produce` and `_one`, then runs every job.

#### `produce(job, extra_flags=EXTRA_FLAGS, quiet=True)`

- **What it does:** one `process_L4.py` run for one L3 file.
  1. Returns `"skipped"` if our output already exists and `REBUILD` is
     `False`.
  2. Removes leftovers from a killed or earlier run: the output, its
     `.badfiles.txt` and `.meta.json`, and the `_incomplete_` file.
  3. Runs `<sys.executable> scripts/process_L4.py --gcd <gcd> --input <l3>
     --cleaned-pulses SRTTWOfflinePulsesDC --output-hdf5 <ours> --scan off`
     plus the sample's flags and `extra_flags`.  No `--chunk-files`: one L3
     file per HDF5.
  4. Returns `"produced"`, or `"FAILED"` (printing the last 2000 characters
     of stderr when `quiet`).
- **Watch out:** the kernel's own python runs the tray, so the kernel must be
  the IceTray one (start Jupyter from inside the environment).

#### `_one(item)`

- **What it does:** runs `produce` for one `(index, job)` pair, then, under a
  lock, counts the result and prints one progress line (count, elapsed
  seconds, sample, tag, status).

`REBUILD = False` (set `True` to redo existing outputs) and `WORKERS = 4`
(parallel trays; 1 = sequential).  With `WORKERS > 1` the jobs run in a
`ThreadPoolExecutor`: threads are enough because each only waits on its
subprocess.  Finally prints the status counts and the total time.

### Section 4 — Compare, pooled over every file

**Cell 10 (code).**

1. `TABLE = P.compare_table(...)`, `NAMES` (the 15 row names), `REWRITTEN`
   (name → rewritten flag).
2. For every job whose output exists: `P.load_comparison(ours, ref, ...)`.
   An exception is recorded in `problems` and the file is skipped.  Each
   missing column is recorded in `problems` too.  The matched values of each
   row are appended to `pool[name]["ours"]`, `["pass2"]`, and the sample
   name to `["sample"]`.
3. Concatenates the pooled arrays.
4. Prints a per-file table (events ours / pass2 / matched), the total
   matched, and up to 20 problems.

*Produces:* `pool`, `per_file`, `problems`, `n_matched`.

*Check:* matched equals both sides' counts for each file; no problems.

### Section 5 — The summary table

**Cell 12 (code).**  For each row, `P._verdict(name, ours, pass2)` on the
pooled arrays (or `"no data"` if the row is empty).  Prints variable, kind
(`rewritten`, `derived`, `control`), verdict, `n`, agreement fraction and
max |diff|.  Then prints either `Every row agrees.` or the rows that are not
`identical`/`agrees`, with the first sentence of their `EXPECTED_DEVIATION`
or `DEPENDS_ON` note.

*Produces:* `SUMMARY`: list of `(name, rewritten, status, n, agree, max_abs)`.

### Section 6 — Agreement per variable

**Cell 14 (code).**  A horizontal bar chart: one bar per row with data,
sorted by agreement (worst at the bottom), coloured by verdict (a palette
checked for colour-vision deficiency), with the verdict and fraction written
beside each bar.  Rows that are not rewritten are labelled `(control)`
(this includes `fill_ratio`, which the table calls `derived`).

### Section 7 — Where the differences are

**Cell 16 (code).**  For every row that is not `identical`: a histogram of
`ours - pass2` (log y axis), titled with how many events differ.  A spike at
zero with a thin tail means a handful of odd events; a shifted or broad
distribution means a different definition.  Prints a message instead if
every row is identical.

### Section 8 — Drill into one variable

**Cell 18 (code).**  Set `VAR` (default `"accumulated_time"`).  Prints how
many events differ, the count per sample, how many of them have exactly 0
stored by pass2, and the 15 events with the largest absolute difference
(sample, ours, pass2, diff).

## verification/README.md

A step-by-step record, readable on its own, of how the rewritten variables
were checked against pass2: why a check was needed (§1), the answer-key idea
(§2), establishing that both sides see the same input (pulse series, L3 cut,
GCD, event set; §3), the bugs the first run exposed (§4), the first result
and the fitting that followed (§5-6, kept as a historical snapshot that later
sections correct), how `accumulated_time` (§7) and `VICH` (§8) were solved
from the original source, two bugs found by reading `FirstHLC.cxx` and
`CalculateVariables.cxx` (§9), the final table (§10: 14 of 15 rows bitwise
identical over 56,301 events), what is still open (§11) and how to recover
the deleted fitting tools from git (§12, commit `d9392e96c2`).

---

## scripts/make_release.py

**What it is for.**  Assembles a self-contained, runnable directory to hand
to a collaborator.  The release serves three uses: turning L3 `.i3` into L4
`.i3` (`process_L4.py`), training (`make_dataset.py`,
`train_L4_classifier.py`), and the trained models.  `check_application.py`
ships as the first check to run.

**Where it sits.**  A development tool; it does not ship itself.  It runs
without IceTray: the file list is found by reading the source with Python's
`ast` module (parsing, not importing).  Only the model check imports
`oscnext_l4.varmap` (standard library only).

### How the release is assembled

1. **What goes in is computed, not listed.**  Starting from the four entry
   scripts (`process_L4.py`, `make_dataset.py`, `train_L4_classifier.py`,
   `check_application.py`) plus `oscnext_l4/__init__.py`:
   - **modules**: every import of `oscnext_l4` (absolute or relative, at any
     depth, including imports inside functions) is followed, to a fixpoint;
   - **scripts**: any `scripts/<name>.py` that a shipped file names in a
     string constant (docstrings excluded), or that the release templates
     name in their text, is added, and its imports followed too;
   - **config**: a file in `config/` is added when a shipped file or a
     shipped config file names it in a string (`varmap` opens
     `variables.json`; `productions.json` points at `config/README.md`).
2. **Checks.**  It fails, and writes nothing, when: an internal import names
   a module that does not exist or a name the module does not define; a
   relative import leaves the package or appears in a script; a dynamic
   `import_module("oscnext_l4...")` is found (it cannot be followed); a
   model check fails (below); or a `{{...}}` field is left unfilled.
3. **Models** (from `--models DIR`): for each of `noise` and `muon`,
   `L4_<tag>_model.txt` and `.json` must come as a pair; the sidecar's
   `features` must equal the booster's own `feature_names=` line; every
   feature must be in `FEATURE_MAP` (the frame side of the variable table);
   the sidecar's `model_sha256`, if recorded, must match the `.txt`, and
   its `n_trees`, if recorded, must equal the number of trees in the `.txt`
   (the tree count is only checked when the sha256 did not already fail);
   and when both models are present, the muon model's recorded noise
   model (by sha256) must be the shipped noise model.  Both models are
   required unless `--partial-models`.  A model trained on a non-standard
   input list is shipped with a warning in the README.
4. **Build in a temporary directory** `<out>.tmp-build`: copy the closure,
   copy the models into `models/`, write `models/README.md` if any model is
   missing, fill and write `README.md`.
5. **Verify the copy**: re-run the closure *inside* the release (every import
   must resolve there, and the file set must equal the source's), compile
   every `.py`, and refuse any `__pycache__` or `.pyc`.
6. Write the marker file `.oscnext_l4_release` (build date, commit, dirty
   flag, models directory, file list), then replace `--out` with the
   temporary directory.  An existing non-empty `--out` is replaced only if it
   carries that marker.

**What ships** (today's closure, from `--list`): `config/README.md`,
`config/productions.json`, `config/variables.json`; `oscnext_l4/__init__.py`,
`classifier.py`, `data.py`, `dataset.py`, `frame_objects.py`, `l3vars.py`,
`rewritten.py`, `tray_io.py`, `variables.py`, `varmap.py`;
`scripts/check_application.py`, `diagnose_env.py`, `make_dataset.py`,
`process_L4.py`, `scan_files.py`, `train_L4_classifier.py`; plus `models/`,
the filled `README.md` and the marker file.

**What does not ship:** the notebook, `oscnext_l4/runner.py`, `verification/`,
`presentation/`, `reference/`, `docs/`, `icetray-oscNext/`, `CLAUDE.md`,
`setup_env.sh`, and the other scripts (`make_release.py`, `plot_inputs.py`,
`check_leakage.py`, `inspect_production_table.py`, `collect_meta.sh`).
Models never enter git; they come only from `--models`.

### The templates and their `{{...}}` fields

The fields are filled by plain text replacement (`_fill`).  Afterwards any
remaining `{{UPPER_CASE}}`, `<fill ...>` or `<model-dependent ...>` marker is
an error.  Template comments (`<!-- Template: ... -->`) are removed.

| template | field | filled with | from |
|---|---|---|---|
| `release/physics_caveats.md` | `{{AT_DEFAULT_CUT}}` | each model's efficiency and rejection at its default cut, on its own test set | `text_at_default_cut` (sidecar `metrics.at_default_cut`) |
| | `{{SIGNAL}}` | which signal sets and weighting the models were trained with | `text_signal` (sidecar `provenance`) |
| | `{{MUON_BACKGROUND}}` | which background the muon model used, and whether (and with which noise model) the noise cut was applied | `text_muon_background` (sidecar `provenance`) |
| `release/README.md` | `{{MODELS}}` | a Markdown table describing the shipped models, plus warnings | `text_models` |
| | `{{PHYSICS_CAVEATS}}` | the filled `physics_caveats.md` | |
| | `{{CONTENTS}}` | the file tree, the third-party imports, the build date and commit | `_contents` |
| `release/models_README.md` | `{{STATUS}}` | which model files are missing and which were shipped | `build` |

What each template says:

- **`release/README.md`**: the release's front page.  Environment setup on
  cvmfs (IceTray v1.17.0); how to run `process_L4.py` with `--apply-cut` on
  L3 files and what the output holds; the `check_application.py` check to
  run first; the training commands (noise first, then muon on the noise
  survivors, into a directory of one's own rather than `models/`); a table of
  the three rewritten modules and their agreement with pass2; then the three
  filled fields.
- **`release/physics_caveats.md`**: every known way the package differs from
  the official pass2 L4: the rewritten modules and their agreement, the
  `accumulated_time` residual, `micro_count` following the production, the
  `first_hlc` tie rule and default vertex, VICH, `FullTimeLengthRatio`,
  pulse-series and key names, the L4 boolean, "these are our models, not the
  collaboration's", signal and weights, muon background, frame keys not
  produced, and limitations inherited from the production.
- **`release/models_README.md`**: written into `models/README.md` only when a
  model is missing.  Lists the four model files, says the `.json` is
  required, and says to retrain into a separate directory.

A fact the sidecars do not carry is written as "not recorded", with the
reason; nothing is guessed.

### Flags

| flag | default | meaning |
|---|---|---|
| `--out DIR` | `<repo>/dist/oscnext_l4_release` | where the release is written |
| `--models DIR` | none | directory with `L4_noise_model.txt/.json` and `L4_muon_model.txt/.json` |
| `--no-models` | off | build without models on purpose; `models/README.md` says what belongs there |
| `--partial-models` | off | ship whichever models exist even if one is missing |
| `--list` | off | print the closure and the external imports, then exit; builds nothing |

Exactly one of `--models` and `--no-models` is required (unless `--list`).
The two are mutually exclusive.

Example:

```bash
python scripts/make_release.py --list                       # what would ship
python scripts/make_release.py --models /data/user/$USER/L4_output/models_pass2 \
    --out /data/user/$USER/oscnext_l4_release
python scripts/make_release.py --no-models                   # code only
```

On failure it prints `make_release: FAILED -- nothing was written.` and the
reasons.  On success it prints the release path, the shipped files, the
models, any missing model, the third-party imports, and a warning if the
checkout had uncommitted changes.

### Module-level constants

`ROOT` (repository root), `PKG` (`"oscnext_l4"`), `ENTRY_SCRIPTS` (the four
entry scripts), `TEMPLATE_DIR` (`<repo>/release`), `DOC_TEMPLATES` (the three
templates, read for script names), `MARKER` (`.oscnext_l4_release`), `TAGS`
(`("noise", "muon")`), `FLAVOUR` (sample name → ν symbol), `NOTE_SETS` (what
the technical note says about a GENIE dataset, by the last four digits of
its number).

### `class ReleaseError(Exception)`

The error raised by every check.  `main` catches it and exits with the
message.

### `_parse(path)`

Reads a file and returns its `ast` tree.

### `_module_path(root, modname)`

- **What it does:** maps `oscnext_l4` → `oscnext_l4/__init__.py` and
  `oscnext_l4.x` → `oscnext_l4/x.py`, if that file exists under `root`.
- **Inputs → outputs:** root, dotted name → relative path or `None` (also
  `None` for anything outside the package or nested deeper than one level).

### `_defined_names(tree)`

- **What it does:** collects the names a module defines at top level:
  functions, classes, assignment targets, imported names, including those
  inside top-level `if`/`try`/`with` blocks and `except` handlers (an inner
  helper `visit` walks those blocks).
- **Inputs → outputs:** ast tree → set of names.  Used to check that
  `from oscnext_l4.x import name` names something `x` defines.

### `_imports(tree, rel)`

- **What it does:** walks every node of one file and sorts its imports.
  - `import oscnext_l4...` and `from oscnext_l4... import ...` → internal.
  - `from . import x` / `from .x import y` → internal (only allowed inside
    the package, and only one level up).
  - Everything else → external (the top-level package name is recorded).
  - A call to `import_module` / `__import__` with a string naming the package
    → error (cannot be followed).
- **Inputs → outputs:** tree, relative path → `(internal, external, errors)`:
  `internal` is `[(module, [names], line)]`.

### `_strings(tree, docstrings=False)`

Returns every string constant in a file, leaving out module, class and
function docstrings unless asked.  Used to find script and config names a
file mentions.

### `_json_strings(obj)`

A generator over every string (keys and values) in a loaded JSON object,
recursively.  Used to follow config files that name other config files.

### `closure(root, entry_scripts, docs=())`

- **What it does:** computes every file the release needs.
  1. Lists the available `scripts/*.py` and `config/*` files.
  2. Reads each document in `docs` (HTML comments removed) and adds any
     script it names.
  3. Processes a to-do list starting with the entry scripts and
     `oscnext_l4/__init__.py`: parses each file, collects its imports
     (errors for bad relative or dynamic imports), adds each internal module
     to the list, checks each imported name exists in its module (a name
     that is itself a submodule of the package is added as a module), and
     adds scripts named in its strings (inner helper `named_scripts`, a regex
     for `name.py` or `scripts/name.py`).
  4. Repeats over `config/` until nothing changes: a config file is added if
     any collected string equals its name or contains `config/<name>`; the
     strings of an added JSON file are collected too.
- **Inputs → outputs:** root, entry scripts, docs → `(files, external,
  errors)`: a set of relative paths, a set of external package names, a list
  of error messages.

### `_sha256(path)`

Hex SHA-256 of a file, read in 1 MB blocks.

### `_booster_features(txt)`

Reads the `feature_names=` line of a LightGBM text model (stopping at the
first tree) and returns the feature list, or `None`.  No lightgbm needed.

### `check_models(model_dir, partial)`

- **What it does:** the model checks listed above, for both tags.
  1. Imports `FEATURE_MAP`, `NOISE_FEATURES`, `MUON_FEATURES` from
     `oscnext_l4.varmap` (adds `ROOT` to `sys.path`).
  2. Per tag: records a tag with neither file as missing; errors on a lone
     `.txt` or `.json`; loads the sidecar; checks feature order against the
     booster, features against `FEATURE_MAP`, the sha256 and the tree count;
     warns on a non-standard input list.
  3. If both are present, checks the muon model's
     `provenance.noise_cut.model_sha256` against the noise model's sha256.
  4. Raises `ReleaseError` on any error, if no model at all was found, or if
     one is missing and `partial` is false.
- **Inputs → outputs:** directory, flag → `(found, missing)`: `found` maps tag
  → `{txt, json, meta, sha256, warnings}`; `missing` lists absent tags.
- **Watch out:** the pairing check only runs when the muon sidecar records
  provenance.  Models trained from notebook-built datasets carry none, so
  their pairing is not checked.

### `_dataset_ids(patterns)`

Extracts dataset numbers (4-6 digits between slashes) from L3 path patterns,
in order, without duplicates.

### `_note_set(ds)`

For a GENIE dataset number `1[246]xxxx`, returns the technical note's
description of it from `NOTE_SETS` (e.g. `"0511"` → a bulk-ice set with
scattering −30 %, Table 5), or `None`.

### `_join(items)`

Joins a list in English: `"a"`, `"a and b"`, `"a, b and c"`.

### `_pct(x)`

Formats a fraction as a percentage with one decimal, e.g. `"95.9 %"`.

### `text_at_default_cut(models)`

Returns the sentence(s) for `{{AT_DEFAULT_CUT}}`: per tag, "our <tag> model
keeps X % of the neutrinos and rejects Y % of the noise/muons at P ≥ c (N
background events left; trained <date>)" from `metrics.at_default_cut`, or
a "not shipped" / "not recorded" sentence.

### `_provenance(models, tag)`

Returns `models[tag]["meta"]["provenance"]`, or `None`.

### `text_signal(models)`

- **What it does:** writes the `{{SIGNAL}}` text from the provenance.
  1. "Not recorded" if no model was shipped or none has provenance.
  2. If the two models were trained on different signal sets, says so and
     describes each; otherwise describes one.
  3. Names the signal sets with dataset numbers and flavours, and the
     production.
  4. For each dataset, what the technical note says about it (via
     `_note_set`), or that it is not identified; adds that the production
     used the nominal sets when a non-nominal or unidentified set is used.
  5. For GENIE-only signal with a recorded norm and index, states the power
     law and that the production trained with a different, physical weight.
     Otherwise names the weighting schemes.
  6. Notes which model predates provenance.

### `text_muon_background(models)`

- **What it does:** writes the `{{MUON_BACKGROUND}}` text: what the muon
  model's background was (detector data with its run count, MuonGun or
  CORSIKA with dataset numbers, or the scheme name), and whether the noise
  cut was applied, at what threshold, and whether that noise model is the one
  shipped (compared by sha256).  "Not recorded" when there is no muon model
  or no provenance.

### `text_models(models, missing)`

- **What it does:** writes the `{{MODELS}}` text: a Markdown table with one
  column per shipped model and rows for files, training date, LightGBM
  version, trees, inputs, train/test event counts, the result at the default
  cut (inner helper `adc`), what it was trained on (inner helper `prov`) and
  the first 16 characters of the sha256.  An inner helper `row` appends one
  table row.  Below it, the model warnings and a note for each missing
  model.  With no models, a paragraph saying `--apply-cut` cannot run.

### `_fill(template, fields)`

Replaces each `{{KEY}}` with its text, removes `<!-- Template: ... -->`
comments, and raises `ReleaseError` if any `{{UPPER_CASE}}`, `<fill ...>` or
`<model-dependent ...>` marker is left.

### `_git(*args)`

Runs `git -C ROOT <args>` with a 10 s timeout; returns the trimmed output, or
`None` on any failure.

### `_contents(files, external, models, build)`

Writes the `{{CONTENTS}}` text: the file tree grouped by directory in a code
block, the third-party packages the code imports (standard-library modules
removed; `h5py`, `simweights` and `matplotlib` noted as optional), and the
build date with the commit (and "with uncommitted changes" when dirty) or
"not from a git checkout".

### `build(out, model_dir, partial)`

- **What it does:** the whole build, in the order described in "How the
  release is assembled": closure (error → `ReleaseError`), template presence
  check, model check (if `model_dir`), git info, refusal to replace a
  non-release `--out`, build in `<out>.tmp-build`, `verify`, marker file,
  and the final swap.  Any exception deletes the temporary directory and is
  re-raised.
- **Inputs → outputs:** output path, models directory or `None`, partial flag
  → `(out, files, external, models, missing, info)`.
- **Side effects:** writes the release directory; removes an earlier release
  at `out`.

### `verify(tree, files)`

- **What it does:** checks the copy.
  1. Re-runs `closure` inside the release, with the filled `README.md` and
     `models/README.md` as documents; any error is fatal.
  2. The file set found there must equal the source's (`extra` / `missing`
     are reported).
  3. Every shipped `.py` must compile.
  4. No `__pycache__` or `.pyc` may be present.
- **Inputs → outputs:** release directory, source file set → nothing, or
  `ReleaseError`.

### `main()`

Parses the flags, enforces "`--models` or `--no-models`" and their mutual
exclusion, runs `--list` or `build`, turns a `ReleaseError` into an exit
with the message, and prints the summary.

---

## presentation/make_figures.py

**What it is for.**  Makes the figures for the slides from files already on
disk: the training set, the model sidecar and the plots the training wrote.
Needs only `numpy` and `matplotlib` (plus `scripts/plot_inputs.py`, which it
calls).  If a source file is missing it says which and carries on.

**Reads** (with `<suf>` = `""` for pass3, `"_pass2"` for pass2):
`<out-root>/models<suf>/L4_<tag>_model.json`,
`<out-root>/ds<suf>/L4_<tag>_dataset.npz`, and
`<out-root>/models<suf>/<tag>_cuts.png`, `<tag>_dist.png`.

**Writes** into `presentation/figures/` (by default): `4.png` (input
distributions; also `<tag>_inputs.png`), `feature_importance.png`,
`input_correlation_signal.png`, `input_correlation_background.png`,
`lightgbm_cuts.png`, `lightgbm_score_dist.png`.

| flag | default | meaning |
|---|---|---|
| `--out-root DIR` | `$OSCNEXT_OUT_ROOT`, else `<repo>/L4_output` | the output tree to read |
| `--tag TAG` | `noise` | which classifier (`noise` or `muon`) |
| `--production P` | `pass3` | `pass2` or `pass3`; picks the directory suffix |
| `--figures DIR` | `presentation/figures` | where figures are written |

> **Watch out.**  The default output root differs from the notebook's: the
> notebook uses `/data/user/$USER/L4_output` when it exists, this script
> uses `<repo>/L4_output`.  Pass `--out-root` or set `OSCNEXT_OUT_ROOT`.

Functions, in file order:

- **`default_out_root()`**: `OSCNEXT_OUT_ROOT`, else `<repo>/L4_output`.
- **`feature_importance(json_path, out_path)`**: reads `importance_gain`
  from the sidecar and draws a horizontal bar chart of each feature's share
  of the total gain, labelled in percent.  Returns `False` if the sidecar has
  no importance.
- **`_rank(a)`**: ranks of an array (0-based), with tied values given their
  average rank.
- **`_spearman(X)`**: the Spearman rank-correlation matrix of the columns of
  `X` (Pearson correlation of the ranks).  Chosen because the inputs have
  very different shapes (a speed over decades, small integers, values piled
  near zero).
- **`input_correlation(npz_path, out_dir)`**: loads the `.npz`, drops events
  with any non-finite input, and for signal and background separately (each
  needs at least 10 events) draws the correlation matrix as a heat map with
  the values written in the cells.  Returns the number of files written.
- **`input_distributions(npz_path, figures, tag)`**: runs
  `scripts/plot_inputs.py <npz> --outdir <figures> --tag <tag>` (so the plot
  is the same one the analysis uses), then copies `<tag>_inputs.png` to
  `4.png`.  Returns `False` if the script is missing, fails or writes
  nothing.
- **`copy_as(src, dst)`**: copies a training plot under the name the slides
  expect, or prints how to produce it.
- **`main()`**: parses the flags, builds the paths, runs the four steps
  (input distributions, feature importance, correlation, the two copies),
  and prints how many of the 6 figures were produced.  Always returns 0.

## presentation/make_update_pdf.py

**What it is for.**  Turns a weekly update script written in Markdown
(`presentation/updates/<date>.md`) into a PDF to read from while speaking.
The Markdown is the versioned source; the PDF is built from it.  Needs
`reportlab`.  It understands only a small subset: `# title`, a plain line
after the title (subtitle), `## heading`, paragraphs, `` `code` `` spans,
and two-column `| key | value |` tables.

```bash
python presentation/make_update_pdf.py presentation/updates/2026-09-21.md
```

| argument | default | meaning |
|---|---|---|
| `markdown` (positional) | required | the Markdown file |
| `-o`, `--output FILE` | the Markdown path with `.pdf` | the PDF to write |

Functions, in file order:

- **`styles()`**: the reportlab paragraph styles: `title`, `sub`, `head`,
  `body`, `note`.
- **`inline(text)`**: escapes `&`, `<`, `>` for reportlab and turns
  `` `code` `` into a Courier span.
- **`blocks(md)`**: splits the Markdown into `(kind, payload)` blocks in
  order: `title`, `head`, `body` (a paragraph, lines joined), `table` (rows
  of the first two cells; separator rows skipped).  Inner helpers
  `flush_para` and `flush_rows` close the current paragraph or table.  The
  first paragraph right after the title becomes `sub`.
- **`build(md_path, pdf_path)`**: turns the blocks into an A4 PDF.  Once a
  table has appeared, later paragraphs use the smaller `note` style; a
  heading starting with "Numbers" that comes before any table gets a
  horizontal rule above it (the reference block at the end).  Prints the
  output path.
- **`main()`**: parses the arguments and calls `build`.  Returns 0.
