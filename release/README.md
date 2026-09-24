# oscNext Level 3 → Level 4, runnable on IceTray v1.17.0

<!-- Template: scripts/make_release.py fills the {{...}} fields from the
     model sidecars and the release it builds.  Edit this file, not the copy
     in a built release. -->

This directory is the L3 → L4 step of the oscNext low-energy selection, cut
down to what three things need:

1. **An L3 `.i3` file becomes an L4 `.i3` file**, with the L4 variables, both
   classifier scores and the `L4_oscNext_bool` added: `scripts/process_L4.py`.
2. **The training**: `scripts/make_dataset.py` (booked HDF5 → training set)
   and `scripts/train_L4_classifier.py` (training set → model).
3. **The trained models**: `models/`.

The L4 variables were verified event by event against the official pass2
production; how, and where this package still differs from it, is in *Physics
caveats* below.

## Environment

IceTray **v1.17.0** from cvmfs (`py3-v4.4.2`, `RHEL_9_x86_64_v2`): the
comparison with pass2 below was run there.  It ships `lightgbm` (4.5.0),
`numpy` and `pytables`, which is all the scripts need beyond IceTray.

```bash
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)
/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/RHEL_9_x86_64_v2/metaprojects/icetray/v1.17.0/env-shell.sh
cd <this directory>
python scripts/diagnose_env.py        # what is and is not in this environment
```

`env-shell.sh` opens a **new shell**: in a job script, run one command through
it instead (`.../env-shell.sh -- python scripts/process_L4.py ...`), or the
command after it runs without IceTray.

The scripts find their package themselves (`oscnext_l4/` and `config/` must stay
next to `scripts/`); nothing needs installing.  `make_dataset.py`,
`train_L4_classifier.py` and `check_application.py` need no IceTray at all,
only numpy, pytables and lightgbm.

## 1. L3 `.i3` → L4 `.i3`

```bash
python scripts/process_L4.py \
    --gcd   <GCD file> \
    --input <L3 file(s), globs accepted> \
    --output-i3 L4_<name>.i3.zst \
    --apply-cut --model-dir models \
    --mc --genie            # GENIE;  --noise for vuvuzela, --muongun, --corsika,
                            # nothing for detector data
```

For **pass2** L3 add `--cleaned-pulses SRTTWOfflinePulsesDC` (pass3's cleaned
series is `SRTTWSplitInIcePulsesDC`, the default).  That flag is the whole
difference between the two productions.

What the output holds:

- the L3 frames plus every `L4_*` key: `L4_first_hlc`, `L4_first_hlc_rho`,
  `L4_iLineFit(Params)`, `L4_VICH_nch/npulses/qtot`, `L4_accumulated_time`,
  `L4_micro_count`, `L4_fill_ratio`, `L4_FullTimeLengthRatio`, the recomputed
  `SRTTWSplitInIcePulsesDCHitStatistics/HitMultiplicity`,
  `L4_NoiseStraightCuts_Bool`;
- `L4_NoiseClassifier_ProbNu` and `L4_MuonClassifier_Data_ProbNu`, and
  **`L4_oscNext_bool`** (noise ≥ 0.70 AND muon ≥ 0.65), written for **every**
  event: nothing is dropped at L4;
- only Physics frames that pass the L3 cut and are in `InIceSplit` (as in the
  production), and only the DAQ frames that still have one;
- **no GCD frames**: keep the GCD file beside it, as for the production's files.

Beside the output: `<output>.meta.json` (how many L3 files went in) and, if any
input was corrupt and dropped, `<output>.badfiles.txt`.  The output appears
under its real name only when the run has finished (`_incomplete_<name>`
until then).  `--help` lists the rest (`--chunk-files`, `--scan`, `--n` for a
smoke test, the flags that select the technical note's readings instead of the
production's).  For many files, `scripts/scan_files.py` finds the corrupt ones
once, up front.

### Check this first: does the tray read the model inputs correctly?

Training reads the model inputs from HDF5 columns; the tray reads them from
frame objects.  One input (`cog_z`) was read from a field its frame object does
not have, which nothing could see until the two paths were compared on a real
file.  Compare them once on your files before using the scores:

```bash
python scripts/process_L4.py --gcd <GCD> --input <one L3 file> \
    --output-hdf5 check.hdf5 --apply-cut --model-dir models [the flags above]
python scripts/check_application.py check.hdf5 --model-dir models
```

It recomputes each score from the booked columns exactly as training reads
them and compares it event by event with the score the tray wrote; it must say
`PASS`.

## 2. Training

Three steps per classifier, noise first: the muon training set is built only
from events that pass the trained noise model (P ≥ 0.70), as in the
production.

```bash
# (a) L3 -> booked HDF5, one tree per sample:  <HDF>/<sample>/L4_<sample>*.hdf5
python scripts/process_L4.py --gcd <GCD> --input "<L3 glob>" \
    --output-hdf5 <HDF>/nue/L4_nue.hdf5 --chunk-files 10 --mc --genie
#     ... likewise for every sample of the production in config/productions.json

# (b) noise
python scripts/make_dataset.py --stage noise --production pass2 \
    --hdf-base <HDF> --outdir <DS>
python scripts/train_L4_classifier.py --tag noise \
    --dataset <DS>/L4_noise_dataset.npz --outdir models

# (c) muon, on the survivors of the noise model just trained
python scripts/make_dataset.py --stage muon --production pass2 \
    --hdf-base <HDF> --outdir <DS> --noise-model models/L4_noise_model.txt
python scripts/train_L4_classifier.py --tag muon \
    --dataset <DS>/L4_muon_dataset.npz --outdir models
```

`config/productions.json` says which samples each production has, their role
(signal / noise background / muon background) and how each is weighted;
`config/README.md` explains every value.  The noise and muon sets share one
signal train/test split, so the combined cut has a common held-out set.  Each
model's `.json` records what made it: production, samples, weighting, and for
the muon model the noise model it was cut with (by sha256).

## 3. The models

{{MODELS}}

## Rewritten modules

The L4 segment calls three modules this IceTray does not carry.  They are
rewritten in pure Python in `oscnext_l4/rewritten.py`, each from the original
source, and each checked event by event against the real pass2 L4 files:

| variable(s) | original module (project) | agreement with pass2 |
|---|---|---|
| `L4_first_hlc`, `L4_first_hlc_rho` | `FirstHLC<I3RecoPulse>` (SimpleVertex) | bitwise identical |
| `L4_accumulated_time` (and `L4_separation_in_cogs`, not a BDT input) | `CalculateVariables` (analysis, event_selection) | 99.84 % identical |
| `L4_VICH_nch`, `L4_VICH_npulses`, `L4_VICH_qtot` | `I3CutL7Module` (tau_bdt) | 100 % identical |

The docstring of each function says which implementation the production ran
and what the measurement was.

{{PHYSICS_CAVEATS}}

## What is in this directory

{{CONTENTS}}
