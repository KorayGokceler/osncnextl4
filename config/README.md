# `productions.json` — what every field means, and which ones bite

This file is the one place that knows a per-production fact.  `select()` in
`oscnext_l4/productions.py` reads it and returns `(GCD, SAMPLES)`; everything
downstream — booking, the registry check, loading, weights, datasets,
training — is production-independent and needs no change.

**Read this before editing the JSON.**  Several values look arbitrary and are
not: they were arrived at by measurement, and three of them fail *silently*
when wrong.  JSON cannot carry comments, which is the reason this file exists;
keep the two in step.

---

## The three that fail silently

A wrong value here does not raise.  It produces a plausible number that is
wrong, and the symptom appears much later.

### 1. `noise_weight_unit` — `"per_ns"` (pass3) or `"hz"` (pass2)

The vuvuzela noise weight is stored in 1/ns at pass3 and already in Hz at
pass2.  Getting it wrong **scales every noise rate by 1e9**, and nothing
checks it.  The only symptom is an absurd rate in the training, long after the
cause.

`select()` sets it as part of switching production, so there is no supported
way to pick pass2 and forget.  Verified at pass3 by measurement: the noise
rate came out at 41.4 mHz against Table 13's 36.6 mHz, agreement within 13%.

### 2. `cleaned_pulses`

`null` means the code's own default, `SRTTWSplitInIcePulsesDC` (pass3).  pass2
uses `SRTTWOfflinePulsesDC`.  The uncleaned series is `SplitInIcePulses` in
**both**, which is why it is not configured here.

Technical note sec. 3.2 (p.25) defines both outright, and the official
project's `selection/globals.py` says the same.  **This single field is the
entire delta between a pass3 run and a pass2 run**; everything else adapts.

### 3. `weight` — a SCHEME, never a sample name

`genie` / `noise` / `corsika` / `muongun` / `data`.  This is what lets one
code serve both productions.  Both have a `muon_bg`, but pass3's is CORSIKA
and pass2's is MuonGun: different input columns, different formula.  Keying
anything downstream on the sample NAME breaks the moment a production names
its sets differently; keying on the scheme does not.

`data.AUX_SCHEMES` asks each scheme for its own weight columns, so a MuonGun
file is never asked for `CorsikaWeightMap` and no false alarm is printed.

---

## Fields

| field | meaning |
|---|---|
| `gcd` | the GCD for every MC sample of this production |
| `cleaned_pulses` | see above; `null` = the code's default |
| `noise_weight_unit` | see above |
| `samples` | name → sample spec, **in order**; the order is preserved |

Per sample:

| field | meaning |
|---|---|
| `l3` | a glob, or a list of globs (a set that is two datasets) |
| `flags` | passed to `process_L4.py` verbatim |
| `kind` | the ROLE: `signal` / `noise_bg` / `muon_bg` |
| `weight` | the weighting SCHEME (above) |
| `gcd` | *optional*, a list parallel to `l3` — one GCD per pattern |
| `runs_by_year` + `l3_template` + `gcd_template` | *optional*, expand to `l3` and `gcd` (below) |

### `kind` is a role, and roles are why one notebook serves both

`bdt_roles` says what each classifier needs: the noise BDT takes
`signal + noise_bg`, the muon BDT `signal + muon_bg`.  Stated as roles rather
than names, a production that has an extra signal set gets it and one that
does not is unaffected.

This used to be the name list `("nue", "numu", "noise")`, written when pass3
was the only production.  pass2 HAS a NuTau set (160511) and it IS signal —
the production's own `L4_model_data.py` harvests 12xxxx, 14xxxx and 16xxxx
alike — so the name list silently dropped it.  Roles do not have that failure
mode.

### The GCD is verified, not assumed

That pass2's averaged GCD is the right one was checked rather than trusted:
`cog_z`, `z_sigma` and `z_travel`, which we compute from it and pass2 stored
from theirs, agree **to the last bit over 8144 events**.

---

## `pass2.samples.data` — the templated one

Detector data is the production's actual muon background (note sec. 3.6.3).
It is the only sample with `runs_by_year` and templates rather than a literal
`l3`, and the only one with a per-pattern `gcd`.

### Why templates instead of 18 written-out paths

The run numbers would otherwise appear twice — once in each of the L3 and GCD
lists — and could drift apart.  The loader substitutes `{run:08d}` and
`{season:02d}`, where **season = year − 2000** (`IC86.12` holds the runs taken
in 2012).  That is one defined substitution, not a template language.

### The 18 runs are the note's, not ours

Technical note sec. 3.6.3 (p.39) names them:

> The background sample is comprised of the following data runs, which were
> selected to cover years roughly equally in order to avoid strong dependence
> of the muon rejection on the specific season due to muon flux seasonal
> variations.

Three per year, 2012–2017.  **The even coverage is the point of the list.**
The atmospheric muon flux varies seasonally, so a background drawn from one
part of the year teaches the classifier that season rather than the muon.  If
you substitute runs, keep the property, not the count.

Two conditions come with the choice, and neither is optional:

1. **The noise cut comes first.**  The background is data that has already
   passed the noise classifier at 0.7 — that is what makes it 99% muon
   (Table 13: 490 mHz muon against 5.2 mHz neutrino and 0.28 mHz noise).
   Training on data that has not been through it spends the muon BDT's
   capacity re-learning a rejection the noise BDT already did.  The note's own
   Figures 19–24 carry `L4_NoiseClassifier_ProbNu > 0.7` in their cut box.
2. **The signal side stays GENIE MC.**  It is a data-vs-MC classifier by
   construction.

The fridge's `L4_model_data.py` says *"one run per month 2012-2018"* — a later
and larger list.  Where they differ, the note is what the published Table 13
numbers were produced with.

### `l3_template` cannot be `*.i3.zst`

The layout is verified on disk:

```
.../data/level3/IC86.12/Run00120200/
    Level2pass2_IC86.2012_data_Run00120200_0527_1_20_GCD.i3.zst
    oscNext_data_IC86.12_level3_v02.00_pass2_Run00120200_Subrun00000000.i3.zst
    oscNext_data_..._Subrun00000000.hdf5
    oscNext_data_..._Subrun00000000.json
```

Each subrun contributes **three** files and the run's GCD ends in
`_GCD.i3.zst`, so `*.i3.zst` would hand I3Reader a GCD as an input and count
three entries per subrun.  It also inflates the file count threefold: a
directory of 634 entries is about 211 L3 files, and the 18 runs come to
roughly 4,400 rather than the 13,125 a naive `ls | wc -l` suggests.

### `gcd_template` must end `.i3*`, not `.i3.zst`

The GCD lives in the run directory, one per run — detector data cannot use the
averaged MC GCD, because the dead DOMs and the calibration are exactly what
changes between runs.  Its name carries fields that vary per run
(`_0527_1_20_`), so it is globbed rather than constructed.

**The extension is not constant.**  Most runs carry `_GCD.i3.zst`; the 2015
and early-2016 runs carry `_GCD.i3.gz`, while the L3 files beside them are
`.i3.zst` either way, so nothing else gives it away.  A `.i3.zst` pattern
matched nothing there and silently dropped **all of 2015 and two thirds of
2016** — precisely the seasonal gap the run list exists to prevent.  `_GCD.i3`
is specific enough that nothing else in the directory can match.

### MuonGun and data share `muon_bg` on purpose — and must never be stacked

So that a muon-BDT run picks up whichever the production offers.  But one is
measured and one is simulated: a model fitted on their union learns the
difference between simulation and measurement as much as it learns the muon.
Section 6 of the notebook picks one and says which, preferring data as the
production does.

### `data` is the only sample with empty `flags`

No truth, no MC weight dict, nothing to propagate.  An empty list is exactly
what "this is data" means to `process_L4.build_key_list`, which is why
`process_L4.py` needed no change to support it.

---

## What differs between the two productions

Verified against real files and the technical note; the full account is in
`docs/pass2_verification.md`.

| | pass3 | pass2 |
|---|---|---|
| cleaned pulses | `SRTTWSplitInIcePulsesDC` | `SRTTWOfflinePulsesDC` |
| uncleaned pulses | `SplitInIcePulses` | the same |
| noise weight unit | 1/ns | Hz |
| L3 cut | `L3_oscNext_bool` (data quality ANDed by the pass3 script) | `L3_oscNext_bool` too — `oscNext_L3.py` folds data quality INTO `IC2018_LE_L3_Full`, so both branches of `l3_cut` agree and the cut is applied either way |
| `I3GenieInfo` | present | **absent** — the weight falls back to `NEvents × gen_ratio`, which is the production's own formula |
| muon background | CORSIKA | MuonGun, and detector data |

An earlier version of this table claimed pass2 drops the data-quality cut.
That was read out of the technical note and is wrong; the production L3 script
settles it.  See CLAUDE.md, "Running on pass2".

---

## Pointing at a different config

```bash
export OSCNEXT_L4_CONFIG=/path/to/my_productions.json
```

`select()` reads that instead.  The file must have the same shape; the loader
validates the fields it needs and names what is missing rather than failing
later with a KeyError.
