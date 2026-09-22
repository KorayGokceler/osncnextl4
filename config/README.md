# `config/` — the JSON tables, and which fields bite

Two files, both **pure data**: reading either is `json.load()` and nothing
else.  `productions.json` knows the per-production facts (paths, GCD, pulse
series, weight schemes); `variables.json` knows every variable, both where it
sits in a booked HDF5 file and where it sits in an I3 frame.  This README
carries what JSON cannot: why each value is what it is, and which ones fail
*silently* when wrong.  **Read it before editing either file.**

---

# `productions.json` — what every field means, and which ones bite

This file is the one place that knows a per-production fact.  It is **pure
data**: reading it is `json.load()` and nothing else, and section 1 of the
notebook builds `(GCD, SAMPLES)` from it in about twenty lines.  Everything
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

The notebook calls `set_noise_weight_unit` in the same cell that picks the
production, beside the thing it depends on, so picking pass2 and forgetting
the unit is not an available mistake.  Verified at pass3 by measurement: the
noise rate came out at 41.4 mHz against Table 13's 36.6 mHz, agreement within
13%.

### 2. `cleaned_pulses`

`SRTTWSplitInIcePulsesDC` at pass3, `SRTTWOfflinePulsesDC` at pass2.  The
uncleaned series is `SplitInIcePulses` in **both**, which is why it is not
configured here.

**Both are written out, and pass3's is not left `null`.**  It used to be:
`null` meant "pass no `--cleaned-pulses` flag and let `process_L4.py`'s
default win", and that default is `CLEANED_PULSES_DEFAULT` in
`oscnext_l4/variables.py` — pass3's name, in a different file.  So the config
claimed to be the one place that knows a per-production fact while deferring
one of them to a Python constant.  Writing it out costs a flag on the command
line that names the series it was already using, and makes the run
self-describing.  Anything reading this field (`pass2.py` does) now gets a
string for either production instead of `None` for one of them.

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
| `__…__` | ignored by every reader; a place to leave a note in a format with no comments |

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

### The 18 paths are written out, and that is on purpose

They were templated once, with `{run:08d}` and `{season:02d}` expanded by a
loader.  Writing them out is what lets this file be PURE DATA — reading it is
`json.load()` and nothing else, with no module in between.

The cost is that the run numbers appear in both the `l3` and the `gcd` list
and could in principle drift apart.  That failure is LOUD, not silent:
`runner.run_gcd_pairs` refuses a sample whose two lists differ in length, and
a run whose GCD pattern matches nothing is skipped by name with the directory
listing printed beside it.  Edit the two together.

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

Section 1 of the notebook reads that instead, and `oscnext_l4/pass2.py` with
it.  The file must have the same shape.

There is no `select()` and no `productions.py` any more: the notebook does the
twenty lines of filtering and path-building itself, in the same cell as the
production switch.  That cell is also where `set_noise_weight_unit` is
called — beside the thing it depends on, so selecting pass2 and forgetting the
unit is not an available mistake.

---

# `variables.json` — one row per variable, both sides of it

Until now this table was **six** hand-written dicts spread over two modules,
all keyed by the same variable names: `REGISTRY`, `ALTS`, `AUX` and
`AUX_SCHEMES` in `oscnext_l4/data.py`, `FEATURE_MAP` and `COLUMN_ALTS` in
`oscnext_l4/classifier.py`.  They are now one row each, and
`oscnext_l4/varmap.py` turns that row into the views the code still uses under
their old names.

## Why one row — the critical synchronisation point

Training reads a **column out of an HDF5 file**.  The trained model is applied
to an **I3 frame**.  Those are different media with different names for the
same quantity — the hdfwriter converter renames, so `fill_ratio_from_mean` in
the frame is `fillratio_from_mean` in the file.  Two tables, two spellings,
one variable.

If one side is edited and the other is not, training fits one quantity and
application reads another, and **the model produces wrong predictions without
raising**.  Nothing in the output says so.

The old defence was a checker that parsed `classifier.py` with `ast` — parsed
rather than imported, since `data.py` cannot import `classifier.py` — and
compared the two dicts.  That is ~95 lines of machinery whose whole job was to
detect an edit that should not have been possible.  With one row it is not
possible: `data.check_feature_map()` is now a dict comparison, and still runs
in the notebook.

It is not pure ceremony, though.  The two sides of a row can still be made to
name *different quantities* by hand, and the check catches exactly that: for
every row it asks whether the `hdf5` accept-set and the `frame` accept-set
intersect, and whether every BDT input has both sides at all.

## The shape of a row

```json
"fill_ratio": {
  "hdf5":       ["L4_fill_ratio", "fillratio_from_mean"],
  "hdf5_alts": [["L4_fill_ratio", "fill_ratio_from_mean"], ...],
  "frame":      ["L4_fill_ratio", "fill_ratio_from_mean"],
  "frame_alts": ["fillratio_from_mean"]
}
```

| field | meaning |
|---|---|
| `hdf5` | the `(table, column)` a booked HDF5 file holds — what training reads |
| `hdf5_alts` | further `(table, column)` pairs naming the **same** quantity; whichever is actually in the file is used |
| `frame` | the `(key, field)` the same quantity has in an I3 frame — what the tray reads |
| `frame_alts` | field names that change between versions, tried in order |
| `schemes` | **present only on weight columns.** Its presence is what marks a row as a weight column rather than a BDT input |

A row with `schemes` lands in `AUX`/`AUX_SCHEMES`; every other row lands in
`REGISTRY`, and lands in `FEATURE_MAP` as well if it has a `frame` side.
`hdf5_alts` produces `ALTS` (with the primary pair first), `frame_alts`
produces `COLUMN_ALTS`.

**`hdf5_alts` must only ever hold the same physical quantity.**  For
`fill_ratio` every entry is the fill ratio *about the mean*; the file also
carries `from_rms` and `from_nch`, and putting one of those here would read a
different piece of physics silently.

## `bdt_features` — the order is load-bearing

```json
"bdt_features": {"noise": [5 names], "muon": [10 names]}
```

These are the model's input vectors, and **the order is the feature order**.
It is written out explicitly rather than derived from the order of the
`variables` block, because the two differ: `NchCleaned` is an input to *both*
BDTs, so it sits in the noise block of `variables` but third in the muon list.
Deriving the lists from key order would silently hand the muon model a
differently ordered vector.

Both lists are confirmed against the production source
(`L4_noise_model_train.py` and `L4_muon_model_train.py` in the fridge), not
read off Tables 11/12 — 5 + 10, 14 unique.  Table 12 lists ten variables and
`NchCleaned` was once missing from the muon list because it was already in the
noise one.

## The spellings that were measured, not assumed

| variable | why it reads the way it does |
|---|---|
| `fill_ratio` | **`fillratio_from_mean`**, no underscore — verified against real pass3 output.  Table 11 writes `fill_ratio_from_mean`, which is the *frame* spelling; the converter renames |
| `iLineFit_speed` | `lf_vel` in the file.  `LFVel` was the old assumption and Table 11 names it `L4_iLineFit.speed`; all three are alternatives, and the first run caught this as a real conflict |
| `noise_weight` | the column is **`weight`**, not `value`.  Being wrong left the noise weight entirely NaN, so the noise sample's `w_phys` would have been zero |
| `pdg` | `I3MCWeightDict.PrimaryNeutrinoType` at pass3; pass2's older genie-icetray has no such column and it comes from `MCInIcePrimary.pdg_encoding`.  `genie_weight` needs the **sign** of this, and a missing value is a silent 2.33× error on antineutrinos — so only spellings that are unambiguously the primary neutrino's type belong here |
| `gen_ratio` | the production divides by `NEvents × gen_ratio`.  pass2 **stores** the ratio, so it is read rather than reconstructed from the neutrino's sign |
| `micro_count` | `STW_m3500p4000_DTW200`; `STW7500_DTW200` is an older naming the pass2 source contradicts elsewhere |
| `cog_z`, `z_sigma`, `z_travel` | the hit-statistics key keeps the **pass3** spelling (`SRTTWSplitInIcePulsesDC…`) on a pass2 run too.  The values are computed from whichever series was actually passed; only the label is fixed.  Deliberate and cosmetic — writer and reader use the same literal, so they stay consistent |
| `MuonWeight*` | three spellings are booked because which one a production wrote is not fixed; the first that is present is used |

## `AUX` is not uniform across samples — ask `aux_for`

Not every weight column exists in every sample: `noise_weight` is only in the
vuvuzela noise MC, `OneWeight`/`PrimaryNeutrino*` only in GENIE.  Checking all
of them against a single sample prints **false alarms**, three per file for a
MuonGun set asked about `CorsikaWeightMap`.

`schemes` is keyed by weight **scheme**, not by `kind`, for the same reason
`productions.json` is: one kind can carry several schemes — pass3's `muon_bg`
is CORSIKA and pass2's is MuonGun.

`schemes_without_aux` lists the schemes that have **no** weight columns by
design, so an empty list can be told apart from a caller that built its list
wrongly.  Detector data is the case: its weight is `1/livetime`, a property of
the runs, with nothing per-event to read.

`kind_to_scheme` is only a fallback for a sample spec written before
`productions.json` had a `weight` field.

## Adding a variable

1. Add one row with both sides.  If you only have one side, say so by leaving
   the other out — a BDT input without a `frame` side is reported as a
   conflict, a candidate without one is not.
2. If it is a BDT input, add its name to the right `bdt_features` list, at the
   position the model should see it.
3. Run `data.check_feature_map()`.  It reads the config and nothing else.
4. Run `data.check_registry(TABLES, names)` against a real file to confirm the
   column is actually there under that spelling.

`OSCNEXT_L4_VARIABLES` points `varmap` at a different file, the way
`OSCNEXT_L4_CONFIG` does for `productions.json`.
