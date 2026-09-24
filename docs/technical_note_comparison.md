# What we write to HDF5 + a line-by-line comparison with the technical note

Source: `reference/OscNext_v00.074_pass2_technical_note.pdf` (pass2, 83 pages).
Relevant sections: 3.4 (DeepCore Filter), 3.5.1 (L3 variables, Tables 7-8),
3.6 (L4, Tables 10-13).

> **The note is for pass2.** The pipeline runs on pass3 and on pass2 (the
> `--cleaned-pulses` flag is the whole difference).  Most of the L4 logic is
> the same; the naming changed (marked below).

> **READ THIS FIRST -- what is current and what is history.**  Part 1 and
> sections 2.4-2.7 describe the code as it is.  Sections 2.1-2.3 and the
> "Priority order" were written BEFORE the pass2 cross-check and are kept as
> the record of how the questions were posed; where they disagree with the
> code, the code and CLAUDE.md win.  In short, what changed since:
> - **VICH** is `I3CutL7Module`'s four causality bands around the trigger
>   (`oscnext_l4/rewritten.py`, `vich`), 100.00% identical to pass2 -- not
>   the speed window and fiducial COG described in 2.1-2.2.
> - **accumulated_time** follows `CalculateVariables`' charge-quartile rule by
>   default (99.84% identical to pass2); `--accumulated-time-note` gives the
>   note's crossing.  It was never an off-by-one.
> - **micro_count** starts from the UNCLEANED series by default, as pass2
>   does (bitwise identical); `--micro-count-cleaned` follows Table 11.
> - All five rewritten inputs are verified: 14 of 15 compared rows are
>   bitwise identical over 56,301 events (`verification/README.md`).

---

# Part 1 — Exactly what we write to HDF5

`scripts/process_L4.py` builds a **key list** (`build_key_list`) and the booker
turns every key into **its own HDF5 table**.  The table name is the frame key.
Every table carries the shared index `Run`, `Event`, `SubEvent`.

How a frame object becomes columns (hdfwriter's per-type converters):

| Frame type | in HDF5 |
|---|---|
| `I3Double` / `I3Int` / `I3Bool` | a single column: `value` |
| `I3MapStringDouble` / `I3MapStringInt` | **one column per map key** |
| `I3Particle` | `x, y, z, time, energy, length, speed, zenith, azimuth` |
| `I3LineFitParams` | `lf_vel`, `lf_vel_x/y/z`, `n_hits` (pass3 names) |
| `I3HitStatisticsValues` | `cog_x/y/z`, `z_min`, `z_max`, `z_mean`, `z_sigma`, `z_travel`, … |
| `I3HitMultiplicityValues` | `n_hit_strings`, `n_hit_doms`, `n_hit_doms_one_pulse`, `n_pulses` |
| `I3FillRatioInfo` | `fillratio_from_mean`, … (the converter drops the underscore) |
| `I3EventHeader` | `run_id`, `event_id`, … + `time_start_mjd_day/sec/ns` |

**Pulse series are NOT booked.** An event with 5000 pulses does not fit a flat
table.  Only the summary numbers computed from them are written -- which is all
the BDT needs.

## Keys written

### Always (`BASE_KEYS` + `L3_KEYS` + `COMMON_VAR_KEYS`)

| Key | What for |
|---|---|
| `I3EventHeader` | index + livetime (the MJD fields) |
| `IC2018_LE_L3_Vars` | **4 muon + 1 noise BDT inputs live here** (map → many columns; `FullTimeLengthRatio` is recomputed at L4) |
| `IC2018_LE_L3_bools` | the L3 cut flags |
| `L3_oscNext_bool` | `IC2018_LE_L3_Full AND Data_quality_bool` |
| `Data_quality_bool` | SLOP / LID errata data quality |
| `SRTTWSplitInIcePulsesDCHitStatistics` | `cog_z`, `z_sigma`, `z_travel` (muon BDT) |
| `SRTTWSplitInIcePulsesDCHitMultiplicity` | `n_hit_doms` (candidate) |

### What we produce at L4 (`L4_HDF5_KEYS`)

| Key | Type | A BDT input? |
|---|---|---|
| `L4_micro_count` | map | **yes** — noise |
| `L4_fill_ratio` | I3FillRatioInfo | **yes** — noise |
| `L4_iLineFit` + `L4_iLineFitParams` | I3Particle + Params | **yes** — noise (the speed) |
| `L4_FullTimeLengthRatio` | I3Double | **yes** — noise |
| `L4_VICH_nch` | I3Double | **yes** — muon |
| `L4_accumulated_time` | I3Double | **yes** — muon |
| `L4_first_hlc_rho` | I3Double | **yes** — muon |
| `L4_first_hlc` | I3Particle | no (the source of rho) |
| `L4_VICH_npulses`, `L4_VICH_qtot` | I3Double | no (diagnostic) |
| `L4_ToI` + `L4_ToIParams` | I3Particle + Params | no (candidate/legacy) |
| `L4_separation_in_cogs` | I3Double | no |
| `L4_QR_Box` | — | no (no slc-veto, not produced) |
| `L4_n_flux_events` | I3Double | no — **weighting** |
| `L4_oscNext_bool`, `L4_NoiseStraightCuts_Bool` | I3Bool | no |
| `L4_NoiseClassifier_ProbNu`, `L4_MuonClassifier_Data_ProbNu` | I3Double | with `--apply-cut` |

### By sample kind (for the weights)

- `--mc --genie` → `I3MCWeightDict` (`OneWeight`, `NEvents`,
  `PrimaryNeutrinoEnergy/Zenith/Type`), `NEvPerFile`, `I3GenieSystWeightDict`
- `--noise` → `I3MCWeightDict`, `noise_weight` (**no `MCInIcePrimary`**)
- `--corsika` → `CorsikaWeightMap`, `PolyplopiaPrimary`, `I3CorsikaInfo`
- `--muongun` → whichever of the `MuonWeight*` candidates exists

### Sidecar: `<output>.hdf5.meta.json`

`n_l3_files` (the weight divisor), `physics_frames`, `after_stream_filter`,
`booked`, `elapsed_s`.  Written next to the HDF5, not inside it.

---

# Part 1b — Where each variable is computed

For every BDT input: which file/function in the code, which page/table in the
note.  The tray segments are in `oscnext_l4/variables.py`; the three
rewritten modules in `oscnext_l4/rewritten.py`; `FullTimeLengthRatio` in
`oscnext_l4/l3vars.py`.

## Noise BDT — 5 inputs (note p.36, Table 11)

| Variable | Where it is computed | Note |
|---|---|---|
| `NchCleaned` | **we do not compute it** — it arrives from L3 (`IC2018_LE_L3_Vars`) | p.28, sec. 3.5.1 |
| `micro_count` | `_micro_count()`<br>chain inside `oscNext_L4_noise_cut_variables`: StaticTWC → DeepCore fiducial → I3TimeWindowCleaning → count | p.36, Table 11 |
| `iLineFit_speed` | the `linefit.simple` segment, inside `oscNext_L4_atm_muon_classifier_variables`.  Not our code -- an IceTray project. | p.36, Table 11 |
| `fill_ratio` | `I3FillRatioModule` at the end of `oscNext_L4_noise_cut_variables`.  An IceTray project. | p.36, Table 11 |
| `FullTimeLengthRatio` | `_full_time_length_ratio()` | p.36, Table 11 |

## Muon BDT — 10 inputs (note p.41, Table 12)

| Variable | Where it is computed | Note |
|---|---|---|
| `ICVetoHits` | **from L3** (`IC2018_LE_L3_Vars`) | p.28, sec. 3.5.1 |
| `RTVeto250Hits` | **from L3** | p.28-29, sec. 3.5.1 + Table 8 |
| `NchCleaned` | **from L3** | p.28, sec. 3.5.1 |
| `NAbove200Hits` | **from L3** | p.28, sec. 3.5.1 |
| `VICH_nch` | `vich()` in `rewritten.py` — **our rewrite** of `tau_bdt.I3CutL7Module` | p.41, Table 12 (one line); the definition is I3CutL7Module's |
| `accumulated_time` | `_accumulated_time()` — **our rewrite** | p.41, Table 12 |
| `first_hlc_rho` | `_first_hlc()` → `_add_rho_36()` | p.41, Table 12 |
| `cog_z` | the `common_variables.hit_statistics` segment, in `oscNext_L4_hit_statistics`.  An IceTray project. | p.41, Table 12 |
| `z_sigma` | same segment | p.41, Table 12 |
| `z_travel` | same segment | p.41, Table 12 |

## Summary: whose code is it?

| Source | How many | Which |
|---|---|---|
| **Arrives ready from L3** | 4 | `NchCleaned`, `ICVetoHits`, `RTVeto250Hits`, `NAbove200Hits` |
| **IceTray projects** | 5 | `iLineFit_speed` (linefit), `fill_ratio` (fill_ratio), `cog_z`/`z_sigma`/`z_travel` (common_variables) |
| **Our pure-Python rewrites** | 5 | `micro_count`, `FullTimeLengthRatio`, `first_hlc_rho`, `accumulated_time`, `VICH_nch` |

> All of the risk was in the last row, and it has been measured: all five
> agree with the real pass2 L4 files -- bitwise, except `accumulated_time`
> at 99.84% (a tie-order residual, CLAUDE.md open risk 2).

## Helper functions (not inputs, but everything uses them)

| Function | What it does |
|---|---|
| `get_pulses()`, `iter_map()` | `frame_objects.py`: unpack a pulse series (mask or map) and iterate it |
| `calc_rho_36()` | `frame_objects.py`: distance from string 36, bit for bit the production's formula |
| `PropagateGenieInfo` | carries `n_flux_events` from the S frame to the P frames (for weighting) |
| `l3_cut` | inside `oscNext_L4`: `L3_oscNext_bool`, else `IC2018_LE_L3_Full` |

## How to read the note

`reference/OscNext_v00.074_pass2_technical_note.pdf`:

- **p.26-27, sec. 3.4** — the DeepCore Filter.  VICH's speed window
  `[0.25, 0.4] m/ns` and the definition of the "veto region" are **here**, not
  in Table 12.
- **p.27-29, sec. 3.5.1 + Tables 7/8** — the L3 variable definitions
  (`NchCleaned`, `ICVetoHits`, …) and the L3 fiducial DOM list.
- **p.35, Table 10** — the LightGBM hyperparameters.
- **p.36, Table 11** — the noise BDT's 5 inputs.
- **p.41, Table 12** — the muon BDT's 10 inputs.
- **p.48, Table 13** — the L3/L4 rates (the weight verification targets).


# Part 2 — Comparison with the technical note

## 2.1 What is verified

**Hyperparameters (Table 10)** — `max_depth 6`, `num_leaves 25`, `max_bin 32`,
`min_data_in_leaf 500`, `feature_fraction` noise 0.8 / muon 0.7,
`lambda_l1 2.0`, `lambda_l2 1.0`, `min_gain_to_split 2.0`,
`is_unbalanced False`.  Identical in `scripts/train_L4_classifier.py`.

**Cut values** — noise `> 0.7`, muon `> 0.65` (sec. 3.6.2, 3.6.3).  The same in
the code.

**Preprocessing** — "event weights manually re-scaled to balance the samples...
all weights re-scaled to be in range 0-1" (sec. 3.6.1).  Section 6 of the
notebook does exactly that.

**`micro_count`** — the full text of Table 11: *"Start with the **cleaned**
pulse series.  Look at pulses occurring within [-3.5 µs, +4 µs] from the trigger
time.  Slide a time window of 200 ns that maximizes the number of triggered DOMs
in it.  Get the number of triggered DOMs in that sliding time window."*

`MICROCOUNT_SUBKEY` is **exactly the same name**, and the window and width
match.  The counting semantics are right too: `I3TimeWindowCleaning
(TimeWindow=200)` finds the window maximising the DOM count, and `_micro_count`
then counts DOMs.

**Deviation found (fixed):** the chain started from the *uncleaned* series and
its only cleaning step (`I3SeededRTCleaning`) was bypassed -- its output
`L4_SRTTWPulses` was read nowhere.  Brought in line with the note: it now starts
from `cleaned_pulses` and the SeededRT block was removed (L3 has already applied
SRT cleaning).  Details: `CLAUDE.md` → "Booking/read audit", item 4.  L3's own
version differs: `STW9000_DTW300Hits` ([-4, +5] µs, 300 ns) -- so the two are
not fully correlated and the BDT extracts information from both.

**`accumulated_time`** — Table 12: *"Time to reach 75% of an event's charge in
the cleaned pulse series."*  Our code: `fraction=0.75`,
`pulses_key=cleaned_pulses`.  The fraction and the series were already
verified.  **The rest is now settled against pass2** (8144 events, both values
in the same frame -- `verification/README.md` §7):

| rule | agreement | median diff |
|---|---|---|
| all pulses, 0.75, the pulse **BEFORE** the crossing | **99.40%** | **0** |
| per-DOM charge, same rule | 54.25% | 0 |
| all pulses, fraction 0.70, at the crossing | 44.35% | 10 ns |
| all pulses, 0.75, **at** the crossing (what the note describes) | 0.98% | 52 ns |

**The reference time was never the problem** -- `t[idx] − t[0]` is right, and
the trigger-time hypothesis recorded here before is retired.  What pass2 does
is take the pulse *before* the cumulative charge reaches 75%, at which the
event holds LESS than 75% of its charge.  That is an off-by-one, not "the time
to reach 75%".

**We follow the note by default**, as for `micro_count`;
`--accumulated-time-pass2` reproduces pass2.  The two differ by ~52 ns on
values of hundreds to thousands of ns, so the choice is not cosmetic -- but
whichever is used, training and application must use the SAME one, because
`classifier.py` reads the column without knowing the convention.

**`first_hlc_rho`** — Table 12: *"Radial distance from string 36 (roughly the
center of DeepCore) of the first HLC hit."*  Our code embeds the string 36
coordinates (`46.29, −34.88`) and computes the same thing.

**The VICH speed window and veto DOM set** — sec. 3.4 (DeepCore Filter):

> *"the center-of-gravity (COG) of the hits inside the fiducial volume is
> calculated, then a relative velocity is derived between each hit of the veto
> region and the COG's position/time vertex.  If the derived speed of any veto
> hit is contained within **[0.25, 0.4] m/ns**, the hit is discarded on the
> basis that the veto hit could be causally related to a muon crossing the
> detector."*

**THIS WAS THE SOURCE OF `VICH_SPEED = (0.25, 0.40)`, AND IT IS THE WRONG
SOURCE.**  Read in full, sec. 3.4 (p.26-27, lines 483-493) describes the
**Level 2 DeepCore Filter** -- a different algorithm from L4's VICH.  Note the
verb: that filter **discards** hits inside the window.  Our `_vich` took the
same window as its **selection** rule.

The note's only description of `L4_VICH_nch` is Table 12's one line, *"Number
of triggered DOMs in the Veto Region that can be caused by muons"* -- no
window, no region, no reference point.  So nothing about our VICH is verified
by the note, and the pass2 cross-check agrees: five hypotheses have been
retired by measurement, all landing at ~12% agreement with the median
difference pinned at 2 DOMs (`verification/README.md` §8).

> The note also says outright (p.60, line 880) that *"some algorithms used
> throughout their event selection may use differing definitions of what
> precisely constitutes the veto region"*.  An earlier version of this file
> claimed L3's Table 7 definition "would have been wrong" for VICH; that was
> inferred from the same misread passage.  Table 7 was tested against pass2
> and does no better (11.4%), but it was never excluded on principle.

**The direction of `dt`** — the note does not state it, but the physics is
clear: a muon crossing the detector hits the veto region **first**, and the
fiducial COG forms afterwards.  Our code requires `dt = t_COG − t_hit > 0` --
the right direction.

---

## 2.2 The concrete deviation found: VICH's COG

**The note (sec. 3.4):** the COG is computed from the hits **inside the fiducial
volume** ("the COG of the hits **inside the fiducial volume**").

**Our code used to do:**

```python
cog = charge_weighted_cog(iter_hits(cln, geo))   # the WHOLE cleaned series
```

So veto-region hits contributed to the COG.  In events with a muon those hits
pull the COG **up and out** → the distance `d` to the veto DOM and `t_COG`
change → the computed speed shifts → the chance of landing in the
`[0.25, 0.4]` window changes.  The effect is systematic in muon events and small
in neutrino events -- i.e. **exactly in the direction that destroys the
separating power**.

**Fix (applied):** restrict the COG to the fiducial DOMs,
`deepcore_fiducial_domset("IC86")`.  `fiducial_cog=True` is now the default;
`False` restores the old behaviour for comparison.  Measured effect on a test
event: z moved from −400 to −43.

This may still not match the note's description *exactly* -- the note does not
say whether the COG is charge weighted, and we take it charge weighted -- but
the fiducial restriction is something the note states **explicitly**.

---

## 2.3 Remaining deviations

| Topic | The note | Ours | Effect |
|---|---|---|---|
| **Muon BDT background** | **real detector data** (99% muon at this stage), 18 runs from 2012-2017, balanced over the seasonal muon flux | pass2: the same 18 runs (`data` in productions.json), MuonGun only when data is not loaded; pass3: CORSIKA | The note says a classifier trained with MuonGun performed "similarly" but was not used.  The same can be expected of CORSIKA; but **no data/MC check is possible** and populations that are not simulated (muon bundles, say) cannot be learned. |
| **VICH pulse series** | the DC Filter does its own SRT cleaning on `SplitUncleanedInIcePulses` (every HLC hit is kept) | raw `SplitInIcePulses` | Direction unclear.  Ours sees more hits → VICH may come out slightly high.  The original pass2 code passes `InputPulses=uncleaned_pulses`, which matches ours. |
| **`FullTimeLengthRatio`** | Table 11 lists `IC2018_LE_L3_Vars.FullTimeLengthRatio` -- i.e. an **L3 variable** | we compute it at L4 (the pass3 L3 map has no ratio) | The components are in L3 (`CleanedFullTimeLength`, `UncleanedFullTimeLength` -- Table 8).  **Direction RESOLVED:** the x axis of this variable in Figure 13 runs 0.0-1.0 → the ratio is `cleaned/uncleaned`, the direction our code takes.  Verified against the L3 components: maximum deviation 0 over 143 events. |
| **HitStatistics key** | `SRTTWOfflinePulsesDCHitStatistics` (pass2) | `SRTTWSplitInIcePulsesDCHitStatistics` (pass3) | A known pass2→pass3 rename, confirmed with `reference/pass3_L3_process.py`. |
| **`accumulated_time` rule** | *"Time to reach 75% of an event's charge"* | DEFAULT: `CalculateVariables`' charge-quartile rule -- the last pulse whose cumulative charge has not passed 75% (99.84% identical to pass2) | `--accumulated-time-note` takes the pulse at the crossing, the note's reading (0.98% agreement). |
| **`VICH` everything** | one line, Table 12 | `I3CutL7Module`'s four (distance, dt) bands around the first trigger with config id 1010/1011 | **100.00% identical to pass2** (8144 events).  The speed-window version described above reproduced ~12% and is gone. |
| **ντ** | Table 13 has ντ CC (0.129 mHz) | pass2: NuTau 160511 is signal (samples are taken by role); pass3: no such set in the paths we have | ~3% of the signal at pass3. |
| **BDT engine** | LightGBM (sec. 3.6.1) | LightGBM | The same; the Table 10 parameters are used directly. |

---

## 2.4 The missing BDT input (fixed)

Table 12 lists **10** variables for the muon classifier and the text confirms it
(*"Table 12 lists the 10 input variables"*).  Our `MUON_FEATURES` had **9** --
`IC2018_LE_L3_Vars.NchCleaned` had been skipped.

`NchCleaned` is an input to both the noise and the muon BDT; being in the noise
list, it was overlooked.  The muon BDT would have been trained without one of
its documented inputs.  Added, in the Table 12 order:

```
ICVetoHits, RTVeto250Hits, NchCleaned, NAbove200Hits, VICH_nch,
accumulated_time, first_hlc_rho, cog_z, z_sigma, z_travel
```

Unique BDT variables: **14** (5 noise + 10 muon, `NchCleaned` shared).

---

## 2.5 Name mismatches — resolved

Verified with `check_registry` against real pass3 HDF5 output:

| Variable | The note (Table 11) | The actual HDF5 column |
|---|---|---|
| `fill_ratio` | `L4_fill_ratio.fill_ratio_from_mean` | **`fillratio_from_mean`** (no underscore) |
| `iLineFit_speed` | `L4_iLineFit.speed` | `L4_iLineFitParams.lf_vel` |
| `noise_weight` | — | `noise_weight.weight` (not `value`) |

All three are in `oscnext_l4.data.ALTS`; whichever is actually in the file is
used.  `found: 5/5` and `10/10` -- all 14 BDT inputs found.

**The HDF5 column name need not match the frame field name.** The hdfwriter
converter may rename (`fill_ratio_from_mean` → `fillratio_from_mean`).  That is
why `check_feature_map()` takes both sides' alternatives
(`data.ALTS` and `classifier.COLUMN_ALTS`) into account -- otherwise it would
mistake a legitimate name difference for a conflict and raise a false alarm.
It still catches a real conflict (regression tested: injecting
`cog_z` → `cog_x` was caught).

## 2.6 Weight verification targets (Table 13)

The L3 rates -- **the single best indicator** of the weight chain:

| Component | L3 [mHz] | after L4 noise | full L4 | efficiency |
|---|---|---|---|---|
| GENIE νe CC | 0.95 | 0.90 | 0.84 | 88.5 % |
| GENIE νμ CC | 3.77 | 3.65 | 3.11 | 82.5 % |
| GENIE ντ CC | 0.129 | 0.124 | 0.119 | 92.1 % |
| GENIE ν NC | 0.53 | 0.51 | 0.46 | 85.5 % |
| Atm. μ (MuonGun) | 505 | 490 | 28.1 | 5.6 % |
| Vuvuzela noise | 36.6 | 0.28 | 0.28 | 0.7 % |
| **Total MC** | **547** | 495 | 32.9 | 6.0 % |

These will not match exactly in pass3; the **order of magnitude** should.

> **This table exposed a bug.** The notebook divided the weights by
> `d["_n_files"]`, and that was the **number of HDF5 files**.  With 100 L3 files
> booked into one HDF5 the divisor became 1 → the weights were **100 times too
> large**.  You would have seen ~95 mHz for nue instead of ~0.95.
> `process_L4.py` now writes `n_l3_files` into `<output>.hdf5.meta.json` and
> `load_sample` reads it.  Older HDF5 files have no sidecar, so the notebook
> prints a warning.

> **Measured (see CLAUDE.md 5d):** the noise rate came out at 41.4 mHz for the
> full sample against the note's 36.6 -- agreement within 13%, so the
> `NOISE_NS_SCALE = 1e9` assumption holds.  The signal side is a factor 1.5
> high, which is what an invented E⁻³ power law should give.

---

## 2.7 The muon classifier's training sample — the note names the runs

Section 3.6.3 (p.39) is the only place any source we have states outright what
the muon BDT is trained on, and it settles three things at once.

**The background is detector data, not simulation, and it is data that has
already passed the noise cut.**  Quoted:

> For this classifier, detector data (**following L4 noise cut**) is used as
> the background sample used for training, as it is **99% muons at this stage**
> and is considered more robust than using MuonGun MC at this processing level.
> The signal training sample is **still GENIE MC**.  This also allows
> unsimulated event populations (such as muon bundles) to be removed by the
> classifier.  Note that a classifier was also trained using MuonGun MC for the
> background sample, which **achieved similar performance but was not used** in
> the sample.

Three consequences, in order of how easy they are to get wrong:

1. **The noise cut comes first.**  The background is data at
   `L4_NoiseClassifier_ProbNu > 0.7`; that cut is what makes it 99% muon.
   Train on data that has not been through it and the muon BDT spends part of
   its capacity re-learning noise rejection, on a population the noise BDT has
   already removed.  The note's own figures agree: Figures 19-24 all carry
   `L4_NoiseClassifier_ProbNu > 0.7` in their cut box.  This is CLAUDE.md open
   risk 5i, now confirmed from the note as well as from the fridge.
2. **The signal side stays GENIE MC.**  It is a data-vs-MC classifier by
   construction.
3. **MuonGun was tried and performed "similarly".**  So a MuonGun-trained model
   is not wrong, it is simply not the production's -- and it cannot learn the
   populations that are not simulated, which is the stated reason for
   preferring data.

**The 18 runs, listed outright** (p.39, and now the `data` sample of
`config/productions.json`):

| year | runs |
|---|---|
| 2012 | 120200, 120700, 121650 |
| 2013 | 122650, 123250, 124650 |
| 2014 | 125150, 125700, 126300 |
| 2015 | 126850, 127400, 127850 |
| 2016 | 128000, 128550, 129050 |
| 2017 | 129650, 130150, 130700 |

> ... selected to cover years roughly equally in order to avoid strong
> dependence of the muon rejection on the specific season due to muon flux
> seasonal variations.

**The even coverage is the point, not a detail.**  The atmospheric muon flux
varies seasonally with the temperature of the stratosphere, so a background
drawn from one part of the year teaches the classifier that season as much as
the muon.  Three runs per year over six years is the note's answer to that, and
a substitute list should keep the property rather than the count.

**This disagrees with the fridge, and the note wins for pass2.**
`L4_model_data.py` harvests *"one run per month 2012-2018"* -- a later, larger
list.  Where the two differ, the note is what the published Table 13 numbers
were produced with; the fridge script is the state of the code at a later date.

**What the muon classifier achieved** (p.46), i.e. what to measure ours
against: it rejects **over 94%** of atmospheric muons relative to the rates
before the L4 cuts, while keeping **over 80%** of all neutrinos -- best for
ν_τ CC at **92.1%**, worst for ν_μ CC at **82.5%**, which the note attributes
to the similarity between a muon from a ν_μ CC interaction and an atmospheric
one.  The resulting sample is roughly **100:10:1** muon:neutrino:noise.  The
per-channel numbers are the last column of Table 13 in 2.6 above.

Figure 21 is the muon classifier's **feature importance**, so our own gain plot
has a published counterpart to sit beside -- the same comparison the noise
model's does not have.


# Priority order (historical -- written before the pass2 cross-check)

Items 1, 3 and 4 are closed (CLAUDE.md open risks 1, 2 and
`verification/README.md`); item 2 is still open.

1. **`accumulated_time` reference time** (2.1) -- the fraction and the series
   are verified, the zero point is open.
2. **`fill_ratio`'s `SphericalRadiusMean`** -- never re-tuned for oscNext, and
   it carries 61% of the noise model's gain (CLAUDE.md 5g).  The single
   highest-value knob.
3. **Whether the VICH COG is charge weighted** (2.2) -- the fiducial
   restriction is done; this assumption remains.
4. **Compare against the reference L4 files** -- the only way to close items 1
   and 3 for good.
