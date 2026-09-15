# oscNext L4 — project briefing

This file is loaded automatically in every Claude Code session.  Its purpose:
to let a session that has never seen the codebase grasp the physics/IceTray
context and the current risks quickly enough to reason usefully about them.

## What this repository does

It rebuilds the **Level 3 → Level 4** processing step of the IceCube oscNext
(low energy neutrino) analysis.  It computes the L4 discriminating variables
from L3 `.i3` files, books them to HDF5, and trains two LightGBM classifiers:

- **noise**: rejection of pure noise (vuvuzela)
- **muon**: rejection of atmospheric muons (background = CORSIKA)

Reference: oscNext technical note v00.07 (sections 3.4-3.6, Tables 10-13).
**The method is the note's own:** LightGBM with the Table 10 hyperparameters.

> The project used pybdt (IceCube's AdaBoost-based BDT library) for a while --
> a deliberate deviation whose cost was then measured: on the same data, the
> same variables and the same split, AdaBoost kept 65.8% of the signal at 99%
> noise rejection where LightGBM kept 95.9% (the Table 13 target is ~96%).
> The pybdt path was removed for that reason; the measurement itself lives on
> the `claude/lightgbm-compare` branch.

## Why it was rewritten (critical context)

The IceTray meta-project in use (`py3-v4.4.2`) does **not** have:

- The `icecube.oscNext` project at all → `I3Classifier` (model application),
  `oscNext_cut` and `calc_rho_36` are unavailable.
- `icecube.hdfwriter` was missing **from the cvmfs metaproject** →
  `oscnext_l4/booker.py` falls back to pytables.  The user's own build
  (`/data/user/$USER/icetray_build/build`) DOES have hdfwriter -- a real run
  prints `Booking: icecube.hdfwriter`.  The fallback is not in use but the
  code stays (it is needed if we return to a cvmfs environment).
- The old project dependencies: `tau_bdt.I3CutL7Module` (VICH),
  `analysis.event_selection` (the Dunkman variables: accumulated_time,
  separation_in_cogs), `slc-veto` (QR box, optional).

So VICH, accumulated_time and separation_in_cogs were rewritten in pure Python
inside `oscnext_l4/variables.py`, **following the definitions in the technical
note**, and have **not been verified against the original C++ implementation**.
See "Open risks".

## File map and data flow

```
.i3 (L3 output, pass3)
   │
   ▼
scripts/process_L4.py  ──uses──►  oscnext_l4.variables (the oscNext_L4 segment)
   │                          ├─ common: first_hlc, rho_36, FullTimeLengthRatio
   │                          ├─ muon vars: ToI, iLineFit, VICH, accumulated_time
   │                          ├─ noise vars: micro_count, fill_ratio
   │                          └─ hit_statistics: cog_z, z_sigma, z_travel, n_hit_doms
   │
   ├─uses──►  oscnext_l4.booker (add_booker: hdfwriter if present, else SimpleBooker)
   │
   ▼
.hdf5  (L4_output/hdf5/<sample>/L4_*.hdf5)
   │
   ▼
notebooks/oscNext_L4.ipynb   ◄── THE INTERFACE (11 sections, 0-10)
   │
   ▼  section 6: L4_output/ds/L4_<tag>_dataset.npz
   │
scripts/train_L4_classifier.py  →  L4_<tag>_model.txt + .json + plots
   │
   ▼
oscnext_l4.classifier  (the L4Classifier tray module, add_L4_classifiers)
```

Supporting files:
- `oscnext_l4/env.py` — **every `icecube` import goes through here**; when an
  import fails it reports why.  `have_lightgbm()` lives here too.
- `setup_env.sh` — find the environment / open a shell / run one command /
  register a Jupyter kernel.
- `scripts/scan_files.py` — find corrupt `.i3.zst` files, write a healthy list.
- `oscnext_l4/runner.py` — the `process_L4.py` driver plus a live progress bar.
- `oscnext_l4/data.py` — `REGISTRY`/`ALTS`, `dump_tables`, `check_registry`,
  `check_feature_map`, `load_sample`, `add_weights`.
- `scripts/diagnose_env.py` — what is and is not in the environment.
- `docs/pipeline.md` — which file runs when.
- `docs/technical_note_comparison.md` — exactly what we write to HDF5, compared
  line by line with the technical note (Tables 7/10/11/12/13).

To dump the HDF5 columns, use section 2 of the notebook.

## The critical synchronisation point

`FEATURE_MAP` (`oscnext_l4/classifier.py`) and `REGISTRY`
(`oscnext_l4/data.py`) must agree line for line.  If one reads a different
column than the other, the model produces wrong predictions **silently** -- it
does not raise.

This is checked in code, not by eye: `data.check_feature_map()` reads
`FEATURE_MAP` **with AST** (no icetray needed) and lists the conflicts.  It is
normal for FEATURE_MAP to hold extra candidate variables; what is dangerous is
**the same name pointing at a different column**.

The first run caught a real conflict: `iLineFit_speed` was `LFVel` in REGISTRY,
`lf_vel` in FEATURE_MAP, and `L4_iLineFit.speed` in Table 11 of the note.  All
three are in `data.ALTS` -- whichever is **actually in the file** is used.

The model format is deliberately not joblib/pickle but LightGBM's native text
format (`.txt`) plus a JSON sidecar: the IceTray environment has no
sklearn/joblib, only `lightgbm` + `numpy`.

## Current status

- [x] Environment verified (no `oscNext` project; no `slc-veto`; the user's own
      build DOES have `hdfwriter`)
- [x] IceTray/lightgbm import layer (`oscnext_l4/env.py` + `setup_env.sh`)
- [x] Robust against corrupt input files (`--scan` + `--retries`)
- [x] nue processed (100 files → 256,799 events), CORSIKA (500 files → 6,462)
- [x] Column names pinned down (14/14 BDT inputs found)
- [x] Noise classifier trained: 95.9% efficiency at 99% rejection, train/test
      gap +0.1 points
- [ ] numu / noise need reprocessing -- cut short by corrupt `.i3.zst`
- [ ] Muon classifier not trained
- [ ] The rewritten variables (VICH, accumulated_time) not verified
- [ ] Noise MC statistics inadequate (~2,000 events) -- nothing right of 99%
      rejection is measurable

## Open risks / places worth thinking about

> The technical note has been read and compared with the code line by line:
> **`docs/technical_note_comparison.md`**.

1. **VICH — NOT verified, and the justification was wrong.**  This was
   recorded as "largely verified" because sec. 3.4 gives a speed window of
   [0.25, 0.4] m/ns.  Reading that section in full (p.26-27, lines 483-493)
   shows it describes the **Level 2 DeepCore Filter**, a different algorithm:
   that filter SRT-cleans SplitUncleanedInIcePulses, splits it into fiducial
   and veto series, takes the COG of the fiducial hits, derives a speed
   between each veto hit and that vertex, and DISCARDS hits in the window.
   Our `_vich` borrowed that window as its SELECTION rule.  The note's only
   description of `L4_VICH_nch` is Table 12's one line -- no window, no
   region, no reference point.

   The pass2 cross-check agrees that it is wrong and has retired five
   hypotheses (8144 events; best agreement in brackets):

   - the speed window: a 36-cell sweep of both edges peaks at **18.0%**, and
     the median difference stays pinned at **2 DOMs in every cell**;
   - the veto region: pass2's count exceeds the DOMs hit in our region in only
     **0.16%** of events, so the region is big enough, and Table 7's L3 region
     does no better (11.4%);
   - the COG: unweighted (12.0%) and non-fiducial (11.9%) are the same as ours
     (12.0%);
   - the reference point: the verified first HLC hit instead of the COG, 11.7%;
   - the input series: the cleaned series instead of the uncleaned one, 2.8%.

   Everything lands at ~12% with a median difference of 2, which is what you
   see when the swept parameter is not the one that differs: **the algorithm is
   structurally different, not a parameter away.**  Note p.60 (line 880) warns
   outright that "some algorithms used throughout their event selection may use
   differing definitions of what precisely constitutes the veto region".

   **The fastest resolution is the `tau_bdt` source** (`I3CutL7Module`), which
   is not in this meta-project.  The direction `dt = t_COG − t_hit > 0` is
   consistent with the physics, but that is all that can be said for the
   current implementation.
   **Deviation found (fixed):** the note says the COG is computed from the hits
   *in the fiducial volume*; the code used the whole cleaned series.  In events
   with a muon the veto hits pulled the COG upward (test: z −400 → −43).
   `fiducial_cog=True` is now the default; `False` restores the old behaviour.
   **Still open:** is the COG charge weighted?  The note does not say; we take
   it charge weighted.  **The pulse series IS verified:** the original pass2
   code (`reference/oscNext_L4_pass2_original.py`) passes
   `InputPulses=uncleaned_pulses  # Use uncleaned hits` to `I3CutL7Module`, so
   our use of the raw `SplitInIcePulses` matches the original.

2. **accumulated_time — SOLVED against pass2 (99.40%).**  Table 12 says "Time
   to reach 75% of an event's charge in the cleaned pulse series"; the fraction
   and the series were already verified (the original passes
   `PulseSeries=cleaned_pulses` to Dunkman's `CalculateVariables`).  What was
   open was the exact rule, and the pass2 cross-check settled it on 8144 events
   with both values in the same frame:

   | rule | agreement |
   |---|---|
   | all pulses, 0.75, the pulse **BEFORE** the crossing | **99.40%** (median diff 0) |
   | per-DOM charge, same rule | 54.25% |
   | all pulses, fraction 0.70, at the crossing | 44.35% |
   | all pulses, 0.75, **at** the crossing (what the note describes) | 0.98% |

   So **pass2 has an off-by-one**: at the pulse it reports, the event holds
   LESS than 75% of its charge, which is not "the time to reach 75%".
   The zero point was never the problem -- `t[idx] − t[0]` is right.

   It was found by INVERTING rather than guessing: the charge fraction
   accumulated at pass2's stored time sits just below 0.75 in at least 84% of
   events (median 0.7258, 84th pct 0.7449) and never above, i.e. exactly one
   entry's charge share short.  The median shortfall 0.024 is an average
   pulse's share at this hit multiplicity, so the size matched too.

   **Decision: the DEFAULT follows the note** (at the crossing), as it does for
   micro_count (open risk 5b) -- we follow the description, not the original's
   slip.  `--accumulated-time-pass2` / `oscNext_L4(accumulated_time_pass2=True)`
   reproduces pass2.  The two code paths were checked against each other on
   4000 random events: 0 mismatches.  The production sort was also made stable,
   so tied pulse times give a reproducible index.
   `separation_in_cogs` is not a BDT input -- low priority.

3. **FullTimeLengthRatio direction — RESOLVED.** The text of Table 11 does not
   give the direction of the ratio, but **Figure 13** does: the x axis of the
   `IC2018_LE_L3_Vars.FullTimeLengthRatio` distribution runs **0.0 – 1.0**.
   The other direction (uncleaned/cleaned) would be ≥ 1 and would not fit that
   axis.  So it is `cleaned / uncleaned` -- the direction the code
   (`_full_time_length_ratio`) takes.  No longer an inference; it is in the note.
   **Remaining (small) difference:** the note lists it as an L3 variable
   (`IC2018_LE_L3_Vars.FullTimeLengthRatio`); the pass3 L3 map has no ratio,
   only its components (`CleanedFullTimeLength`, `UncleanedFullTimeLength`), so
   we produce the ratio at L4 by dividing.  The value should be identical.
   **Verified:** the division was compared against the L3 components, maximum
   deviation 0 over 143 events.
   **The physics explanation was CORRECTED (by measurement).** The docstring
   said "real event ~1, noise ~0"; the actual measurement (126 nue + 17 noise):

   |  | ratio (median) | cleaned | uncleaned |
   |---|---|---|---|
   | nue | 0.16 | 1626 ns | 10,100 ns |
   | noise (passing L3) | 0.27 | 2780 ns | 10,290 ns |

   The ratio **never approaches 1**: `SplitInIcePulses` spans the whole readout
   window, so the uncleaned duration is ~10 µs in every event and the variable
   is effectively "cleaned duration / 10 µs".  The separating direction is also
   **inverted**: the cleaned series of noise is longer than that of nue.  It
   does separate, but not in the expected way.  (17 noise events is few; the
   ordering is what this file shows, while the magnitude finding is structural.)
   An `inf`/`NaN` guard was also added: the `uncleaned <= 0` check prevented
   division by zero but **let a NaN denominator through** (`NaN <= 0` → False),
   so NaN was written silently.  The result is now checked with `np.isfinite`
   and a ratio > 1 warns once.

3a. **Missing muon BDT input (fixed).** Table 12 lists **10** variables;
   `MUON_FEATURES` had **9** -- `NchCleaned` had been skipped (it was in the
   noise list, so it was overlooked).  Added.  Unique BDT variables: 14
   (5 noise + 10 muon, `NchCleaned` shared).

3b. **Column names — RESOLVED.** Verified with `check_registry` against real
   pass3 output: `fill_ratio` → **`fillratio_from_mean`** (no underscore),
   `iLineFit_speed` → `lf_vel`, `noise_weight` → `weight`.  All three are in
   `ALTS`.  `found: 5/5` and `10/10` -- all 14 BDT inputs found.
   The HDF5 column name need not match the frame field name (the hdfwriter
   converter renames); `check_feature_map()` takes both sides' alternatives
   into account.

3c. **The `noise_weight` column (fixed).** `AUX` said
   `("noise_weight", "value")`; the version verified in the earlier LightGBM
   notebook is `("noise_weight", "weight")`.  Being wrong, the noise weight
   stayed **entirely NaN** → the noise sample's `w_phys` would have been zero.
   Fixed; both are in `ALTS`.

3d. **CORSIKA weighting (fixed).** The `corsika_weight` on the pybdt path gave
   every event an **equal** weight (`np.ones/n_files`) -- i.e. there was no
   weighting at all.  It did not merely break the absolute rate: the muon BDT
   saw the simulation's flat spectrum rather than the atmospheric one, so
   **the shape of the training set was wrong**.  The method from the earlier
   LightGBM notebook was carried over: `simweights` + `GaisserH3a`, else
   `CorsikaWeightMap.Weight / (NEvents × OverSampling)`.  Matching goes through
   `Run/Event/SubEvent` (row order is not trusted).
   **The normalisation deliberately differs from the old code:** that code
   passed `nfiles=1` per HDF5 and divided by the HDF5 count at the end; our
   parts hold several L3 files (`--chunk-files`), so each part's own
   `n_l3_files` is passed as `nfiles` and there is no further division.

4. **The weight chain** (`PropagateGenieInfo`, `process_L4.py` MC_KEYS /
   NOISE_MC_KEYS / CORSIKA_KEYS) -- when pass3 has no I3GenieInfo it falls back
   to NEvents × fraction.  How often that triggers, and how much it shifts the
   result, has not been measured.

5. **fill_ratio — RESOLVED (from the original code).** The text of Table 11
   leaves the vertex **blank**: *"about some vertex (details here)"*, an
   unfilled cross-reference, so it could not be derived from the note.  The
   original pass2 code says it: the main segment passes
   `fill_ratio_vertex=L4_FIRST_HLC_KEY` to `oscNext_L4_noise_cut_variables` --
   the position of the first HLC hit, exactly what we pass.
   `RecoPulseName=cleaned_pulses` and `SphericalRadiusMean=1.6` match verbatim
   too, and the original's own comment reads *"Was optimised for GRECO but has
   not been re-optimised for oscNext"* -- so 1.6 being untuned for oscNext is a
   **known** situation, not an omission of ours.  It can be re-optimised with a
   feature importance / incremental scan (the original suggests as much).

5a. **iLineFit_speed — RESOLVED (from the original code).** Table 11 only says
   *"Speed fitted by the improved LineFit algorithm"* and gives no parameters.
   The original pass2 code is our call verbatim:
   `tray.AddSegment(linefit.simple, ..., inputResponse=cleaned_pulses,
   fitName=L4_LINEFIT_KEY)`.  So "improved LineFit" = `linefit.simple`, no
   extra parameters.  The x axis of Figure 13 is logarithmic, 10⁻³ – 10³ (m/ns).

5b. **micro_count: the original code and the note CONTRADICT -- DECISION: the
   note.**  The original pass2 code starts the chain from `uncleaned_pulses`
   (`I3StaticTWC(InputResponse=uncleaned_pulses)`), while Table 11 says
   *"Start with the cleaned pulse series"*.  **We follow the note** (the
   default, a permanent decision).
   **Measured -- the difference is almost nil:** the two chains were computed on
   the same events.  115 of 126 nue events and 16 of 17 noise events are
   **exactly equal**; the medians agree (5 and 3).  The reason: the closing
   200 ns window is what decides, and since noise hits are spread over ~10 µs
   the densest 200 ns window catches the same hits in either series.  That is
   also why the dead SeededRT code in pass2 went unnoticed for years.
   So the fix is **right but small**; the original assessment ("the
   classifier's separating power was seriously degraded") was overstated.
   Both can be produced for comparison: `process_L4.py --micro-count-uncleaned`,
   `run_all(extra_args=["--micro-count-uncleaned"])` from the notebook, or
   `oscNext_L4(micro_count_uncleaned=True)` directly.  `fill_ratio` uses the
   cleaned series either way (as the original does).
   Details in "Booking/read audit", item 4.
   **Verified from the original:** the DeepCore fiducial `I3OMSelection` step
   (absent from the note but present in the original -- and in ours),
   `TriggerConfigIDs=[1010, 1011]`, `WindowMinus/Plus = 3500/4000`, `dtw = 200`,
   the subkey name `STW_m%ip%i_DTW%i`, and that the count is of **DOMs**
   (`len(reco_pulse_series.values())` -- ours is `len(pmap)`, the same thing).

5c. **RESOLVED — the signal train/test split differed between the two
   classifiers.**  Section 6 of the notebook used to call `make_datasets`
   separately for the noise and muon BDT, drawing `istrain = rng.random(...)`
   **afresh** each time.  Both use the same nue+numu signal events, so one event
   could be in the noise BDT's *training* set and the muon BDT's *test* set.
   That is harmless for the individual models, but the L4 cut is their
   **conjunction** (originally `noise ≥ 0.7 AND muon ≥ 0.65`) and there was no
   common held-out set on which to evaluate it → the final efficiency looked
   better than it was.  **Fix: draw the signal split once and share it.**
   **Done:** section 6 of `notebooks/oscNext_L4.ipynb` stacks the signal once
   and draws `SIG_ISTRAIN` once; both `.npz` files carry the same split.  The
   CORSIKA leak was closed in the same place: `split_by_shower` splits the
   background by **shower** (`Run`) rather than by event -- since one air shower
   is repeated `OverSampling` times, an event-level split spread its copies
   across train and test and inflated the muon BDT's test efficiency.

5d. **`NOISE_NS_SCALE = 1e9` — VERIFIED (by measurement).** The unit of the
   vuvuzela `noise_weight` was assumed to be 1/ns in pass3.  In the first real
   run the test set's total noise rate came out at **20.5 mHz**; since the test
   set is 1,016 of 2,051 events (49.5%), the full sample is **41.4 mHz** --
   **within 13%** of the **36.6 mHz** in Table 13 of the note.  The assumption
   holds.  Signal side: 3.99 mHz (test) → 7.97 mHz (full), against the note's
   nue CC + numu CC + NC total of 5.25 mHz -- a factor **1.5**, which is what
   an invented E⁻³ power law should give.  The noise weights are also **exactly
   uniform** (0.10% per event = 1/1016): expected, because vuvuzela is generated
   over a fixed livetime, not an error.  Consequence: for noise the weighted and
   unweighted efficiency/rejection are **identical at every cut**.

5e. **Noise MC statistics — MEASURED, INADEQUATE.** 100 L3 files give a total
   of **2,051** events (train 1,035 / test 1,016) against 329,545 signal.  A
   ratio of **159:1**.  Consistent with the measured 8% L3 pass rate (84% for
   nue).  The reference's statistics must be far larger: Table 10 gives
   `min data in leaf = 500` for the noise BDT, which with our 1,035 training
   events would mean 2 leaves.

5f. **LightGBM measured — the bottleneck was the engine, not the background
   statistics.**  Same `.ds`, same 5 variables, same train/test split, same
   weights, same measurement code.  Test set, unweighted event counts:

   | rejection | best pybdt/AdaBoost | LightGBM |
   |---|---|---|
   | 90% | 94.0 | **99.0** |
   | 95% | 91.7 | **98.5** |
   | 99% | 65.8 | **95.9** |

   The Table 13 target (~96% at 99.2% rejection) was met -- and met while
   handicapped by `min_data_in_leaf=500` (828 background events fitted) and
   stopping early at 122 trees.  No overtraining: the train/test efficiency gap
   is **+0.1 points**.  The `w_phys` cross-check agrees (95.2% / 98.92%).
   **Confidence limit:** the 99% row rests on 10 background events, right at
   the edge of what is measurable; the 95% row, with 50, is firmer.
   An earlier conclusion held that "the bottleneck is background statistics,
   not model capacity" -- this measurement **retracted** it.  1,035 background
   events are enough for LightGBM.  More vuvuzela MC is still valuable (to
   measure right of 99%) but is **not blocking**.

5g. **`fill_ratio` dominates — this enlarges item 5.** The LightGBM gain
   distribution: `fill_ratio` **60.9%**, `NchCleaned` 26.0%,
   `FullTimeLengthRatio` 8.1%, `micro_count` 3.9%, `iLineFit_speed` **1.1%**
   (nearly dead).  The variable the model leans on most is the one that depends
   on `SphericalRadiusMean=1.6`, which the original code's own comment calls
   *"optimised for GRECO but not re-optimised for oscNext"*.  So the "could be
   re-optimised" note in item 5 is no longer a curiosity: it is **the single
   highest-value knob**.

5h. **The overtraining criterion is the efficiency gap, NOT KS.** `p_KS` was
   measured on this data and found useless: it stamped the best model
   "overtrained" (p=0.002) and the worst one "clean" (p=0.98).  The reason: KS
   compares the train/test **score distributions**, and with 165k signal events
   it catches differences that are statistically significant and physically
   irrelevant.  `train_L4_classifier.py` prints **the efficiency on train and
   on test at the same rejection, and their difference** instead (`--gap-at`,
   default 90% rejection -- at 99% only ~10 events sit above the cut and the
   gap is mostly noise).

6. **No ντ and no real detector data** — the signal is defined as νe+νμ
   (ντ CC is ~3%), and the muon BDT's background is CORSIKA rather than real
   data.  How much that substitution shifts things will become clear once the
   data/MC agreement check is in place.

## Booking/read audit — the bugs found

The code was reviewed end to end; three of these produced **silent data
corruption**.

**1. `__I3Index__` was overwriting the real data (fixed).**
`h5.walk_nodes("/", "Table")` descends into subgroups too.  hdfwriter writes
`/X` (data) and `/__I3Index__/X` (index) for every key; keyed by leaf name the
index overwrote the data → every table appeared to have `start/stop` columns
and **every variable came out NaN**.  `_table_nodes()` now skips everything
under `__I3Index__`.

**2. Repeated `Run/Event/SubEvent` matched wrongly (fixed).**
When a table was not aligned with `I3EventHeader` (because the key is absent in
some frames), matching went through a `(Run, Event, SubEvent)` dictionary.  That
triple is **not unique**: in MC `run_id` is the set number, `event_id` restarts
from zero in every L3 file, and `--chunk-files` puts 10 files in one part.  The
last entry won in the dictionary → events either stayed NaN or picked up
**another event's value**.
Fix: matching now goes through hdfwriter's `/__I3Index__/<key>` table
(`exists`/`start`).  When there is no index and the triples repeat, NaN is left
and reported rather than matching wrongly in silence.

**3. `n_flux_events` was not updated per file (fixed).**
`PropagateGenieInfo` read the value once and froze it.  One tray processes 10 L3
files with `--chunk-files` and **each file has its own `I3GenieInfo`** → the
first file's value was applied to all of them and the later files' weights came
out wrong.  It is now updated at every `I3GenieInfo`, and a change is logged.

**4. `micro_count` was counted without noise cleaning (fixed).**
The chain was **documented** as `uncleaned_pulses` → `I3StaticTWC` →
`I3SeededRTCleaning` → `I3OMSelection` → `I3TimeWindowCleaning` → count, but
`I3OMSelection` took the StaticTWC output (`L4_TWPulses`) as its input, not the
SeededRT output (`L4_SRTTWPulses`).  `L4_SRTTWPulses` was written into the frame
and **read nowhere** (verified by grep) -- so the only noise-cleaning step in
the chain was effectively disabled.

**This bug is not ours; it is inherited from the original pass2 code** -- found
afterwards: the same lines are in `reference/oscNext_L4_pass2_original.py`, and
even the author's own comment is doubtful:
`srt_tw_pulses = "L4_SRTTWPulses"  #TODO Is this actually used?`.  So **the
pass2 numbers were produced without noise cleaning too** (the micro_count panel
of Figure 12 is in that state).  Our fix follows **the note**, not the original
code -- a deliberate deviation.

Consequence: `micro_count` was counted over raw hits.  None of the remaining
steps removes noise (StaticTWC is a wide time slice, OMSelection a spatial cut,
TimeWindowCleaning the densest 200 ns window).

**The effect was measured afterwards and came out SMALL** (see open risk 5b):
the two chains give exactly the same count in 91% of nue and 94% of noise
events.  The 200 ns window already removes the noise in practice.  The fix is
right for conformity with the note, but the "separating power was degraded"
phrasing originally written here is not supported by measurement -- it was
overstated.

The technical note (Table 11) says *"Start with the **cleaned** pulse series"*.
The fix follows it: the chain now starts from `cleaned_pulses`
(`SRTTWSplitInIcePulsesDC`) and the SeededRT block was removed -- L3 has already
applied SRT cleaning to that series (that is the "SRT" in its name), so doing it
again would be double cleaning.  The new chain matches the note's four steps
exactly: `cleaned → StaticTWC [-3500,+4000] ns → DeepCore fiducial → 200 ns DTW
→ count`.

`oscNext_L4_noise_cut_variables` no longer takes an `uncleaned_pulses`
parameter, and the segment no longer depends on STTools.

**This fix invalidates the existing HDF5 files** -- nue and CORSIKA must be
reprocessed (numu/noise were going to be anyway).

**Minor:** `n_l3_files` was misleading in smoke output produced with `--n` (the
tray stops early and the list is not read in full).  It now writes `None` plus
`n_l3_files_unreliable`, and `load_sample` does not use it as a divisor.
The `DAQ()`/`Simulation()` methods in `PropagateGenieInfo` were dead code (they
are never called once `Process()` is overridden) -- removed.

## Reference documents (`reference/`)

- `reference/OscNext_v00.074_pass2_technical_note.pdf` — the official oscNext
  technical note for pass2 (83 pages).  **Not for pass3**, but most of the L4
  logic (variable definitions, BDT hyperparameters) is taken to be the same in
  pass3.
- `reference/oscNext_L4_pass2_original.py` — **the original oscNext L4 tray
  segment** (Tom Stuttard, pass2).  The whole body is commented out
  (`#TODO migrate` -- it was never ported to the GitHub IceTray), but the
  parameter values and module chains are a **first-hand source**.  Our
  `oscnext_l4/variables.py` is a rewrite of it.  What it verifies:
  `fill_ratio_vertex=L4_FIRST_HLC_KEY`, `SphericalRadiusMean=1.6`,
  `linefit.simple`, VICH's use of `uncleaned_pulses`, Dunkman's use of
  `cleaned_pulses`, the micro_count parameters, the straight-cut thresholds, and
  the L4 cut thresholds (noise ProbNu ≥ 0.7, muon ProbNu ≥ 0.65).
  The one contradiction: the micro_count chain starting from
  `uncleaned_pulses` (see open risk 5b).
- `reference/pass3_L3_process.py` — the user's **actual pass3 L3 processing
  script** (it uses GRECO `grecovariables.DeepCoreCleaning`/`DeepCoreCuts`).
  Compared against the L3 output `oscnext_l4/variables.py` assumes, and
  **verified**:
  - `SRTTWSplitInIcePulsesDC` is exactly `CLEANED_PULSES_DEFAULT`
  - `L3_oscNext_bool = IC2018_LE_L3_bools["IC2018_LE_L3_Full"] AND
    Data_quality_bool` is exactly the logic of the `l3_cut` function
  - the SLOP filter / LID errata data-quality cut also matches the assumption
    in the code comment

  **Careful:** this script never mentions `IC2018_LE_L3_Vars` (it neither
  writes nor reads it) -- that `DeepCoreCuts` also produces it was an
  *inference*.  It was confirmed by the real file dump below.

## What is actually in an L3 file (verified)

The Physics frame of `genie_NuE_IC86.023800.000000.i3.zst` was dumped, so this
is **verified**, not assumed:

- **`IC2018_LE_L3_Vars`** (`I3MapStringDouble`) **EXISTS**, 14 columns:
  `C2HR6`, `CausalVetoHits`, `CleanedFullTimeLength`, `DCFiducialHits`,
  `ICVetoHits`, `NAbove200Hits`, `NchCleaned`, `NoiseEngine`, `RTVeto250Hits`,
  `RTVetoCutHit`, `STW9000_DTW300Hits`, `UncleanedFullTimeLength`,
  `VertexGuessZ`, `VetoFiducialRatioHits`.
  Every column we read from L3 in `FEATURE_MAP`/`REGISTRY` is in that list.
- `IC2018_LE_L3_bools` (`I3MapStringBool`): `IC2018_LE_L3_Full`, `..._No_Nch`,
  `..._No_Nch_No_RTVeto`, `..._No_RTVeto`, `..._No_UncleanedTime`
- `SRTTWSplitInIcePulsesDC` and `SplitInIcePulses` both exist -- both are
  `I3RecoPulseSeriesMapMask`, and `get_pulses()` unpacks them with
  `.apply(frame)`
- `L3_oscNext_bool`, `Data_quality_bool` exist
- `I3GenieInfo` exists → we should not be falling back to `NEvents × 0.7/0.3`
- `MCInIcePrimary` is **ABSENT** → truth information must come from
  `I3MCWeightDict`
- HitStatistics / HitMultiplicity are **ABSENT** → L3 deletes them, so our
  recomputation (`oscNext_L4_hit_statistics`) is necessary

**Two traps:**
1. `I3GenieResult` cannot be deserialised: *"Attempting to read version 2 from
   file but running version 1 of I3GenieResult"* -- the file is newer than the
   installed `simclasses`.  Harmless for now because we do not book that key,
   but a job could die if `--output-i3` is used.
2. The frame also contains `pole_grecofilter_onlineLowEnL3_Vars` -- a separate
   map belonging to the *online* filter, not to be confused with
   `IC2018_LE_L3_Vars`.

## Environment setup — known traps

**1. `env-shell.sh` opens a new shell.** Writing `./env-shell.sh` and
`python ...` on consecutive lines of a script runs the second line *after*
leaving that shell, i.e. without the environment.  This is the number one cause
of "icetray will not import".  For a single command:
`./env-shell.sh -- python ...` or `./setup_env.sh run python ...`.

**2a. `cannot import name '...' from 'oscnext_l4.data'` — the module cache.**
Python keeps a module once imported; a `git pull` updates the file but
`from oscnext_l4.data import new_function` still looks at the old module object
and raises ImportError.  To tell them apart:
```python
from oscnext_l4 import data
print("in the file  :", "def aux_for" in open(data.__file__).read())
print("in memory    :", hasattr(data, "aux_for"))
```
`file True, memory False` → the cache; **Kernel → Restart**.
Both False → `git pull` was not run.
Section 0 of the notebook enables `%autoreload 2`, so this should not recur.

**2b. Jupyter's WORKING DIRECTORY does not change with a kernel restart.**
Symptom: a subprocess dies with `returncode=2` and the path points somewhere
like `~/.local/share/Trash/files/...`.  Cause: the repository directory was
deleted or moved but the Jupyter server is still running in the old one, so
relative paths resolve there.

A kernel restart is NOT enough -- the working directory comes from **the
server**.  Stop it and restart from the right directory:
```bash
cd ~/l4/osncnextl4 && python -m jupyter lab --no-browser --port=8896
```
Check in the notebook: `import os; os.getcwd()`.  Section 0 locates the
repository root itself and reports clearly when it cannot; `OSCNEXT_L4_ROOT`
overrides the search.  `configure_runner` also catches this at the start.

> Deleting the directory from a file manager or Jupyter's delete button is not
> `rm` -- it **moves it to the trash** (`~/.local/share/Trash/files/`).  The
> files stay there and keep consuming the home quota.  Recover what you need
> from there, then remove it for real with `rm -rf`.

**2. The Jupyter kernel.** The only way the notebook sees IceTray is for the
kernel to be the python inside env-shell.  Starting Jupyter from inside the
environment is cleanest (README step 3); otherwise register the kernel with
`./setup_env.sh kernel` and select it in the notebook.

**3. `lightgbm` appears to be missing in the kernel.** It is not in the
`icecube` namespace -- an ordinary pip package that must be installed in the
env-shell python.

**4. The location of `I3Tray` depends on the version** —
`icecube.icetray.I3Tray` (v1.5+) versus a top-level `I3Tray` (combo).
`env.get_I3Tray()` tries both.

**5. One missing project used to lock the whole repository.**
`oscnext_l4/variables.py` used to import `tensor_of_inertia`, `fill_ratio` and
`DeepCore_Filter` at module level; if one was absent the file could not be
imported at all.  It now uses `optional_project()`, and a missing project
raises a clear error when the segment that produces its variable is called
(`require_project(...)`).

Build search order (`setup_env.sh` and `env.find_env_shells()`):
`$OSCNEXT_I3_BUILD` → `$I3_BUILD` → `/data/user/$USER/icetray_build/build` →
`/data/user/$USER/*/build` → `~/*/build` → cvmfs metaprojects.

## Corrupt input files (solved)

The pass3 production contains half-written `.i3.zst` files:

```
FATAL (I3Reader): Error reading .../genie_NuMu_IC86.023799.000046.i3.zst
                  at frame 4: input stream error!
```

`I3Reader` takes the whole file list at once, so one corrupt file **kills the
entire tray**.  That is why the numu (100 files) and noise (100 files) jobs were
cut short.  The `Table ... is still connected ... This is a BUG!` message that
follows is a **consequence**, not a separate bug -- it means the HDF5 was not
closed properly when the tray died.  Those partial HDF5 files cannot be opened
and should be deleted.

`process_L4.py` protects in two layers:
1. **Pre-scan** (`--scan quick`, the default) -- the first 25 frames of each
   file are read and the ones that do not open are dropped.
2. **Run time** (`--retries 3`, the default) -- if the tray still dies, the file
   name is extracted from the error message with a regex, blacklisted, the
   partial HDF5 is removed, and the run is retried without that file.  (The tray
   was moved into a factory function: an `I3Tray` cannot be `Execute`d twice.)

Dropped files are written to `<output>.hdf5.badfiles.txt`.  The efficient route
is to scan once per set with `scan_files.py --good-list` and then use
`--input-list ... --scan off`.

## Speeding up HDF5 production

Measured: nue 100 files / 1019 s (~10 s/file), CORSIKA 500 files / 2578 s.
Single process, single core.

**Measure first, optimise second:**
```bash
python scripts/process_L4.py --usage --n 2000 --scan off --input <one_file> ...
```
IceTray prints per-module CPU time.  Do not try to optimise before seeing which
module is slow.

**Speedups are NOT automatic** -- they have to be asked for.  The default
`run_all()` is still a single process.

**1. Parallelism (the biggest win).** Split the input files across N processes:
```python
run_all(jobs=8, chunk_files=10)                       # every sample
run_process_parallel("numu", jobs=8, chunk_files=10)  # one sample
```
Separate processes, separate outputs (`L4_nue_job0_part000.hdf5`, ...), no
shared state → almost linear speedup.  They all match the `L4_nue*.hdf5` glob
and each part writes its own `.meta.json`, so `n_l3_files` sums correctly.
**cobalt is a shared machine** -- `jobs=8` is reasonable, `jobs=64` is not.

**2. Skip the unnecessary computation.** `I3TensorOfInertia` (`L4_ToI`) and
`separation_in_cogs` are in neither Table 11 nor Table 12, i.e. they are not BDT
inputs; they are off by default and `--run-optional` turns them on.

**3. Scan once.** Scan each set once with `scan_files.py --good-list`, then use
`--input-list ... --scan off`.  `run_process_parallel` already uses `--scan off`.

**4. Book fewer keys.** 33 keys are written; dropping large maps such as
`I3GenieSystWeightDict` from `process_L4.build_key_list` speeds up writing and
shrinks the files.

## `--n` counts frames, not events

`process_L4.py --n N` → `tray.Execute(N)` → **N frames** are processed.  A frame
is not an event: the stream carries G/C/D (GCD), Q (DAQ) and P (Physics) frames,
and one DAQ event can produce several P frames (sub-events).  On top of that,
only some P frames match `--sub-event-stream` and only some pass the L3 cut.

So 200 frames booking 60 events is normal.  The output shows each stage:

```
Physics frames          : 98
  InIceSplit            : 98  (100.0%)
  after the L3 cut      : 60  (61.2%)
Events booked           : 60
```

Wherever the loss happens, it is visible here: an `InIceSplit` line at 0 means
`--sub-event-stream` is wrong; an L3 line at 0 means the input is not L3 output,
or `Data_quality_bool` is removing everything (use `--no-l3-cut` to test).

When `--n` is given the tray stops early, so **the file list is not read in
full**.  For that reason the "files processed" count is not printed in `--n`
mode (it would mislead: the list may hold 100 files while the tray stopped in
the first).  For a smoke test, pass a single file or use `--scan off` --
scanning 100 files is pointless.

## Running on pass2 (the cross-check)

The rewritten variables are validated against the real pass2 L4 files: our own
L4 production is run over the pass2 **L3** files and compared, event by event,
with the pass2 **L4** files, which already contain every variable.  The "ours"
side is an ordinary `process_L4.py` output, so the same HDF5 is also the
training input for a pass2-trained model.  `scripts/compare_pass2.py`
(`inspect` / `plan` / `book` / `report`) and `oscnext_l4/pass2.py`.

**The whole delta between a pass3 run and a pass2 run is ONE FLAG:**

```
--cleaned-pulses SRTTWOfflinePulsesDC
```

Everything else adapts by itself.  What was checked, and why nothing more is
needed:

- **The pulse series.**  Technical note sec. 3.2 (p.25) defines both outright:
  uncleaned is `SplitInIcePulses` -- *the same string as pass3*, so the default
  stands -- and cleaned is `SRTTWOfflinePulsesDC` (pass3:
  `SRTTWSplitInIcePulsesDC`).  Only the cleaned name is overridden.
  The original pass2 script cannot settle this: `uncleaned_pulses` and
  `cleaned_pulses` are parameters there with no defaults, supplied by a
  production script we do not have.  Its straight-cut block does mention
  `SRTTWOfflinePulsesDCHitStatistics`, but that block is stale (it also uses an
  `L4_MicroCount`/`STW7500_DTW200` naming the same file contradicts elsewhere),
  so the note is the source.  Table 12 (p.41) agrees with sec. 3.2.
- **The L3 cut adapts on its own.**  `L3_oscNext_bool` and `Data_quality_bool`
  appear NOWHERE in the pass2 note -- they are the pass3 L3 script's own
  additions.  `l3_cut` therefore falls through to
  `IC2018_LE_L3_bools["IC2018_LE_L3_Full"]`, which pass2 does have (pp.36-44).
  Consequence: on pass2 the data-quality cut is not applied, i.e. the cut is
  slightly looser than on pass3.  That changes WHICH events arrive, not what
  any variable evaluates to, so it is harmless for the comparison (which runs
  on matched events) -- but the pass2 training population is not identical to
  pass2's official L4 population.
- **The L3 variable names are unchanged.**  `IC2018_LE_L3_Vars` exists at pass2
  with the same spelling for every column we read (`NchCleaned`, `ICVetoHits`,
  `RTVeto250Hits`, `NAbove200Hits`, ...).  Pass2's map carries BOTH
  `CleanedFullTimeLength`/`UncleanedFullTimeLength` AND their ratio
  `FullTimeLengthRatio`, so our division is compared head to head with the
  stored ratio.
- **The hit statistics keep the pass3 NAME on a pass2 run.**  `HITSTAT_KEY` /
  `HITMULT_KEY` derive from `CLEANED_PULSES_DEFAULT`, a module-level constant,
  not from the runtime parameter -- and `classifier.py` derives its own from
  the same literal.  So the values are computed from the series that was
  actually passed while the label keeps the pass3 spelling, and writer and
  reader stay consistent.  Cosmetic, deliberately left alone; the comparison
  table simply carries a different key name on each side.

### What is actually in the pass2 files (verified)

The P frames of `oscNext_genie_level3_v02.00_pass2.121122.000000.i3.zst` and of
the matching level4 file were dumped, so this is **verified**, not assumed:

- **L3**: `SRTTWOfflinePulsesDC` and `SplitInIcePulses` both present, both
  `I3RecoPulseSeriesMapMask` -- sec. 3.2 confirmed on real data.  No `L4_*` key.
- **pass2 L3 does NOT delete the hit statistics.**  Unlike pass3,
  `SRTTWOfflinePulsesDCHitStatistics` / `...HitMultiplicity` are already in the
  L3 frame, so pass2's L4 never recomputed them.  We recompute from the same
  series with the same module, which makes `cog_z` / `z_sigma` / `z_travel` a
  particularly clean control row.
- **L4** carries 28 `L4_*` keys.  Everything we compare is there under the
  spelling our constants already use: `L4_micro_count` (`I3MapStringInt`),
  `L4_fill_ratio` (`I3FillRatioInfo`), `L4_accumulated_time`,
  `L4_first_hlc_rho`, `L4_VICH_qtot`, `L4_separation_in_cogs` (`I3Double`),
  `L4_iLineFitParams` (`I3LineFitParams`), plus `L4_QR_Box` (pass2 had
  slc-veto) and `L4_ToIEval2/3`.
  **`L4_VICH_nch` and `L4_VICH_npulses` are `I3Int`**, where we write
  `I3Double`; both book to a `value` column, so the comparison is unaffected.
  **There is no `L4_FullTimeLengthRatio`** -- confirming it is an L3 variable
  at pass2, read from `IC2018_LE_L3_Vars`.
  `L4_oscNext_bool` is pass2's L4 cut, so the file holds every event with a
  bool rather than only the survivors.
- **Booking-audit bug 4 is visible in the real file.**  `L4_TWPulses`,
  `L4_SRTTWPulses`, `L4_TWPulses_DCFid` and `L4_TWPulses_DCFid_DTW200` all sit
  in the frame, and the fiducial selection is built from `L4_TWPulses` (the
  StaticTWC output) -- so `L4_SRTTWPulses` is written and never read.  The dead
  cleaning step is not an artefact of the commented-out source; it is in the
  production output.
  **Consequence for the comparison:** `micro_count` is EXPECTED to differ under
  our default (which follows the note's cleaned series).  Rerun with
  `--micro-count-uncleaned` to test the implementation rather than the
  decision -- our chain is then pass2's chain with the dead step left out.
  `pass2.EXPECTED_DEVIATION` prints this in the report when the row differs,
  and the summary line does not count it as a failure.
- **One trap:** `L4_Dunkman_SRTTWOfflinePulsesDC_Variables` cannot be
  deserialised -- it needs `analysis.event_selection`, one of the missing
  projects that made us rewrite `accumulated_time` in the first place.  It is
  not booked (the extracted `L4_accumulated_time` I3Double is), so this only
  ever mattered for `inspect`, which now guards the type lookup.
- **Not every L3 file has an L4 partner**: 121122 file `000000` exists at L3 and
  not at L4.  `pair_files()` reports such orphans rather than skipping quietly.

### The fitting scaffolding is TEMPORARY

`oscnext_l4/fit_pass2.py` and the `fit` / `fit-report` subcommands of
`scripts/compare_pass2.py` exist only to FIND the two definitions our rewrite
does not reproduce.  They are not part of the pipeline and nothing in the
production path imports them.

**When VICH and accumulated_time are settled: put the answer in
`oscnext_l4/variables.py` (the code plus a docstring recording the evidence),
record the finding here, then DELETE the scaffolding.**  Everything worth
keeping has to live in the code and in this file, so that removing the tooling
loses nothing.

**Two constraints on how it is run:**

1. **One L3 file per HDF5.**  `(Run, Event, SubEvent)` is unique only within a
   single L3 file (booking audit, bug 2), so a `--chunk-files` production
   cannot be matched against the answer key.  `pass2.match()` refuses to match
   when it sees a repeated triple instead of producing a plausible-looking
   disagreement table that is really an event-mixing artefact.
2. **Read the control rows first.**  The report separates *control* rows --
   the original IceTray modules run on both sides (`iLineFit_speed`, the hit
   statistics) or values taken straight from L3 (`NchCleaned`, `ICVetoHits`) --
   from the *rewritten* rows.  If a control row disagrees, the two sides did
   not see the same input and nothing below it means anything.

**For a pass2 training run:** there is no CORSIKA at pass2 in the paths we have
-- the muon background is MuonGun, so the pass3 CORSIKA weighting (open risk
3d) does not carry over.  The noise BDT is unaffected.  NuTau (160519) exists
at pass2, but the signal definition is still nue+numu (open risk 6).

## Conventions

- **Code, comments, docstrings, printed output, plot labels and documentation
  are all ENGLISH**, including this file and the notebook's markdown.
- Data never enters the repository (`.gitignore`: `L4_output/`, model/data
  extensions).
- The notebook must be cleaned with `nbstripout` before committing.
- Every rewritten function in `oscnext_l4/variables.py` records in its docstring
  where the original came from and why it differs -- read those docstrings
  before changing anything.
- Files under `reference/` are left **exactly as they are**: someone else's code
  or a historical record, never edited or translated.
