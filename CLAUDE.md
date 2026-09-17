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
- `icecube.hdfwriter` was thought to be missing from the cvmfs metaproject, so
  `oscnext_l4/booker.py` carries a pytables fallback.  **That is out of date.**
  The environment actually in use is PURE CVMFS -- `icetray/v1.17.0` under
  `py3-v4.4.2/RHEL_9_x86_64_v2`, with no local build anywhere in the path --
  and hdfwriter IS there.  Every measurement in this file, the pass2
  cross-check included, was produced on it.  `SimpleBooker` has therefore never
  run end to end; treat it as untested code, not as a working fallback.
  **hdfwriter is DEPRECATED in v1.17.0:** importing it warns *"icecube.hdfwriter
  is deprecated and will be removed in a future release.  Use icecube.tableio
  instead."*  That is a real forward risk, because `pass2.match()` depends on
  the `/__I3Index__/<key>` tables hdfwriter writes -- a metaproject upgrade that
  drops it breaks both booking and event matching.
- The old project dependencies: `tau_bdt.I3CutL7Module` (VICH),
  `analysis.event_selection` (the Dunkman variables: accumulated_time,
  separation_in_cogs), `slc-veto` (QR box, optional).
- **`LowEnVariables` (the `LowEn` C++ library) -- CHECKED, ABSENT.**
  `oscNext_master.py` points at it as "the new C++ versions" of the low-energy
  variables (`svn/sandbox/LowEnVariables/trunk/private/LowEnVariables/LowEnAlgorithms.cxx`),
  and a GRECO processing script uses it for `TimeTo75 = LowEn.TimeToSum(t, q,
  0.75)` on `SRTTWOfflinePulsesDC` -- an independent confirmation that
  accumulated_time's series and fraction are what we already verified (open
  risk 2).  `icetray.load("LowEnVariables")` fails in this build, so it cannot
  be run here.  It carries no VICH-like function in the code we have seen.

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
   │                                │
   │                                ├─ oscnext_l4.rewritten.{first_hlc,dunkman,vich}
   │                                └─ oscnext_l4.{geom,pulses,weighting,l3vars}
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
- `oscnext_l4/variables.py` — **the tray segments only**, structured to be read
  beside the production's own `oscNext_L4.py`: the same `L4_*` key names, the
  same segments in the same order, and the small per-segment helpers left inline
  exactly as the original leaves them.  Everything else was moved out so that
  stays true:
  - `oscnext_l4/rewritten/` — the pure-Python replacements for the three
    icetray modules this meta-project lacks.  `first_hlc.py`
    (`SimpleVertex FirstHLC<I3RecoPulse>`), `dunkman.py`
    (`analysis CalculateVariables` -> accumulated_time, separation_in_cogs),
    `vich.py` (`tau_bdt I3CutL7Module`).  Each is ONE `tray.AddModule` line in
    the original.  **The evidence for each definition is in its docstring** --
    where it came from, which of the several implementations the production
    actually ran, and what the measurement was.  Read those before touching
    anything here.
  - `oscnext_l4/geom.py`, `pulses.py` — `calc_rho_36`, `iter_map`,
    `get_pulses`.  The production takes these from
    `oscNext/frame_objects/geom.py` and `pulses.py`; the layout here mirrors
    that.
  - `oscnext_l4/weighting.py` — `PropagateGenieInfo`.  The production does this
    in `oscNext_master.py`, not in its L4 segment, which is why the original L4
    script has no counterpart.
  - `oscnext_l4/l3vars.py` — `FullTimeLengthRatio`.  An L3 variable
    (`oscNext_L3.py` writes it, and pass2's L3 map carries it); pass3's L3 map
    carries only the two components, so it is divided out at L4 instead.
- `oscnext_l4/env.py` — **every `icecube` import goes through here**; when an
  import fails it reports why.  `have_lightgbm()` lives here too.
- `setup_env.sh` — find the environment / open a shell / run one command /
  register a Jupyter kernel.
- `scripts/scan_files.py` — find corrupt `.i3.zst` files, write a healthy list.
- `oscnext_l4/runner.py` — the `process_L4.py` driver plus a live progress bar.
- `oscnext_l4/data.py` — `REGISTRY`/`ALTS`, `dump_tables`, `check_registry`,
  `check_feature_map`, `load_sample`, `add_weights`,
  `set_noise_weight_unit`.
- `oscnext_l4/productions.py` — **which production the notebook is pointed at,
  and the ONE place that knows a per-production fact.**  Every sample declares
  `kind` (its ROLE: signal / noise_bg / muon_bg) and `weight` (its weighting
  SCHEME: genie / noise / corsika / muongun).  Everything downstream keys on
  those, never on the sample NAME -- which is what lets one code serve both
  productions, because the roles are the same while the names are not (pass3's
  muon background is `corsika`, pass2's is `muongun`).  `data.add_weights`
  takes SAMPLES and resolves the weighter through `scheme_of`, and
  `AUX_SCHEMES` says which weight columns to ask a file for -- keyed by scheme,
  since CORSIKA and MuonGun are both `muon_bg` and want different columns.
  `select("pass2"|"pass3", HDF_BASE)` returns `(GCD, SAMPLES)` and sets the
  GCD, the sample paths, the `--cleaned-pulses` flag and the **noise weight
  unit** together.  That last one is the reason the function exists: pass3
  stores the vuvuzela weight in 1/ns and pass2 in Hz, and getting it wrong does
  not raise -- it scales every noise rate by 1e9.  Notebook cell 3 carries
  `PRODUCTION = "pass2"|"pass3"`; the output trees are suffixed so the two
  never overwrite each other (pass3 keeps its bare names).  A pass2 run loads
  `PASS2_NOISE_BDT` (nue, numu, noise) and section 6 skips the muon BDT when no
  muon background is loaded.
- `scripts/diagnose_env.py` — what is and is not in the environment.
- `docs/pipeline.md` — which file runs when.
- `docs/technical_note_comparison.md` — exactly what we write to HDF5, compared
  line by line with the technical note (Tables 7/10/11/12/13).
- `docs/pass2_verification.md` — the step-by-step record of checking the
  rewritten variables against the official pass2 production, from the first
  run to 14 of 15 rows bitwise identical: what each step established, the bugs
  it exposed on both sides, how `VICH` and `accumulated_time` were solved, and
  what measurement could and could not settle without the source.  Written to
  be readable on its own; §5 and §6 are deliberately kept as the historical
  snapshot they were, and say so.

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

- [x] Environment verified: PURE CVMFS, `icetray/v1.17.0` under
      `py3-v4.4.2/RHEL_9_x86_64_v2`.  No `oscNext` project, no `slc-veto`, no
      `tau_bdt`, no `LowEnVariables` -- but `hdfwriter` IS present (deprecated).
      Running this pipeline on cvmfs needs no change; it is what it runs on.
- [x] IceTray/lightgbm import layer (`oscnext_l4/env.py` + `setup_env.sh`)
- [x] Robust against corrupt input files (`--scan` + `--retries`)
- [x] nue processed (100 files → 256,799 events), CORSIKA (500 files → 6,462)
- [x] Column names pinned down (14/14 BDT inputs found)
- [x] Noise classifier trained: 95.9% efficiency at 99% rejection, train/test
      gap +0.1 points
- [ ] numu / noise need reprocessing -- cut short by corrupt `.i3.zst`
- [ ] Muon classifier not trained
- [x] **The rewritten variables verified against the real pass2 production:
      14 of 15 rows BITWISE IDENTICAL over 56,301 events in five samples**
      (NuE, NuMu, NuTau, MuonGun, Noise), every control row among them.
      The exception is `accumulated_time` at 99.84%, max difference 286 ns,
      whose residual is not reproducible (open risk 2).
- [ ] Noise MC statistics inadequate (~2,000 events) -- nothing right of 99%
      rejection is measurable

## Open risks / places worth thinking about

> The technical note has been read and compared with the code line by line:
> **`docs/technical_note_comparison.md`**.

1. **VICH — SOLVED (100.00%).**  `_vich` is now `VetoCausalHits` from the
   pythonic reimplementation of the LowEnVariables algorithms used by the GRECO
   online filter.  Measured against the real pass2 L4 files over 8144 events:
   `nch` **100.00%**, `npulses` **100.00%**, `qtot` **100.00%**, median
   difference 0 on all three.

   For every trigger with a matching config id (DeepCore SMT3, 1011), the hit
   CLOSEST IN TIME to it is the reference; with `dt = t_ref - t_hit` and `d` the
   distance to it, a hit counts when `d < 750`, `dt > -5d + 500`,
   `dt < d/0.3 + 150` and `dt > d/0.3 - 1850`.  The production loops over every
   matching trigger and accumulates; `nch` takes the union of the selected
   hits' DOMs.  Input: the uncleaned series, pulse by pulse
   (`GetHitInformation(..., hitMode=0)`), which was already verified from
   `reference/oscNext_L4_pass2_original.py`.

   **The old implementation could not have worked, and the search shows why.**
   It followed sec. 3.4's speed window [0.25, 0.40] m/ns over the veto DOMs
   around a fiducial COG, and reproduced pass2 in ~12% of events with the
   median difference pinned at 2 DOMs.  That passage describes the **Level 2
   DeepCore Filter**, a different algorithm that uses the window to DISCARD
   hits; Table 12's one line is the note's only description of `L4_VICH_nch`.

   Five hypotheses were retired by measurement, all landing at ~12%: the speed
   window (a 36-cell sweep of both edges peaked at 18.0%, median difference
   pinned at 2 in EVERY cell -- the signature of sweeping the wrong knob), the
   veto region (closed by construction: `DOMS.DOMS("IC86")` gives 554 fiducial
   + 4606 veto DOMs, disjoint, 5160 = the whole in-ice detector, so
   `DeepCoreVetoDOMs` already IS "not fiducial"), the COG (unweighted 12.0%,
   non-fiducial 11.9%), the reference point (first HLC hit, 11.7%) and the
   input series (cleaned, 2.8%).

   Three structural differences separate the real algorithm from the old one,
   and none is reachable by tuning: **no speed window** (four conditions in the
   (distance, dt) plane instead), **no veto DOM list** -- restricting the same
   bands to the veto DOMs drops agreement from 100% to **19.4%**, so the region
   is implicit in the bands -- and the **trigger** rather than a COG as the
   reference.  Note p.60 warned outright that "some algorithms used throughout
   their event selection may use differing definitions of what precisely
   constitutes the veto region"; it was more than that.

   `_vich` no longer takes `cleaned_pulses` or `fiducial_cog`.  One trap worth
   keeping: the MERGED and THROUGHPUT triggers carry **no** config id, and
   reading it unguarded raises and loses every trigger in the event -- which is
   exactly how the first attempt produced NaN in all 8144 events.

   **`tau_bdt` WAS found, and it is Python, not C++.**  The oscNext_meta
   V01-00-07 build carries `src/tau-bdt/python/I3CutL7Module.py` (note the
   hyphen in the directory, the underscore in the import), which is exactly
   what `from icecube.tau_bdt import I3CutL7Module` loads.  It confirms the
   bands, the `break` after the first matching trigger, `nch` as DOMs with at
   least one selected pulse, `npulses` as the pulse count, and `+= pulse.charge`
   with no clamping.  **It also corrected one thing measurement could not:**
   `TriggerConfigIDs` defaults to **[1010, 1011]** and the L4 tray does not
   override it, where we had used 1011 alone.  pass2 simulation carries no 1010
   trigger, so all 8144 events agreed either way; on detector data the first
   matching trigger could be a 1010 and the reference pulse would differ.
   Fixed.

   Three implementations of these bands exist and they do NOT agree with each
   other: `I3CutL7Module.py` (the one L4 runs) breaks after the first trigger;
   GRECO's `VetoCausalHits` loops over all matching triggers and accumulates;
   `I3CutL7Module_JP_Matt.py` breaks but restricts itself to 1011 and returns
   charge only.  Where they differ, the module L4 actually calls decides.

   One deliberate deviation remains: the original seeds its reference search
   with `refPulseTime = 0`, `refPulsePos = (0,0,0)`, so an event with no pulse
   closer to the trigger than t=0 would be computed against the origin.  A
   trigger at ~10 us makes that unreachable, and it never occurred in the 8144
   events; reproducing it would only copy a latent bug.

2. **accumulated_time — SOLVED from the source (99.84%), and it was never an
   off-by-one.**  `analysis/private/analysis/event_selection/CalculateVariables.cxx`
   -- the C++ module the L4 tray calls as `CalculateVariables` -- does not look
   for a 75% crossing at all.  It bins the time-sorted pulses into CHARGE
   QUARTILES and writes the variable in the Q3 branch:

       bin_edges = [0, Q/4, Q/2, 3Q/4, Q]
       accumulated_charge += current_charge
       part_index = index of the first edge >= accumulated_charge
       } else if ( part_index == 3 ) {
           variables.accumulated_time = pulse.GetTime() - first_slc.t();

   `part_index == 3` means `0.5Q < accumulated_charge <= 0.75Q`, and the line
   runs for every pulse in that band, so what survives is the LAST pulse whose
   cumulative charge has not passed 0.75Q.  That is exactly what the pass2 fit
   had found empirically (99.40% against 0.98% for the crossing), so the
   earlier description of pass2 "having an off-by-one" was wrong: it is what
   quartile binning gives.  The fit found the right rule for the wrong reason.

   Reading the source fixed three things measurement could not reach:

   - the bound is `<=`, not `<` (`lower_bound` returns `bin_edges[3]` when the
     cumulative charge lands exactly on 0.75Q);
   - when NO pulse falls in the Q3 band the struct default survives, and
     `Variables.h` says outright *"Default constructor (initialize all to
     zero)"* -- that is the source of the events where pass2 stores exactly 0.
     It needs one pulse to carry the cumulative sum from under 0.5Q to over
     0.75Q, i.e. more than a QUARTER of the event's charge (an earlier entry
     here said three quarters, which was wrong);
   - the module writes NO `Variables` object at all when the pulse map holds
     four or fewer DOMs, so neither Dunkman key reaches the frame.

   Together these took the agreement from 99.42% to **99.84%** and the maximum
   difference from 3927 ns to **286 ns** over 56,301 events.

   **THE RESIDUAL IS NOT REPRODUCIBLE, AND THAT IS MEASURED, NOT ASSUMED.**
   The original sorts with `std::sort` and a comparator that compares only
   `GetTime()`.  `std::sort` is not stable, so pulses sharing a time are left
   in an unspecified order.  If two tied pulses carry different charge, which
   of them is "the last in Q3" changes with that order.

   That was a hypothesis, and this file stated it as fact until it was tested.
   The test: the same quartile rule computed three times on one pass2 L4 file,
   8144 events, differing ONLY in the secondary sort key.

   | tie-break | agreement |
   |---|---|
   | smaller charge first | 99.89% |
   | **stable (ours)** | **99.80%** |
   | larger charge first | 99.67% |

   **The three differ, so tied times exist and they decide the residual.**  And
   none reaches 100%, so the order `std::sort` produced is not any simple rule
   we could adopt.  (The variants live in `fit_pass2.ACC_VARIANTS` as
   `production_tie_*`; the same file gives 99.40% for the old `prev_entry`
   rule, so the move to quartiles is confirmed here too.)

   **We keep the stable sort and do NOT chase the better number.**  99.89% vs
   99.80% is 7 events in 8144 -- 9 differing against 16, about 2 sigma on one
   file.  Adopting an arbitrary tie-break to capture that is fitting to noise,
   and it would trade a reproducible answer for one that is not.

   A second hypothesis was tested and ELIMINATED: summing the total charge in
   map order rather than with `np.sum` (the original accumulates it in a first
   loop over the map while the cumulative runs in time order) changed not one
   event.

   Remaining difference: 0.16% of events on the sixth of ten muon BDT inputs,
   bounded by one pulse spacing.  `--accumulated-time-note` /
   `oscNext_L4(accumulated_time_pass2=False)` follows Table 12's description
   instead ("time to reach 75%"), which agrees with pass2 in 0.98% of events.
   LowEnVariables' own `TimeToSum` uses `>= fraction`, i.e. the crossing, so
   the note's reading is the better-documented one -- it is simply not what
   pass2 produced.

2a. **first_hlc -- TWO REAL BUGS, both found by reading the module.**
   `SimpleVertex/private/SimpleVertex/FirstHLC.cxx`, registered as
   `I3_MODULE(FirstHLC<I3RecoPulse>)`, is what
   `tray.AddModule("FirstHLC<I3RecoPulse>", ...)` instantiates.  (A Python
   class of the same name in `analysis/python/yanez/FirstHLC.py` is a
   different author's module: its parameters are `InputPulseSeries`/`Vertex`
   where the L4 tray passes `HitSeriesName`/`OutputName`.)

   - **Ties go to the LAST DOM.**  The C++ reads `if (hitTime > hlc_time)
     continue;` -- it skips only a strictly later hit, so a hit at exactly the
     current best time overwrites it, and the highest OMKey among tied hits
     wins.  This code kept the first.  That was the whole of the 146 m
     disagreements: a tie resolved to a different string.
   - **An event with no HLC hit still gets a vertex.**  `GetFirstHLCHit`
     returns a bool that `Physics` ignores, so the search's starting value is
     written: position (1000, 1000, 1000), time 1e10, i.e. a `first_hlc_rho`
     of 1407.3.  The production's `fill_ratio` and muon BDT both saw that
     number where we wrote nothing at all.

   Both fixed.  `first_hlc_rho` and `fill_ratio` went from 0.9987 and 0.9831
   to **bitwise identical** over 56,301 events -- which also explains why
   `fill_ratio` disagreed ten times more often than `first_hlc_rho` did:
   `rho` compares only sqrt((x-46.29)^2+(y+34.88)^2), so a different DOM on
   the same string gives an identical rho and a different z, and only
   `fill_ratio`, which is handed the whole vertex, could see it.

2b. **separation_in_cogs -- the definition was a guess, and it was wrong.**
   `CalculateVariables.cxx`: `variables.separation = CalcDistance(cog_q1,
   cog_q4)`, with the quartiles accumulated in the same charge-quartile loop
   as accumulated_time, and `Variables.h` documenting it as *"Distance between
   CoG_Q1 and CoG_Q4"*.  This code split the event into two halves by HIT
   COUNT and compared those COGs: a different split measure (count, not
   charge) and different parts (halves, not the outer quartiles -- the middle
   half of the charge is excluded entirely).  Fixed.  Not a BDT input, so it
   is behind `--run-optional` and is not in the comparison table; adding a row
   would verify it for free.

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
   ORIGINAL.**  The original pass2 code starts the chain from
   `uncleaned_pulses` (`I3StaticTWC(InputResponse=uncleaned_pulses)`), while
   Table 11 says *"Start with the cleaned pulse series"*.  **We follow the
   original**, so the default reproduces pass2; `--micro-count-cleaned` follows
   the note.  This reverses an earlier decision to follow the note, for the
   same reason as open risk 2: reproducing the production's numbers comes
   before matching its documentation.
   **Measured -- the difference is almost nil:** the two chains were computed on
   the same events.  115 of 126 nue events and 16 of 17 noise events are
   **exactly equal**; the medians agree (5 and 3).  The reason: the closing
   200 ns window is what decides, and since noise hits are spread over ~10 µs
   the densest 200 ns window catches the same hits in either series.  That is
   also why the dead SeededRT code in pass2 went unnoticed for years.
   So the fix is **right but small**; the original assessment ("the
   classifier's separating power was seriously degraded") was overstated.
   Both can be produced for comparison: `process_L4.py --micro-count-cleaned`,
   `run_all(extra_args=["--micro-count-cleaned"])` from the notebook, or
   `oscNext_L4(micro_count_uncleaned=False)` directly.  `fill_ratio` uses the
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

5i. **The production trains the muon BDT ONLY on events that pass the noise
   cut -- we do not.**  The official workflow (fridge,
   `processing/samples/oscNext/selection/level4`, README) runs in this order:

       python L4_noise_model_train.py
       python ../tools/apply_model.py -d L4_model_data.hdf5 \
              -m L4_noise_model.joblib -k L4_NoiseClassifier_ProbNu.value
       python L4_muon_model_train.py

   with its own reason: *"This is then cut on in the muon classifier before
   training to avoid training using events that will be cut anyway."*

   Section 6 of the notebook trains both classifiers on the SAME event
   population, so our muon BDT sees events the production's never would.  That
   changes the background and signal distributions it fits, not just its
   normalisation.  The muon classifier is not trained yet, so this is cheap to
   fix now: apply the trained noise model to the dataset, cut at
   `P_noise >= 0.70`, and build the muon dataset from the survivors -- keeping
   the shared signal split (open risk 5c) intact.

   The same README also shows the production stores its models as
   `.joblib` + `.hdf5`, where we use LightGBM's native text plus a JSON
   sidecar -- a deliberate difference (the IceTray environment has no
   sklearn/joblib), not an oversight.

5j. **Where the pass2 L4 was actually produced.**  The fridge README pins it:
   L1-L5 ran under `oscNext_meta V01-00-07` on **py2-v3.1.1**, deployed at

       /cvmfs/icecube.opensciencegrid.org/users/Oscillation/software/oscNext_meta/releases/V01-00-07/build/

   That is a readable cvmfs path and it is the build that contains every
   project this repository had to rewrite around (`tau_bdt`,
   `analysis.event_selection`, `slc-veto`), plus -- per the README's own `cp`
   instruction -- the real trained models in
   `oscNext/resources/models/L4_noise_model.joblib` / `.hdf5`.  The `.hdf5`
   sidecar would give the production's exact BDT input list rather than our
   reading of Tables 11/12.  Python 2 is why `oscNext_L4.py` in the GitHub port
   is still entirely commented out with `#TODO migrate`.

6. **No real detector data (and ν_τ is now IN at pass2).**  The muon BDT's
   background is simulation rather than the real data the production used --
   see "Five real differences" item 2, which is the larger half of this risk.
   The ν_τ half is closed at pass2: the notebook now takes its signal sets by
   ROLE (`kind == "signal"`) rather than from a hardcoded `["nue", "numu"]`,
   so pass2's NuTau (160511) is included, as `L4_model_data.py` does.  pass3
   has no NuTau set in the paths we have, so a pass3 run is still νe+νμ --
   by what the production offers, not by a branch in our code.

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
- `icetray-oscNext/` — **the official `icecube.oscNext` project**, the one this
  meta-project lacks.  What it settles and what it does not:
  - `oscNext/python/selection/oscNext_L4.py` is **byte-identical to our
    `reference/oscNext_L4_pass2_original.py`** -- still entirely commented out
    (`#TODO migrate`).  So it adds nothing about L4 that we did not have.
  - `oscNext/python/selection/oscNext_L3.py` IS real, uncommented code (686
    non-comment lines), and it shows how the collaboration actually builds a
    veto series: `I3OMSelection` with `OmittedKeys=DOMList.DeepCoreFiducialDOMs`
    on `splituncleaned`, i.e. **"veto" means NOT FIDUCIAL**, not a separate
    list.  **It is NOT a narrower list, and an earlier entry here said it
    was.**  Measured on the real object: `DOMS.DOMS("IC86")` gives 554
    fiducial and 4606 veto DOMs, disjoint, summing to 5160 = 86 x 60, i.e.
    the whole in-ice detector.  `DeepCoreVetoDOMs` IS the complement of
    `DeepCoreFiducialDOMs`, so our `_vich` region already equals the
    production's "not fiducial".  The veto region is therefore closed as a
    VICH hypothesis **by construction**, not merely by the 0.16% ceiling
    test (open risk 1).  (The only residue: `~fiducial` taken over the
    geometry would also admit IceTop DOMs, which never appear in
    `SplitInIcePulses`.)
  - **`tau_bdt` is NOT in it**, so `I3CutL7Module` is still unread.  The only
    mention is `VICH = "L7VetoHitsTotalPE"`, a parameter of the old GRECO L5
    BDT (`TauBDTL5`), not of our variable.
  - **`n_flux_events` and `I3GenieInfo` occur ZERO times in the whole
    project.**  Our `PropagateGenieInfo` path is entirely our own, and the
    production formula is the one we call the fallback.
  - **A four-part event ID.**  `frame_objects/simulation.py`'s
    `FixSimEventHeaders` writes `run_id = dataset_id`, `sub_run_id = file_id`,
    `event_id` incrementing and unique WITHIN a file, `sub_event_id`
    untouched -- and `oscNext_master.py` runs it with `assert_unique=True`.
    So the identifier that is unique across files is
    `(run, sub_run, event, sub_event)`, where `data._ids()` uses three and
    drops `sub_run`.  That omission is exactly why the cross-check needs one
    L3 file per HDF5; booking `SubRunID` would lift the constraint.  Worth
    checking whether the I3EventHeader table already carries it.
  - `frame_objects/geom.py` also offers a GEOMETRIC fiducial definition
    (`is_dom_in_deepcore_fiducial(use_dom_list=False)` ->
    `get_deepcore_containment`: rho36 < 150 m and z inside the DeepCore band)
    alongside the DOM-list one, with its own `#TODO does this exactly
    correspond to the definition of fiducial in the DeepCore filter?`.  A
    third candidate for VICH's region, untested.
  - `frame_objects/muongun.py` gives the MuonGun weight for a future pass2
    muon BDT: `raw_weight / num_events / prob_passing_KDE`.
  - **The whole GENIE weight chain is confirmed by it.**
    `frame_objects/weighting.py` adds a single power law with `norm=2.e-2`,
    `spectral_index=-3.` *for machine-learning training samples* -- our `NORM`
    and `GAMMA` to the digit, and no longer "an invented power law".
    `frame_objects/neutrinos.py` computes
    `weight = OneWeight * flux / (NEvents * gen_ratio)` with
    `flux = norm * E**index` and `gen_ratio = 0.7` for GENIE (`1 - 0.7` for
    antineutrinos), and weighting.py divides by the file count at the end.
    Every piece matches `data.genie_weight`.
    **This inverts open risk 4:** what our code calls the *fallback*
    (`NEvents * nu/nubar fraction`) IS the production formula, and our
    `n_flux_events` path -- read from `I3GenieInfo`, which the official code
    never touches -- is the deviation.  They agree only if
    `n_flux_events == NEvents * gen_ratio`, still unmeasured; pass3 carries
    both, so it can be checked.
  - `frame_objects/noise.py` copies the vuvuzela weight through with **no unit
    conversion at all**, so the production applies no scale factor -- which is
    why `NOISE_NS_SCALE` is a per-production fact about what vuvuzela wrote,
    and why "hz" (factor 1) is right for pass2.
  - `selection/globals.py` confirms from code what sec. 3.2 says:
    `UNCLEANED_PULSES = "SplitInIcePulses"`,
    `CLEANED_PULSES = "SRTTWOfflinePulsesDC"`,
    `SUB_EVENT_STREAM = "InIceSplit"`, `L4_CUT_BOOL_KEY = "L4_oscNext_bool"`.
  - `frame_objects/geom.py` gave the real `calc_rho_36`:
    `np.sqrt((x-46.29)**2 + (y+34.88)**2)`.  Our constants were right but we
    used `np.hypot`, a different algorithm that disagrees in the last bit --
    which is the whole reason `first_hlc_rho` matched bitwise in only 80.6% of
    events while agreeing to 1e-6 in 99.85%.  **Fixed**; the two forms now
    agree on 20000/20000 random points where hypot managed 16728.
  - `oscNext/python/tools/classifier.py` is the `I3Classifier` our
    `oscnext_l4/classifier.py` replaces, but it is generic -- the feature list
    is passed in by `oscNext_L4.py`, which is dead -- so the L4 input list
    cannot be recovered from it.
  - `frame_objects/pulses.py` has `calc_space_time_relation_between_pulses`,
    which looks like VICH at a glance but is an SRT-style pulse-to-pulse
    causality helper, not it.  Its `pulse_info` segment also shows that the
    production DERIVES the hit-statistics key from the series name
    (`stats_key = pulses + "HitStatistics"`), where our `HITSTAT_KEY` is a
    module-level constant carrying the pass3 spelling -- the deliberate,
    cosmetic deviation already recorded under "Running on pass2".
  - **`FullTimeLengthRatio`'s direction is now confirmed from production
    code**, closing open risk 3 for good.  `oscNext_L3.py` (live code, line
    708):
    `LE_L3_Vars['FullTimeLengthRatio'] = 1.*CleanedFullTimeLength / UncleanedFullTimeLength`,
    with each length taken as `HitStatistics.max_pulse_time -
    min_pulse_time`.  Three independent sources now agree: Figure 13's axis,
    our own measurement against pass2 (identical over 8144 events), and this.
    The same block defines `NchCleaned` as
    `len(frame["SRTTWOfflinePulsesDC"].apply(frame))` -- a DOM count, as we
    assume.
  - **The detector string differs by LEVEL, and ours follows L4.**
    `oscNext_L3.py` builds its DOM lists with `DOMS.DOMS("IC86EDC")` while the
    L4 script uses `DOMS.DOMS("IC86")`, which is what `oscnext_l4/env.py`
    passes.  Not a conflict -- different levels by design -- and micro_count
    reproducing pass2 exactly confirms "IC86" is right at L4.
  - A full-tree sweep for every L4-relevant term (micro_count, fill_ratio,
    iLineFit, StaticTWC, TimeWindowCleaning, FullTimeLength, CutL7, tau_bdt,
    Dunkman, accumulated, first_hlc, DeepCoreVeto, OMSelection, TriggerConfig,
    ...) hits ONLY `oscNext_L3.py`, `oscNext_L4.py`, `oscNext_L5.py`,
    `pulses.py`, `geom.py`, `oscNext_master.py` and `oscNext_GNN_L7.py`.  The
    remaining nineteen files (corridor_cut, simulation, reco, genie, muongun,
    photons, physics, trigger, i3_to_analysis, processor, metadata, misc,
    scaling, hash_tools, file_transfer, data_quality, load_data_classes,
    GNN_L6, and oscnext_scripts/) contain nothing about the L4 variables.
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
`icecube` namespace -- an ordinary pip package.  In the environment actually in
use it needs no installing: `lightgbm 4.5.0` ships inside cvmfs itself
(`py3-v4.4.2/RHEL_9_x86_64_v2/lib/python3.12/site-packages`), not as a `--user`
install.  So a missing lightgbm means the kernel is not the env-shell python,
not that a package is absent -- check `sys.executable` before installing
anything.

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
- **The L3 cut is EXACT on pass2, and an earlier entry here said otherwise.**
  That entry reasoned from the note -- `L3_oscNext_bool` and
  `Data_quality_bool` are not mentioned in it -- and concluded they were the
  pass3 script's own additions, so that on pass2 our fallback would drop the
  data-quality cut.  **Wrong.**  The official `oscNext_L3.py` writes both, and
  folds data quality INTO the bool we fall back on:

      LE_L3_2018_bools["IC2018_LE_L3_Full"] = (L3_hit_bool * rt_veto_hit_pass
                                               * nch_pass * data_quality_bool)
      frame[L3_CUT_BOOL_KEY] = icetray.I3Bool(
          frame[L3_2018_ALL_BOOLS_KEY]["IC2018_LE_L3_Full"])

  So at pass2 `L3_oscNext_bool` IS `IC2018_LE_L3_Full`, both branches of our
  `l3_cut` give the same answer, and the data-quality cut is applied either
  way.  The pass2 training population matches pass2's own.
  The pass3 script differs and the AND there is real: its `Full` comes from
  GRECO's copied `DeepCoreCuts`, which does not fold data quality in, so
  `reference/pass3_L3_process.py` computes `Data_quality_bool` itself and ANDs
  it.  Two productions, two shapes, same result.
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
  **Consequence for the comparison:** none any more.  The default now starts
  from the uncleaned series as pass2 does, so `micro_count` comes out BITWISE
  IDENTICAL and a disagreement there would be a real finding.  Our chain is
  pass2's chain with the dead step left out, and the two agree exactly because
  that step never did anything.  `--micro-count-cleaned` gives Table 11's
  reading instead, and THAT is expected to differ.
- **One trap:** `L4_Dunkman_SRTTWOfflinePulsesDC_Variables` cannot be
  deserialised -- it needs `analysis.event_selection`, one of the missing
  projects that made us rewrite `accumulated_time` in the first place.  It is
  not booked (the extracted `L4_accumulated_time` I3Double is), so this only
  ever mattered for `inspect`, which now guards the type lookup.
- **Not every L3 file has an L4 partner**: 121122 file `000000` exists at L3 and
  not at L4.  `pair_files()` reports such orphans rather than skipping quietly.

### The fitting scaffolding is GONE (recover it from the tag)

`oscnext_l4/fit_pass2.py` and the `fit` / `fit-report` subcommands existed to
FIND the definitions the rewrite did not reproduce.  Both are settled -- VICH
and accumulated_time are in `oscnext_l4/variables.py` with the evidence in
their docstrings, and the measurements are in open risks 1, 2, 2a and 2b here
-- so they were deleted, as this file said they would be.  Removing them lost
nothing: everything they established is written down.

`scripts/run_crosscheck.py` went with them.  It drove the cross-check from the
command line before `notebooks/pass2_verification.ipynb` existed; the notebook
does the same job and pools the events across files, which the script could
not.

**If a definition ever has to be fitted again** -- a pass3 variable that does
not reproduce, `separation_in_cogs`, a new L4 variable -- the machine is one
command away:

    git show pass2-verified-v1:oscnext_l4/fit_pass2.py > oscnext_l4/fit_pass2.py
    git show pass2-verified-v1:scripts/compare_pass2.py > scripts/compare_pass2.py

It is worth recovering rather than rewriting: it carries the variant/inversion/
grid-scan method that found VICH, and the `production_tie_*` variants that
measured the accumulated_time residual.

**Two constraints on how it is run:**

1. **One L3 file per HDF5.**  `(Run, Event, SubEvent)` is unique only within a
   single L3 file (booking audit, bug 2), so a `--chunk-files` production
   cannot be matched against the answer key.  `pass2.match()` refuses to match
   when it sees a repeated triple instead of producing a plausible-looking
   disagreement table that is really an event-mixing artefact.
   **This is why the cross-check cannot use `runner.run_process_parallel`**,
   and why `notebooks/pass2_verification.ipynb` carries its own parallel loop.
   That runner chunks several L3 files into one part AND names its output by
   job/part index (`L4_nue_job0_part000.hdf5`), so even at `chunk_files=1` the
   file could not be paired with the right L4 partner.  The verification names
   each output after its L3 file instead.
2. **Read the control rows first.**  The report separates *control* rows --
   the original IceTray modules run on both sides (`iLineFit_speed`, the hit
   statistics) or values taken straight from L3 (`NchCleaned`, `ICVetoHits`) --
   from the *rewritten* rows.  If a control row disagrees, the two sides did
   not see the same input and nothing below it means anything.

**For a pass2 training run:** there is no CORSIKA at pass2 in the paths we have
-- the muon background is MuonGun, so the pass3 CORSIKA weighting (open risk
3d) does not carry over.  The noise BDT is unaffected.  NuTau (160511) exists
at pass2, but the signal definition is still nue+numu (open risk 6).

## Verified against the PRODUCTION SOURCE (oscNext_meta V01-00-07)

The build that actually produced pass2 L1-L5 is readable on cvmfs at
`/cvmfs/icecube.opensciencegrid.org/users/Oscillation/software/oscNext_meta/releases/V01-00-07/`,
and the training recipe lives in the fridge at
`/data/sim/DeepCore/2018/workspace/fridge/processing/samples/oscNext/selection/level4/`.
Its `oscNext_L4.py` is **live code**, unlike the commented-out copy in
`reference/`.  Everything below was checked against it line by line.

### The L4 tray -- every variable AGREES

| variable | the production | ours |
|---|---|---|
| `first_hlc` | `FirstHLC<I3RecoPulse>`, `HitSeriesName=cleaned_pulses` | same series |
| `first_hlc_rho` | `calc_rho_36(pos.x, pos.y)` | same, and the formula now matches bit for bit |
| `iLineFit` | `linefit.simple`, `inputResponse=cleaned_pulses` | verbatim |
| `ToI` | `I3TensorOfInertia`, amplitude 1/1, MinHits 3, cleaned | verbatim (optional, not a BDT input) |
| `QR_Box` | `SmallQ_Box`, `RecoPulsesKey=cleaned_pulses` | verbatim (optional) |
| Dunkman | `CalculateVariables(PulseSeries=cleaned_pulses)` | `accumulated_time`, `separation_in_cogs` from the cleaned series |
| `VICH` | `I3CutL7Module(InputPulses=uncleaned_pulses)`, three output keys | uncleaned, same three keys |
| `micro_count` | `uncleaned` -> StaticTWC [1010,1011] -3500/+4000 -> (dead SRT) -> `I3OMSelection(DeepCoreFiducialDOMs, "IC86")` -> DTW 200 -> `len(values())` | identical, minus the dead step |
| `fill_ratio` | `RecoPulseName=cleaned`, `SphericalRadiusMean=1.6`, `VertexName=L4_first_hlc` | verbatim |
| L3 cut | `oscNext_cut(processing_level=3)` first | same |

**Booking-audit bug 4 is confirmed in LIVE code**, not just in the
commented-out copy: `I3OMSelection` takes `tw_pulses`, while `srt_tw_pulses`
is written and never read, right under the author's own
`#TODO Is this actually used?`.

### The BDT input lists -- DEFINITIVE, and ours match exactly

No longer read off Tables 11/12.  From `L4_noise_model_train.py`:

    L4_NOISE_MODEL_INPUT_VARIABLES = [
        "IC2018_LE_L3_Vars.NchCleaned",
        "L4_micro_count.STW_m3500p4000_DTW200",
        "L4_iLineFit.speed",
        "L4_fill_ratio.fill_ratio_from_mean",
        "IC2018_LE_L3_Vars.FullTimeLengthRatio" ]

and from `L4_muon_model_train.py` (the v01.01 list): `ICVetoHits`,
`NAbove200Hits`, `RTVeto250Hits`, `NchCleaned`, `L4_VICH_nch`,
`L4_accumulated_time`, `L4_first_hlc_rho`, `HitStatistics.cog.z`,
`.z_sigma`, `.z_travel`.

5 + 10, 14 unique -- **exactly `NOISE_FEATURES` and `MUON_FEATURES`**, subkey
spellings included.  This also settles open risk 3a: `NchCleaned` does belong
in the muon list.

### The hyperparameters -- identical

`max_depth 6`, `num_leaves 25`, `max_bin 32`, `min_data_in_leaf 500`,
`lambda_l1 2.0`, `lambda_l2 1.0`, `min_gain_to_split 2.0`,
`is_unbalance False`, and **`feature_fraction` 0.8 for noise / 0.7 for muon**
-- `PARAMS` in `train_L4_classifier.py` to the digit, the 0.8/0.7 split
included.  Class balancing also matches in method: they pass
`class_ratio={k: 1.}` with `scale_weights=True`, we balance through `weight`.
The cut thresholds `P_noise >= 0.7` and `P_muon >= 0.65` are confirmed in
`compute_L4_cut`.

### Five real differences, none of them a bug in our variables

1. **The signal includes ν_τ.**  `L4_model_data.py` harvests GENIE 12xxxx,
   14xxxx AND **16xxxx**, all as `CLASSES["neutrino"]`.  We define the signal
   as νe+νμ (open risk 6).  pass2 NuTau (160511) exists in the paths we have,
   so this is fixable.

2. **The muon background is REAL DETECTOR DATA, not simulation.**  The muon
   classifier is trained with `--muon-events data` against `CLASSES["traindata"]`
   -- one run per month, 2012-2018 -- and the L4 cut uses
   `L4_MuonClassifier_Data_ProbNu`.  A MuonGun-trained model exists but the
   file says outright: *"we never actually used the MC trained muon classifier
   anyway (we used the data driven one instead for the sample)"*.  CORSIKA was
   dropped too: *"we never got a decent CORSIKA set"*.  So open risk 6's
   "CORSIKA instead of data" is not a small substitution -- it is the opposite
   of what the production did.

3. **Detector-data weights are `1 / livetime_s` per event**, applied after
   harvesting so the histograms come out in Hz like the MC.

4. **The train/test split is not 50/50.**  `train_fraction` is
   `{"neutrino": 0.02, "noise": 0.3333}` for noise and `{"neutrino": 0.02,
   "traindata": 0.5}` for muon.  **Do not copy the fraction**: the comments
   show they are targeting an absolute size -- *"1% with new nominal dataset
   (0000) ... gives a comparable 2.24e5 events.  Doubling to 2% to address
   overtraining"*.  Our `TRAIN_FRAC = 0.5` on 330k signal gives 165k training
   events, the same regime as their 2.2e5, so 0.5 is right for our statistics
   and 0.02 would leave 6.6k.

5. **The noise straight cuts, exactly** (`L4_NoiseStraightCuts_Bool`, a loose
   cut the production computes but does not use):
   `n_hit_doms >= 8`, `STW9000_DTW300Hits >= 2`,
   `L4_micro_count["STW_m3500p4000_DTW200"] >= 2`,
   `fill_ratio_from_mean >= 0.03`, `z_sigma >= 8`, `z_travel >= -50`.

### `first_hlc_rho` has a known pathology -- worth knowing before we use it

`investigate_first_hlc_rho_issue.py` exists because the variable *"shows
significant disagreement pre- vs post- 2017 run start, once the L4 classifier
cut has been made"*.  The script scans `P_nu` against `first_hlc_rho` and
overlays the string positions: the model learns **step functions at the exact
rho of each string**, and the 2016 and 2017 GCDs differ in string x-y by ~1e-5 m
(string 84: 71.477530 vs 71.477533).  That is enough to flip events across a
learned boundary.

The variable is the seventh of our ten muon inputs, so this is a caution, not
a blocker -- but if our muon BDT shows a season- or GCD-dependent step, this
is the first place to look, and it is not our bug.

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
