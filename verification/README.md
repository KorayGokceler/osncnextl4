# Verifying the rewritten L4 variables against pass2

A step-by-step record of how the L4 variables this repository computes were
checked against the official oscNext pass2 production, what each step
established, and what is still open.

**Result: 14 of 15 rows bitwise identical over 56,301 events** in five samples,
every control row among them.  The exception is `accumulated_time` at 99.84%,
maximum difference 286 ns, whose residual is measurably not reproducible (§7).

Everything below is a measurement on real files, not an inference from
documentation, unless it says otherwise.  The record is kept in the order the
work happened, so **§5 and §6 state conclusions that §7 and §8 later correct**;
they are marked where that is the case, and they are kept because how a wrong
reading survived four rounds of measurement is part of what this document is
for.

---

## 1. Why a verification was needed

`oscnext_l4/variables.py` rebuilds the Level 3 → Level 4 step.  Some of it runs
the original IceTray modules, but several variables had to be **rewritten in
pure Python**, because the meta-project in use (`py3-v4.4.2`) does not contain
the projects the original L4 script imported:

| missing project | variable it produced |
|---|---|
| `tau_bdt` (`I3CutL7Module`) | `VICH_nch`, `VICH_npulses`, `VICH_qtot` |
| `analysis.event_selection` (`CalculateVariables`) | `accumulated_time`, `separation_in_cogs` |
| `oscNext` | `calc_rho_36`, `I3Classifier`, `oscNext_cut` |

Those rewrites followed the technical note's one-line table descriptions.  The
note does not state every parameter, so the details were inferred — and until
this exercise, **never compared against the numbers the original code actually
produced**.

## 2. The idea: pass2 is an answer key

The pass2 production gives both halves of a controlled comparison:

- the **L3 files** are the input our code was built to read;
- the matching **L4 files** already contain every variable, computed years ago
  by the original code.

So the check is the job we already do, run on their input and held against
their output:

```
pass2 L3  --(process_L4.py, our segment)-->  ours.hdf5
pass2 L4  --(compare_pass2.py book)------->  pass2.hdf5
                 match on (Run, Event, SubEvent), compare column by column
```

The "ours" side is an **ordinary `process_L4.py` output**, so the same file is
also the training input for a pass2-trained model.  Nothing about this is a
special production.

Two constraints fell out of the design:

1. **One L3 file per HDF5.**  `(Run, Event, SubEvent)` is unique only within a
   single L3 file, so a chunked production cannot be matched.  The code refuses
   to match when it sees a repeated triple rather than producing a
   plausible-looking disagreement table that is really an event-mixing
   artefact.
2. **Control rows are read first.**  Rows where both sides run the *same
   original module* (`iLineFit_speed`, the hit statistics) or where the value
   comes straight from L3 (`NchCleaned`, `ICVetoHits`) test the setup, not our
   code.  If those disagree, nothing below them means anything.

## 3. Establishing that both sides see the same thing

Before any comparison could be trusted, four things had to be pinned down.

**The pulse series.**  Technical note sec. 3.2 (p.25) defines both outright:
uncleaned is `SplitInIcePulses` — *the same string as pass3* — and cleaned is
`SRTTWOfflinePulsesDC`, where pass3 uses `SRTTWSplitInIcePulsesDC`.  Confirmed
on a real frame.  **The whole delta between a pass3 run and a pass2 run is one
flag:** `--cleaned-pulses SRTTWOfflinePulsesDC`.

The original pass2 script cannot settle this — both series are parameters there
with no defaults — and its straight-cut block, which does mention
`SRTTWOfflinePulsesDCHitStatistics`, is stale.  The note is the source.

**The L3 cut adapts by itself, and it is EXACT.**  This section first said the
opposite, reasoning from the note: `L3_oscNext_bool` and `Data_quality_bool` are
not mentioned there, so they looked like the pass3 script's own additions, and
falling back to `IC2018_LE_L3_bools["IC2018_LE_L3_Full"]` looked like dropping
the data-quality cut.  The official `oscNext_L3.py` shows otherwise: data
quality is folded INTO `IC2018_LE_L3_Full`, and `L3_oscNext_bool` is set to
exactly that bool.  Both branches of our `l3_cut` therefore agree, and the
pass2 event population is the production's own.  (The pass3 script's extra AND
is real, because its `Full` comes from GRECO's copied `DeepCoreCuts`, which
does not fold data quality in.)

**The geometry.**  Both sides were given the same GCD, the one sitting in the
pass2 level4 directory.  That it is the *right* GCD is not assumed: `cog_z`,
`z_sigma` and `z_travel` are computed by us from our GCD and were stored by
pass2 from theirs, and they agree **to the last bit**.

**The event set.**  8144 events booked on our side, 8144 on theirs, **8144
matched**.  Our L3 cut reproduces their event population exactly.

## 4. Bugs the first run exposed

Both were real, and both had been hidden because the code path had never run.

1. **`process_L4.py` — `validate_files` was used but never imported**, so the
   DEFAULT `--scan quick` died with `NameError` before reading a single file.
   Present since the initial commit and still on `main`; unnoticed because the
   notebook path passes `--scan off`.  `pyflakes` over the whole repository
   confirms this was the only undefined name.
2. **`compare_pass2.py book` — no converter for `I3HitMultiplicityValues`**, so
   `I3TableWriter` died mid-run and left a half-written HDF5.  Those converters
   register when `common_variables` is imported; `process_L4.py` gets that for
   free because its hit-statistics segment imports the module to *compute*
   them, while the booking-only script imported nothing.

## 5. First result

> **This is the first run's table, kept as a record of where the investigation
> started.  It is NOT the result.**  Four of its rows were later taken to
> bitwise identical and one was reversed outright; §10 has the final table.

With the control rows all `identical`, the comparison became readable.

| variable | role | result |
|---|---|---|
| `NchCleaned`, `ICVetoHits` | from L3 | identical |
| `iLineFit_speed` | original module | identical |
| `cog_z`, `z_sigma`, `z_travel`, `n_hit_doms` | original module | identical |
| `micro_count` | rewritten | **identical** (starting from the uncleaned series) |
| `FullTimeLengthRatio` | rewritten | **identical** |
| `first_hlc_rho` | rewritten | **99.85%** |
| `fill_ratio` | original module, our vertex | 98.1% |
| `accumulated_time` | rewritten | systematically LARGER than pass2 |
| `VICH_nch` / `npulses` / `qtot` | rewritten | systematically SMALLER |

**`micro_count`** reproduces pass2 exactly once the chain is started from the
uncleaned series, which validates every parameter of it: the StaticTWC window,
the trigger IDs, the DeepCore fiducial selection, the 200 ns time window and
the DOM counting.  The pass2 L4 file also shows the dead cleaning step
(booking-audit bug 4) in the production output itself: `L4_TWPulses_DCFid` is
built from `L4_TWPulses`, so the `L4_SRTTWPulses` in the same frame is written
and never read.

At this point our default followed the note (the cleaned series) and therefore
differed on purpose.  **That decision was later reversed** -- reproducing the
production outranks matching its documentation -- so the uncleaned chain is now
the default and `--micro-count-cleaned` follows the note.  The difference is
small either way: the closing 200 ns window decides, and it catches the same
hits in both series (115 of 126 nue events exactly equal).

### A reporting flaw found and fixed here

`first_hlc_rho` was first reported as `DIFFERS`, and it is not.  Two mistakes
in the report, both mine:

- the verdict was decided by the **worst** event, so about a dozen tie-breaks
  out of 8144 condemned the whole row;
- the agreement column counted **bitwise** equality, and our rho goes through
  `np.hypot` where pass2's went through the oscNext project's `calc_rho_36`.
  The two differ in the last bit for roughly a fifth of events — a different
  square root, not a different definition.

The verdict is now the **fraction** of events agreeing within a tolerance, with
integers still held to exact equality.  This matters beyond cosmetics: the
wrong verdict sent the investigation after a variable that was already right.

## 6. Fitting the definitions that did not reproduce

For the two remaining failures the note gives one line each and the original
code is not available, so the parameters had been invented.  But guessing was
no longer necessary: **the pass2 L4 files hold the reference value and the
pulse series it came from in the same frame**, so many candidate definitions
can be computed per event and scored against the stored number — with no event
matching and no doubt about the input.

23 candidate definitions were tried.  None reproduced either variable.

So the question was **inverted**: the reference is known per event, so compute
what it would have to *mean* under our arithmetic.

**For `accumulated_time`** — the charge fraction accumulated at pass2's stored
time.  If they used the same entries and a fixed fraction, this returns that
fraction in every event.

| entry set | median | 16% | 84% |
|---|---|---|---|
| all pulses | 0.7258 | 0.6822 | 0.7449 |
| per-DOM charge | 0.7395 | 0.6903 | 0.8011 |

It sits just **below** 0.75 in at least 84% of events and never above — exactly
one entry's charge share short.  That is the signature of an **off-by-one**,
and the median shortfall (0.024) is an average pulse's share at this hit
multiplicity, so the magnitude matched as well as the sign.

**For `VICH_nch`** — whether pass2's count can fit inside our veto region at
all.  This is decisive rather than suggestive: no speed window, causality rule
or COG choice can raise a count above the number of veto DOMs that were hit.

```
pass2 count > DOMs hit in OUR veto region :  0.16% of events
median pass2 count 5.0, median veto DOMs hit 25.0
```

So the region is big enough and the selection inside it is what differs.


> **Both inversions pointed at the right knob for the wrong reason.**  Section 8
> reads the sign correctly and the cause wrongly: pass2 is not one entry short
> of 75%, it is binning into quartiles.  Section 9's ceiling test is sound and
> its conclusion — "the region is big enough" — turned out to be an
> understatement: there is no region at all.  Sections 7 and 8 below record
> what the production source then settled.

## 7. `accumulated_time` — SOLVED from the source, and it was never an off-by-one

The fit's best rule — all pulses, fraction 0.75, the entry **before** the
crossing, 99.40% against 0.98% for the crossing itself — was right, and the
explanation attached to it was not.  The production source says why.

`analysis/private/analysis/event_selection/CalculateVariables.cxx`, the C++
module the L4 tray calls as `CalculateVariables`, never looks for a 75%
crossing.  It bins the time-sorted pulses into **charge quartiles**:

```cpp
bin_edges = [0, Q/4, Q/2, 3Q/4, Q]
accumulated_charge += current_charge
part_index = index of the first edge >= accumulated_charge
} else if ( part_index == 3 ) {
    variables.accumulated_time = pulse.GetTime() - first_slc.t();
```

`part_index == 3` means `0.5Q < accumulated_charge <= 0.75Q`, and the line runs
for **every** pulse in that band, so what survives to the frame is the last
pulse whose cumulative charge has not passed 0.75Q.  "One entry before the
crossing" is not a bug in pass2 — it is what quartile binning gives.

Reading the source fixed three things no amount of fitting could reach:

- **the bound is `<=`, not `<`** — `lower_bound` returns `bin_edges[3]` when the
  cumulative charge lands exactly on `0.75Q`;
- **the events where pass2 stores exactly 0 are its struct default.**  Section 5
  guessed a default and guessed the reason wrongly: it argued that reaching 0
  needs one pulse carrying three quarters of the charge.  `Variables.h` says
  *"Default constructor (initialize all to zero)"*, and the real condition is
  that NO pulse falls in the Q3 band — one pulse must carry the cumulative sum
  from under `0.5Q` to over `0.75Q`, i.e. more than a **quarter** of the event's
  charge, which is common enough to explain the rate;
- **the module writes no `Variables` object at all** when the pulse map holds
  four or fewer DOMs, so neither Dunkman key reaches the frame.

Together these took the agreement from 99.42% to **99.84%** and the maximum
difference from 3927 ns to **286 ns**, over 56,301 events.

### The residual was measured, not explained away

The original sorts with `std::sort` and a comparator that reads only
`GetTime()`.  `std::sort` is not stable, so pulses sharing a time are left in an
unspecified order — and if two tied pulses carry different charge, which of them
is "the last in Q3" changes with that order.

That was a hypothesis, and it was written down as a fact before it was tested.
The test: the same quartile rule, computed three times on one pass2 L4 file,
8144 events, differing **only** in the secondary sort key.

| tie-break | agreement |
|---|---|
| smaller charge first | 99.89% |
| **stable (ours)** | **99.80%** |
| larger charge first | 99.67% |

**The three differ, so tied times exist and they decide the residual.  None
reaches 100%, so the order `std::sort` produced is not any simple rule we could
adopt.**

We keep the stable sort and do not chase the better number: 99.89% against
99.80% is 7 events in 8144 — 9 differing against 16, about 2σ on a single file.
Adopting an arbitrary tie-break to capture that is fitting to noise, and it
would trade a reproducible answer for one that is not.

A second hypothesis was tested and **eliminated**: summing the total charge in
map order rather than with `np.sum` (the original accumulates it in a first loop
over the map while the cumulative runs in time order) changed not one event.

### The decision this reversed

The note (Table 12) describes "the time to reach 75%", which agrees with pass2
in 0.98% of events.  `LowEnVariables`' own `TimeToSum` uses `>= fraction`, i.e.
the crossing, so the note's reading is the better-documented one — it is simply
not what pass2 produced.  **Reproducing the production now outranks matching its
documentation**, so the quartile rule is the default and
`--accumulated-time-note` follows the note.  The same reversal was applied to
`micro_count` (§5).

## 8. `VICH` — SOLVED (100.00%)

Section 8 of this record used to say VICH was not solved and that the fastest
route was the `tau_bdt` source.  That turned out to be exactly right, and it is
worth recording why measurement alone could not have finished the job.

**`tau_bdt` was found, and it is Python, not C++.**  The `oscNext_meta
V01-00-07` build carries `src/tau-bdt/python/I3CutL7Module.py` — note the hyphen
in the directory and the underscore in the import, which is why earlier searches
missed it.  That module is what `from icecube.tau_bdt import I3CutL7Module`
loads, and it is what the L4 tray runs.

There is no speed window and no veto region.  For each trigger with a matching
config id, the hit **closest in time to the trigger** is the reference; with
`dt = t_ref - t_hit` and `d` the distance to it, a hit counts when

```
d  < 750
dt > -5 d + 500
dt <  d / 0.3 + 150
dt >  d / 0.3 - 1850
```

Four conditions in the `(distance, dt)` plane.  `nch` is the number of DOMs with
at least one selected pulse, `npulses` the pulse count, `qtot` the charge sum
with no clamping.  Input: the uncleaned series, pulse by pulse.

Measured against the pass2 L4 files over 8144 events: `nch` **100.00%**,
`npulses` **100.00%**, `qtot` **100.00%**, median difference 0 on all three.

### Why the search took as long as it did

The three structural differences from the old implementation are all
unreachable by tuning, which is exactly what the ~12% plateau in §6 was telling
us:

- **no speed window** — four causality bands instead;
- **no veto DOM list.**  Restricting the same bands to the veto DOMs drops
  agreement from 100% to **19.4%**, so the region is implicit in the bands.
  Section 6's ceiling test concluded "the region is big enough"; the truth is
  that imposing it at all is the error;
- **the trigger, not a COG**, is the reference point.

The note warned of this in passing (p.60): *"some algorithms used throughout
their event selection may use differing definitions of what precisely
constitutes the veto region"*.  It was more than a differing definition.

### What the source settled that no measurement could

`TriggerConfigIDs` defaults to **`[1010, 1011]`** and the L4 tray does not
override it, where we had used 1011 alone.  pass2 simulation carries no 1010
trigger, so all 8144 events agree either way — **on detector data the first
matching trigger could be a 1010 and the reference pulse would differ.**  The
module also `break`s after the first matching trigger, where the GRECO variant
of the same bands loops over all of them and accumulates.  Neither fact is
visible in any pass2 measurement.

Three implementations of these bands exist and they do **not** agree with each
other: `I3CutL7Module.py` (the one L4 runs), GRECO's `VetoCausalHits` (no break,
accumulates), and `I3CutL7Module_JP_Matt.py` (breaks, 1011 only, charge only).
Where they differ, the module L4 actually calls decides.

One trap cost a full run: the **MERGED** and **THROUGHPUT** triggers carry no
config id, and reading it unguarded raises and silently loses every trigger in
the event — which is how the first attempt produced NaN in all 8144 events.

One deliberate deviation remains.  The original seeds its reference search with
`refPulseTime = 0`, `refPulsePos = (0,0,0)`, so an event with no pulse closer to
the trigger than `t = 0` would be computed against the origin.  A trigger at
~10 µs makes that unreachable, and it never occurred; reproducing it would only
copy a latent bug.

## 9. Two more bugs, found by reading rather than fitting

Neither of these was on any list of suspects.  Both were found by opening the
module the L4 tray names and reading it.

### `first_hlc` — ties, and the event with no HLC hit

`SimpleVertex/private/SimpleVertex/FirstHLC.cxx`, registered as
`I3_MODULE(FirstHLC<I3RecoPulse>)`, is what `tray.AddModule("FirstHLC<I3RecoPulse>", ...)`
instantiates.  (A Python class of the same name in `analysis/python/yanez/FirstHLC.py`
is a different author's module: its parameters are `InputPulseSeries`/`Vertex`
where the L4 tray passes `HitSeriesName`/`OutputName`.)

- **Ties go to the LAST DOM.**  The C++ reads `if (hitTime > hlc_time) continue;`
  — it skips only a strictly *later* hit, so a hit at exactly the current best
  time overwrites it and the highest OMKey among tied hits wins.  Our code kept
  the first.  That was the whole of the 146 m disagreements: a tie resolved to a
  different string.
- **An event with no HLC hit still gets a vertex.**  `GetFirstHLCHit` returns a
  bool that `Physics` ignores, so the search's starting value is written:
  position (1000, 1000, 1000), time 1e10 — a `first_hlc_rho` of **1407.3**.  The
  production's `fill_ratio` and muon BDT both saw that number where we wrote
  nothing at all.

Both fixed.  `first_hlc_rho` and `fill_ratio` went from 0.9987 and 0.9831 to
**bitwise identical** over 56,301 events.

This also explains a clue that had been sitting in the report unread: `fill_ratio`
disagreed **ten times more often** than `first_hlc_rho` did.  `rho` compares only
`sqrt((x-46.29)^2 + (y+34.88)^2)`, so a different DOM on the same string gives an
identical rho and a different z — and only `fill_ratio`, which is handed the
whole vertex, could see it.

### `separation_in_cogs` — the definition was a guess, and it was wrong

`CalculateVariables.cxx`: `variables.separation = CalcDistance(cog_q1, cog_q4)`,
with the quartiles accumulated in the same charge-quartile loop as
`accumulated_time`, and `Variables.h` documenting it as *"Distance between CoG_Q1
and CoG_Q4"*.

Our code split the event into two halves by **hit count** and compared those
COGs: a different split measure (count, not charge) and different parts (halves,
not the outer quartiles — the middle half of the charge excluded entirely).
Fixed.  It is not a BDT input, so it sits behind `--run-optional` and is not in
the comparison table; adding a row would verify it for free.

## 10. The final result

Five samples — NuE, NuMu, NuTau (160511), MuonGun, Noise — 100 L3 files each,
each file's own GCD, **56,301 events matched**.

**14 of 15 rows bitwise identical.**

| variable | role | result |
|---|---|---|
| `NchCleaned`, `ICVetoHits` | control — from L3 | identical |
| `iLineFit_speed` | control — original module | identical |
| `cog_z`, `z_sigma`, `z_travel`, `n_hit_doms` | control — original module | identical |
| `FullTimeLengthRatio` | rewritten | identical |
| `micro_count` | rewritten | identical |
| `fill_ratio` | original module, our vertex | identical |
| `first_hlc_rho` | rewritten | identical |
| `VICH_nch` / `npulses` / `qtot` | rewritten | identical |
| `accumulated_time` | rewritten | **99.84%**, max difference 286 ns |

Every control row is identical, so the two sides demonstrably saw the same
input and the rewritten rows mean what they say.

**Both classifiers' input lists are now fully verified**, and not against the
note: `L4_noise_model_train.py` and `L4_muon_model_train.py` in the fridge give
the 5 and 10 input names literally, subkey spellings included, and they are
exactly `NOISE_FEATURES` and `MUON_FEATURES`.  The single residual — 0.16% of
events on the sixth of ten muon inputs, bounded by one pulse spacing — is the
only known gap, and it is not reproducible in principle.

## 11. Still open

- **The `accumulated_time` residual is closed as far as it can go.**  Not
  reproducible; see §7.  No further work is warranted.
- **Training our own model on pass2** and comparing it, event by event, against
  `L4_NoiseClassifier_ProbNu` — pass2's own model's score, which sits in the
  same files.  That comparison is the real prize of this phase, and nothing
  blocks it any more.
- **The production trains the muon BDT only on events passing the noise cut**
  (`P_noise >= 0.70`), and we do not.  The muon classifier is not trained yet,
  so this is cheap to fix now.
- **The production's muon background is real detector data, not simulation** —
  their own note says *"we never got a decent CORSIKA set"* and *"we never
  actually used the MC trained muon classifier anyway"*.  This is a larger
  difference than the substitution it was once described as.

## 12. A note on the tooling

`oscnext_l4/fit_pass2.py` and the `fit` / `fit-report` subcommands of
`compare_pass2.py` were **temporary scaffolding**: they existed to find
definitions, nothing in the production path imported them, and they were deleted
once the findings were in `variables.py` and in `CLAUDE.md`.
`scripts/run_crosscheck.py` went with them — `pass2_verification.ipynb`
does the same job and pools events across files, which the script could not.

Removing them lost nothing, because everything they established is written down
here and in the docstrings.  If a definition ever has to be fitted again — a
pass3 variable that does not reproduce, `separation_in_cogs`, a new L4 variable
— the machine is one command away.  **The paths on the LEFT are where those
files lived in that tag, and they have not moved there; the paths on the right
are where they go today**, now that the cross-check lives in `verification/`:

```
git show pass2-verified-v1:oscnext_l4/fit_pass2.py   > verification/fit_pass2.py
git show pass2-verified-v1:scripts/compare_pass2.py  > verification/compare_pass2.py
```

(The second one would overwrite the current `compare_pass2.py`; take the `fit`
subcommands out of it rather than replacing the file wholesale, since it has
moved on since that tag.)

It is worth recovering rather than rewriting: it carries the
variant/inversion/grid-scan method that found VICH, and the `production_tie_*`
variants that measured the `accumulated_time` residual.

**What the method itself is worth remembering for.**  Measurement and source
reading did different jobs, and neither would have finished alone.  The fit
retired five VICH hypotheses and located `accumulated_time`'s rule; it could
never have told us that `TriggerConfigIDs` defaults to `[1010, 1011]`, that ties
go to the last DOM, or that an event with no HLC hit still gets a vertex at
(1000, 1000, 1000).  Reading the source settled all three in an afternoon —
**once the source was found.**  The plateau at ~12%, with the median difference
pinned at 2 DOMs in every cell of every sweep, was the signal to stop turning
knobs and go looking for it.
