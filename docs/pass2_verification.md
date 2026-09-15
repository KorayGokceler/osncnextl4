# Verifying the rewritten L4 variables against pass2

A step-by-step record of how the L4 variables this repository computes were
checked against the official oscNext pass2 production, what each step
established, and what is still open.

Everything below is a measurement on real files, not an inference from
documentation, unless it says otherwise.

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

**The L3 cut adapts by itself.**  `L3_oscNext_bool` and `Data_quality_bool`
appear nowhere in the pass2 note; they are the pass3 L3 script's own additions.
Our `l3_cut` falls through to `IC2018_LE_L3_bools["IC2018_LE_L3_Full"]`, which
pass2 does have.  Consequence: on pass2 the data-quality cut is not applied, so
the cut is slightly looser.  That changes *which* events arrive, not what any
variable evaluates to.

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

With the control rows all `identical`, the comparison became readable.

| variable | role | result |
|---|---|---|
| `NchCleaned`, `ICVetoHits` | from L3 | identical |
| `iLineFit_speed` | original module | identical |
| `cog_z`, `z_sigma`, `z_travel`, `n_hit_doms` | original module | identical |
| `micro_count` | rewritten | **identical** (with `--micro-count-uncleaned`) |
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
and never read.  Our default follows the note and therefore differs on purpose.

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

## 7. `accumulated_time` — SOLVED

The off-by-one was then tested directly, over a 96-cell sweep of entry set ×
fraction × rule:

| rule | agreement | median diff |
|---|---|---|
| all pulses, 0.75, the pulse **BEFORE** the crossing | **99.40%** | **0** |
| per-DOM charge, same rule | 54.25% | 0 |
| all pulses, fraction 0.70, at the crossing | 44.35% | 10 ns |
| all pulses, 0.75, **at** the crossing (what the note describes) | 0.98% | 52 ns |

**pass2 reports a pulse at which the event holds less than 75% of its charge**
— which is not "the time to reach 75%".  The zero point, long listed as the
suspect, was never the problem: `t[idx] − t[0]` is right.

**In the production code:** the default follows the note (at the crossing), as
it does for `micro_count`; `--accumulated-time-pass2` reproduces pass2.  The
two code paths were checked against each other on 4000 random events with 0
mismatches, and the production sort was made stable so tied pulse times land on
a reproducible index.

### What the flag is for

It is not a "pass2 mode".  It **isolates the difference**.  Turned on, we
reproduce pass2 at 99.40%, which proves every other part of the computation —
pulse series, charge handling, sorting, cumulative sum, fraction — matches the
original.  Turned off, we get the note's definition, knowing precisely and
quantitatively what the only remaining difference is.

That is what makes the pass3 production trustworthy: not that we copied the
original, but that we measured where we differ from it and why.

**Still a live decision for pass3:** follow the note, or follow what the
collaboration actually ran?  The two differ by ~52 ns on values of hundreds to
thousands of ns.  Whichever is chosen, **training and application must use the
same one** — `classifier.py` reads `L4_accumulated_time.value` without knowing
the convention, so a mismatch would be silent.  Note that `accumulated_time` is
a **muon** BDT input only, so the noise BDT is unaffected either way.

## 8. `VICH` — not solved, and the old justification was wrong

Open risk 1 recorded VICH as "largely verified" because sec. 3.4 states a speed
window of [0.25, 0.4] m/ns.  Reading that section in full (p.26-27, lines
483-493) shows it describes the **Level 2 DeepCore Filter** — a different
algorithm, which uses that window to **discard** hits:

> ...a relative velocity is derived between each hit of the veto region and the
> COG's position/time vertex.  If the derived speed of any veto hit is
> contained within [0.25, 0.4] m/ns, the hit is discarded on the basis that the
> veto hit could be causally related to a muon crossing the detector.

Our `_vich` borrowed that window as its **selection** rule.  The note's only
description of `L4_VICH_nch` is Table 12's one line — no window, no region, no
reference point.

Five hypotheses are now retired by measurement (8144 events):

| hypothesis | how it was tested | best agreement |
|---|---|---|
| the speed window | 36-cell sweep of both edges | 18.0% |
| the veto region | ceiling test, plus Table 7's L3 region | 11.4% |
| the COG | unweighted, non-fiducial | 12.0% |
| the reference point | the verified first HLC hit instead of the COG | 11.7% |
| the input series | cleaned instead of uncleaned | 2.8% |

Everything lands at ~12% with the **median difference pinned at 2 DOMs in
every variant**, which is what you see when the swept parameter is not the one
that differs: the algorithm is structurally different, not a parameter away.
The note itself warns (p.60, line 880) that "some algorithms used throughout
their event selection may use differing definitions of what precisely
constitutes the veto region".

**The fastest resolution is the `tau_bdt` source.**

## 9. Where this leaves the two classifiers

**Noise BDT — all five inputs verified.**

| input | result |
|---|---|
| `NchCleaned` | identical |
| `micro_count` | identical |
| `iLineFit_speed` | identical |
| `FullTimeLengthRatio` | identical |
| `fill_ratio` | 98.1%, the residual traced to last-bit vertex differences |

**Muon BDT — nine of ten inputs verified**, `VICH_nch` outstanding.  In the
note's own feature-importance figure `VICH_nch` is the least important of the
ten, while `first_hlc_rho`, the most important, is verified at 99.85%.

## 10. Still open

- `VICH_nch`: needs the `tau_bdt` source, or a structurally different idea.
- The `accumulated_time` convention for pass3 (see §7).
- Training our own model on pass2 and comparing it, event by event, against
  `L4_NoiseClassifier_ProbNu` — pass2's own model's score, which sits in the
  same files.  That comparison is the real prize of this phase and it is not
  blocked by VICH.

## 11. A note on the tooling

`oscnext_l4/fit_pass2.py` and the `fit` / `fit-report` subcommands of
`scripts/compare_pass2.py` are **temporary scaffolding**.  They exist to find
definitions, nothing in the production path imports them, and they are deleted
once the findings are in `variables.py` and in `CLAUDE.md`.  Everything worth
keeping lives in the code and in the documentation, so removing them loses
nothing.
