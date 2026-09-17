# The production build, read at the source

What `oscNext_meta V01-00-07` actually does — the py2 build on cvmfs that
produced pass2 L1-L5, and the fridge scripts that trained its classifiers.

This is a **primary-source document**: everything in it is quoted from files
in that build, not inferred from the technical note and not carried over from
the commented-out GitHub port.  Where the build and the note disagree, the
build is what produced the numbers we are trying to reproduce.

    build  : /cvmfs/icecube.opensciencegrid.org/users/Oscillation/software/
             oscNext_meta/releases/V01-00-07/
    fridge : /data/sim/DeepCore/2018/workspace/fridge/processing/samples/
             oscNext/

> STATUS: being written.  Sections marked PENDING have not been read yet.
> `docs/pass2_verification.md` already covers what the L4 VARIABLES do and how
> they were verified; this document is about the WORKFLOW around them.

---

## 1. The build itself

`src/` holds about 150 projects.  Every one this repository had to rewrite
around is there:

    tau-bdt          the VICH module (I3CutL7Module)
    analysis         the Dunkman variables (CalculateVariables)
    SimpleVertex     FirstHLC<I3RecoPulse>
    slc-veto         the QR box
    static-twc       I3StaticTWC, for micro_count
    level3-filter-lowen   DeepCoreCuts -- see section 2, this is what made L3
    NoiseEngine, DeepCore_Filter, fill-ratio, linefit, tensor-of-inertia
    pybdt            the AdaBoost library the note's method predates

So the build is self-contained: it could run L1 to L7 on its own.  Nothing was
missing from it; the projects are missing from the MODERN metaproject, which is
the whole reason for `oscnext_l4/rewritten/`.

## 2. `oscNext_master.py` — the driver, live

The live build registers **every** level:

    PROCESSING_LEVEL_TRAY_SEGMENTS[3] = DeepCoreCuts
    PROCESSING_LEVEL_TRAY_SEGMENTS[4] = oscNext_L4
    PROCESSING_LEVEL_TRAY_SEGMENTS[5] = oscNext_L5
    PROCESSING_LEVEL_TRAY_SEGMENTS[6] = oscNext_L6
    PROCESSING_LEVEL_TRAY_SEGMENTS[7] = oscNext_L7

The GitHub port has 4 and 5 commented out of the registry itself, so it cannot
run L4 at all -- not merely the segment, the level's registration is gone.

`oscNext_L4.py` here is 576 lines with 123 comment lines, i.e. LIVE code.  The
copy in `reference/` (and in `icetray-oscNext/`) is 579 lines with 396 comment
lines: the same file with its body commented out.

### AND L3 IS NOT WHAT WE THOUGHT

Line 33 of the live driver reads `PROCESSING_LEVEL_TRAY_SEGMENTS[3] =
DeepCoreCuts`.  The GitHub port instead uses `oscNext_L3`, and has DeepCoreCuts
commented out -- **they are swapped between the two**.

This matters because CLAUDE.md's "Running on pass2" section concludes that the
pass2 L3 cut is exact, and it reasons from `oscNext_L3.py`:

> The official `oscNext_L3.py` writes both, and folds data quality INTO the
> bool we fall back on

But `oscNext_L3.py` is **not the file that produced pass2's L3**.  DeepCoreCuts
is, from `level3-filter-lowen`.  The conclusion may still hold -- the user's own
pass3 L3 script also uses GRECO's `DeepCoreCuts`, so pass2 and pass3 would then
share a lineage the oscNext_L3 story does not explain -- but the EVIDENCE for it
is currently the wrong file.  **Unresolved; read DeepCoreCuts and settle it.**

## 3. How a level was actually run

PENDING — the command line, and the cluster submission around it.  The
`--tmp-dir` and `--gridftp` flags say this ran per-file on a grid; the fridge
should carry the submission scripts that prove it and show the real
concurrency.

## 4. `oscNext_cuts.py` — what a "level cut" means

PENDING — `oscNext_cut(processing_level=N)` is called at the top of every
level segment.  Exactly which bool it reads and what it does to frames that
fail.

## 5. The L4 training chain, file by file

The fridge's `selection/level4/` holds more than the README's four steps
suggest:

    L4_model_data.py             harvest the training table          17 KB
    L4_noise_model_train.py      train the noise BDT                 4.9 KB
    L4_muon_model_train.py       train the muon BDT                  12 KB
    L4_noise_model_test.py       evaluate it                         5.9 KB
    L4_muon_model_test.py        evaluate it                         9.4 KB
    L4_noise_variable_search.py  CHOOSE the input variables          14 KB
    L4_noise_cuts.py             a straight-cut alternative          14 KB
    L4_noise_cuts_data.py                                            7.7 KB
    L4_noise_cuts_vs_model.py    straight cuts vs the BDT            14 KB
    compare_classifiers.py                                           2.6 KB
    investigate_first_hlc_rho_issue.py                               15 KB

`selection/tools/` holds `apply_model.py`, `classifier_tools.py`,
`classifier_plots.py`, `find_cuts.py`, `test_classifier.py`.

Two of these were not known before and are directly relevant to us:
**`L4_noise_variable_search.py`** is how the five noise inputs were chosen --
we took the list as given, and this is the reasoning behind it.
**`L4_noise_cuts_vs_model.py`** compares the BDT against straight cuts, which is
the argument for using a classifier at all.

PENDING — the files themselves.

**The open question this should settle:** how the FIRST model was bootstrapped.
`compute_L4_cut` is added unconditionally in the L4 segment and loads the
models from `CLASSIFIER_MODEL_DIR`, but training needs L4 variables, which
needs the L4 segment to have run.  Something breaks that circle and we do not
know what.

## 6. The trained models themselves

`oscNext/resources/models/` carries the real pass2 models, dated 30 July 2021:

    L4_noise_model.joblib          131 KB   the noise BDT
    L4_muon_model_data.joblib      214 KB   the one the sample actually used
    L4_muon_model_muongun.joblib   164 KB   the MC-trained one, unused
    L7_classifier_*.joblib                  later levels
    pid_model_*.joblib                      particle ID

Each `.joblib` has a matching `.hdf5` that is a **symlink** into
`/data/ana/LE/oscNext/pass2/resources/classifier_models/level4_noise/` -- so the
sidecars are real files on /data, readable.

There is also a 193-byte `README.txt` in that directory, unread.

PENDING — the sidecar's contents.  The sidecar should give the production's exact input list and any
scaling, which would be a stronger source than our reading of Tables 11/12.
Whether the model can be compared against ours event by event.

## 7. `oscNext_L5.py` — what comes after us

PENDING — which L4 keys L5 reads.  That decides what our own L4 `.i3` output
has to carry if the chain is ever extended past L4.

## 8. What this changes for us

PENDING — written last, from the sections above.
