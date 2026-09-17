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

### `L4_noise_model_train.py`, read

Short and entirely declarative.  What it settles:

**The five inputs, verbatim:**

    L4_NOISE_MODEL_INPUT_VARIABLES = [
        "IC2018_LE_L3_Vars.NchCleaned",
        "L4_micro_count.STW_m3500p4000_DTW200",
        "L4_iLineFit.speed",
        "L4_fill_ratio.fill_ratio_from_mean",
        "IC2018_LE_L3_Vars.FullTimeLengthRatio" ]

**The hyperparameters**, which match `train_L4_classifier.PARAMS` to the digit,
`feature_fraction 0.8` included.  `algorithm = "lightgbm"` -- the same engine,
not a coincidence we have to defend.  `class_ratio = {k: 1.}` with
`scale_weights=True`, and `seed=12345`.

**THE WEIGHT IS READ, NOT COMPUTED.**

    weight_key = "I3MCWeightDict.final_weight"

The production does not weight at training time.  It reads a `final_weight`
column that its own weighting chain (`frame_objects/weighting.py` ->
`FINAL_WEIGHT_KEY`) wrote earlier.  Everything `data.genie_weight` does --
OneWeight, the power law, NEvents * gen_ratio, the file count -- happened
upstream, once, and was stored.

*This is an opportunity, not a problem.*  pass2's `I3MCWeightDict` already gave
us `gen_ratio` and the two agreed exactly; if it also carries `final_weight`,
our `w_phys` can be compared against the production's own number event by
event.  That would close open risk 4 by measurement.  **Check whether
`final_weight` is in the booked columns.**

**The train fractions, with the reasoning:**

    "neutrino" : 0.02   # Was 0.3333 for older datasets (1.95e5 training
                        # events).  1% with the new nominal dataset (0000)
                        # gives a comparable 2.24e5.  Doubling to 2% to
                        # address overtraining.
    "noise"    : 0.3333 # We doubled the livetime for noise dataset 888003
                        # since the original training, so this gives ~twice
                        # as many training events now.  Keeping the fraction
                        # rather than halving it, to increase stats, since
                        # there was some overtraining previously.

Note what the second comment says: **888003 is the noise set we are processing**,
and its livetime was doubled after the original training.  So the noise
statistics available to us are larger than the ones behind the released model.

### `apply_model.py`, read

    python apply_model.py -d <data.hdf5> -m <model.joblib> -k <key>

It loads the classifier, **backs the data file up** (`.backup.hdf5`), reads it,
predicts, adds one column per class named `<key>_<class>_pred`, and rewrites
the same file.  Frames are never involved.

So in the training workflow the model is applied to the TABLE, not to `.i3`.
The tray-side application (`compute_L4_cut` in the L4 segment) is a separate
path, for producing L4 files that L5 can consume.

### `L4_model_data.py`, read — and it answers the bootstrapping question

    processing_stage = "level4"

The training table is harvested **from L4 files**, so L4 processing ran first.
The circle is broken by ITERATION across model versions, not by a special first
pass: the script carries `--special-case legacy`, documented as *"what was used
for the oscNext L4 v01.01 classifier training"*.  So a v01.01 model existed,
L4 files were produced with it, and the next generation was trained on those.
The very first model is not reconstructible from what is here, and the `legacy`
switch is the archaeology of the second.

**The datasets it harvests — and this does not match ours:**

    genie_dataset = "0000"        # "Default nominal set"

    12%s -> 120000    nu_e     CLASSES["neutrino"]
    14%s -> 140000    nu_mu    CLASSES["neutrino"]
    16%s -> 160000    nu_tau   CLASSES["neutrino"]
    888003            noise    CLASSES["noise"]
    detector data, one run per month 2012-2018    CLASSES["traindata"]

We train on **121122, 121291, 141154, 141292, 160511**; the released model was
trained on **120000, 140000, 160000**.  The noise set 888003 is the same.
Whether 12/14/16-0000 exist in the pass2 paths we have is **unchecked**, and
whether the generation parameters differ from the 12xxxx sets we use is
**unknown**.  This is the first real difference between our training and theirs
that is not a deliberate choice.

**ν_τ is in the signal**, as three sets sharing one class label -- which is what
our move to role-based sample selection already reproduces.

**The muon background is detector data, with the reasons stated:**

    #TODO Have removed muon MC for now since (a) we don't have post SLC bug fix
    # full detector MuonGun MC, (b) we never got a decent CORSIKA set and
    # (c) we never actually used the MC trained muon classifier anyway
    # (we used the data driven one instead for the sample)

`use_corsika = False`, and the MuonGun block is commented out.  So our MuonGun
muon BDT is not the production's -- theirs is data-driven.  The noise BDT is
unaffected: it selects only the `neutrino` and `noise` classes.

**A train/test leak the author documents in their own code:**

    "subsample" : "test",
    #TODO "test" subsample only select runs numbered ????50 (e.g. 2% of runs),
    # but also a few of our training runs end in 50 so there is a small and
    # undesirable overlap

Their detector-data test sample overlaps their training runs slightly.  Not our
problem -- we use no detector data -- but it matters when comparing our held-out
numbers against theirs.

### What is harvested, and the variable search behind it

The live list is 16 variables plus `TRUTH_VARIABLES`: the five noise inputs, the
ten muon inputs, and two more --

    "L4_NoiseStraightCuts_Bool.value",
    "L4_NoiseClassifier_ProbNu.value",

The second is **the previous generation's own classifier score**, harvested from
the L4 files.  That is the iteration of section 5 made visible: the L4 files
being harvested already carry a model's output, and `apply_model.py` then writes
the new one beside it.

Below the live list sits the **whole set of variables they tested**, commented
out and kept "for posterity".  Two of its notes are facts we did not have:

    "IC2018_LE_L3_Vars.NchCleaned",
    # Not using this, identical to `n_hit_doms` from hit multiplicity

  -- `NchCleaned` and `n_hit_doms` are THE SAME QUANTITY.  We carry both, and
  `n_hit_doms` is in `REGISTRY` as a candidate; it is not an independent one.

    # "IC2018_LE_L3_Vars.NoiseEngine",
    # Not using this ... is a bool that has been cut on so is single valued
    # and thus has zero power

  -- the general point is worth keeping: anything already cut on at L3 is
  constant in the surviving sample and carries no information.

The tested list also shows `fill_ratio` under BOTH spellings
(`fill_radius_from_mean` and `fillradius_from_nch` in the same block), which is
exactly why `data.ALTS` has to carry several.

### The weight, finally traced

The last block of the script is where `final_weight` comes from:

    from icecube.oscNext.frame_objects.weighting import WEIGHT_DICT_KEY, FINAL_WEIGHT_KEY
    weight_key = WEIGHT_DICT_KEY + "." + FINAL_WEIGHT_KEY

so `I3MCWeightDict.final_weight` is `WEIGHT_DICT_KEY` + `FINAL_WEIGHT_KEY`, both
constants of the production's own weighting module.  **`FINAL_WEIGHT_KEY`'s
value is not yet known**, and it matters: our dump of a pass2
`I3MCWeightDict` shows `weight` and `weight_no_osc` but no column literally
called `final_weight`.  If `FINAL_WEIGHT_KEY == "weight"`, then the production's
final weight is already in the files we are booking, and `data.genie_weight` can
be compared against it event by event -- the open-risk-4 measurement, available
immediately.

Then, for classes that have a livetime, the weight is **overwritten**:

    for this_classification, livetime_s in livetime_dict.items():
        class_mask = classification == this_classification
        weights[class_mask] = 1. / livetime_s      # 1 count, per livetime

Whether `livetime_dict` covers only detector data or the noise simulation as
well is not visible here (it comes from `harvest_i3_files`).  It matters,
because vuvuzela IS generated over a fixed livetime, and our own measurement
already found the noise weights to be exactly uniform at `1/N`.  If the
production does the same, that uniformity is not an artefact of our weighting
but a shared property.  **Unresolved.**

### `final_weight` is just the file division -- and that opens a question

`weighting.py` settles the constants and the arithmetic:

    WEIGHT_DICT_KEY      = "I3MCWeightDict"
    WEIGHT_KEY           = "weight"
    UNMERGED_WEIGHT_KEY  = "unmerged_weight"
    MERGED_WEIGHT_KEY    = "merged_weight"
    FINAL_WEIGHT_KEY     = "final_weight"

    weight_dict[UNMERGED_WEIGHT_KEY] = weight_dict[WEIGHT_KEY]          # 228
    weight_dict[MERGED_WEIGHT_KEY]   = unmerged / num_merged_files      # 231
    weight_dict[WEIGHT_KEY]          = weight_dict[MERGED_WEIGHT_KEY]   # 234
    ...
    final_weight = weight_dict[WEIGHT_KEY] / float(num_files)           # 182
    weight_dict[FINAL_WEIGHT_KEY] = final_weight                        # 185

So `final_weight` is the stored `weight` divided by the file count -- exactly the
division `data.py` does with `_n_files`.  And line 190 does the same for the ML
power law:

    weight_dict[NU_SINGLE_POWERLAW_STEM+"_final_weight"] = \
        weight_dict[NU_SINGLE_POWERLAW_WEIGHT_KEY] / float(num_files)

**pass2's L3 files carry `SinglePowerLawFlux_weight`** -- our own dump showed it.
So the production's un-divided ML training weight is already in the files we are
booking, and `w_phys` can be compared against
`SinglePowerLawFlux_weight / n_l3_files` directly.

**BUT THERE IS A DISCREPANCY TO RESOLVE FIRST.**  `L4_noise_model_train.py`
trains with

    weight_key = "I3MCWeightDict.final_weight"

which is the PHYSICS weight (atmospheric flux, oscillations), **not**
`SinglePowerLawFlux_final_weight`.  Our `data.genie_weight` applies the single
power law `NORM * E^-3`, on the grounds -- recorded in CLAUDE.md -- that
`weighting.py` adds it *"for unbiased samples for training machine learning
algorithms"*.  Both cannot be what the noise BDT was trained on.

Either `WEIGHT_KEY` already holds the single power law for these datasets, or
the noise classifier was trained on physics-weighted events and our E^-3 choice
is a real deviation that changes the training set's shape.  **This is the most
consequential open question in this document.**  It is settled by reading how
`weight_dict[WEIGHT_KEY]` is first filled, above line 228 of `weighting.py`.

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

The 193-byte `README.txt` says what the `.hdf5` files are:

> The HDF5 files containing the **train/test data** for the classifier models
> stored in this directory can be found in
> `/data/ana/LE/oscNext/pass2/resources/classifier_models` on the Madison
> datastore.

So the sidecars are not metadata -- they are **the actual training and test
events** the released models were fitted on.  That is stronger than expected:
our training set can be compared against theirs directly, and our model can be
evaluated on their held-out test set.

PENDING — their contents.  The sidecar should give the production's exact input list and any
scaling, which would be a stronger source than our reading of Tables 11/12.
Whether the model can be compared against ours event by event.

## 7. `oscNext_L5.py` — what comes after us

PENDING — which L4 keys L5 reads.  That decides what our own L4 `.i3` output
has to carry if the chain is ever extended past L4.

## 8. What this changes for us

PENDING — written last, from the sections above.
