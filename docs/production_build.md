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

PENDING — what is in `src/`, which projects, which are absent from the modern
metaproject, and what the build's own metadata says about how it was made.

## 2. `oscNext_master.py` — the driver, live

PENDING — the GitHub port has L4 and L5 commented out of
`PROCESSING_LEVEL_TRAY_SEGMENTS` entirely, so it cannot run L4 at all.  The
live build must register them.  What else differs.

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

PENDING — `L4_model_data.py` (how the training table is harvested and what is
cut before training), `../tools/apply_model.py` (the model is applied to the
HDF5, not to frames), `L4_noise_model_train.py`, `L4_muon_model_train.py`.

**The open question this should settle:** how the FIRST model was bootstrapped.
`compute_L4_cut` is added unconditionally in the L4 segment and loads the
models from `CLASSIFIER_MODEL_DIR`, but training needs L4 variables, which
needs the L4 segment to have run.  Something breaks that circle and we do not
know what.

## 6. The trained models themselves

PENDING — `oscNext/resources/models/L4_noise_model.joblib` and its `.hdf5`
sidecar.  The sidecar should give the production's exact input list and any
scaling, which would be a stronger source than our reading of Tables 11/12.
Whether the model can be compared against ours event by event.

## 7. `oscNext_L5.py` — what comes after us

PENDING — which L4 keys L5 reads.  That decides what our own L4 `.i3` output
has to carry if the chain is ever extended past L4.

## 8. What this changes for us

PENDING — written last, from the sections above.
