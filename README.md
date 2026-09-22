# oscNext L4 — Level 3 to Level 4

The **L3 to L4** step of the IceCube oscNext (low energy neutrino) analysis.
It computes the L4 discriminating variables from L3 `.i3` files, books them to
HDF5, and trains two LightGBM classifiers:

- **noise** — rejection of pure noise (vuvuzela)
- **muon** — rejection of atmospheric muons (background: CORSIKA)

Reference: oscNext technical note v00.07 (sections 3.4-3.6, Tables 10-13).
The method is the note's own: LightGBM with the Table 10 hyperparameters.

## Layout

```
oscnext_l4/     the library
                  tray side      variables, rewritten, frame_objects, l3vars,
                                 tray_io, env       (need icetray)
                  analysis side  data              (needs pytables/numpy)
                  both sides     varmap            (stdlib only)
                  and            runner (drives process_L4 as a subprocess),
                                 classifier (applies the model back to frames)
config/         productions.json, variables.json + README.md -- PURE DATA.
                The two facts that fail SILENTLY when wrong live here.
scripts/        entry points     process_L4, train_L4_classifier,
                                 scan_files, diagnose_env, plot_inputs,
                                 check_leakage, inspect_production_table
notebooks/      oscNext_L4.ipynb -- the interface, sections 0-10
verification/   the pass2 cross-check -- NOT part of the pipeline.  Runs it
                over the pass2 L3 files and holds the result against the real
                pass2 L4 files.  Delete this arm LAST; it is the safety net.
docs/           pipeline.md, technical_note_comparison.md, production_build.md
presentation/   figures and the weekly updates
reference/      the technical note and first-hand source material
```

The library splits by what a module must IMPORT, not by what it does: the tray
side and the analysis side never import each other.  They meet at two places
only -- the HDF5 file (the data) and `config/variables.json` (the naming).

## Setup

**1. The IceTray environment.** `icecube.*` imports only work inside an
IceTray build's `env-shell.sh`.  Search order: `$OSCNEXT_I3_BUILD` →
`$I3_BUILD` → `/data/user/$USER/icetray_build/build` →
`/data/user/$USER/*/build` → `~/*/build` → cvmfs metaprojects.

```bash
./setup_env.sh find          # what is found
./setup_env.sh shell         # open a shell inside the environment
./setup_env.sh run python scripts/diagnose_env.py    # one command
```

> **Trap:** `env-shell.sh` **opens a new shell**.  Writing `./env-shell.sh`
> and `python ...` on consecutive lines of a script runs the second line
> *after* leaving that shell, i.e. without the environment.  For a single
> command use `./env-shell.sh -- python ...` or `./setup_env.sh run python ...`.

**2. lightgbm.** The only ML dependency of training and application.

```bash
./setup_env.sh run python -c "import lightgbm; print(lightgbm.__version__)"
```

**3. Jupyter.** The only way the notebook sees IceTray is for the kernel to be
the python inside env-shell.  Starting Jupyter from inside the environment is
cleanest:

```bash
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)
cd <build_directory> && ./env-shell.sh
cd ~/l4/osncnextl4 && python -m jupyter lab --no-browser --port=8896
```

Otherwise register the kernel with `./setup_env.sh kernel` and select it in
the notebook.

> **Trap:** Jupyter's working directory **does not change with a kernel
> restart** — it comes from the server.  If you see `returncode=2` and a path
> like `~/.local/share/Trash/...`, the server is in the wrong directory: stop
> it and restart from the right one.  The notebook locates the repository root
> itself and says so if it cannot; `OSCNEXT_L4_ROOT` overrides the search.

## Usage

Everything runs from `notebooks/oscNext_L4.ipynb`:

| # | what it does |
|---|---|
| 0 | configuration + environment check |
| 1 | L3 to L4 processing (`process_L4.py`, plus a smoke test) |
| 2 | booking check — what is actually in the HDF5 |
| 3 | feature registry + `FEATURE_MAP` ↔ `REGISTRY` consistency |
| 4 | HDF5 to numpy |
| 5 | weights (`w_phys`) |
| 6 | `.npz` training sets + the train/test split |
| 7 | training (`train_L4_classifier.py`) |
| 8 | validation — efficiency/rejection table, overtraining gap, plots |
| 9 | choosing the cut |
| 10 | applying the model to frames |

It also works from the command line:

```bash
# processing
python scripts/process_L4.py --input-list nue_good.txt \
    --output-hdf5 L4_output/hdf5/nue/L4_nue.hdf5 --gcd <GCD> --scan off

# training
python scripts/train_L4_classifier.py --tag noise \
    --dataset L4_output/ds/L4_noise_dataset.npz --outdir L4_output/models
```

## Known traps

**Corrupt input files.** The pass3 production contains half-written
`.i3.zst` files, and since `I3Reader` takes the whole list at once, one
corrupt file **kills the entire tray**.  `process_L4.py` protects in two
layers: a pre-scan (`--scan quick`, the default) and a run-time retry
(`--retries 3`).  Dropped files are listed in `<output>.hdf5.badfiles.txt`.
Scanning once per set with `scan_files.py --good-list` and then using
`--scan off` is the efficient route.

**`--n` counts frames, not events.** Getting 60 events from `--n 200` is
normal: the stream carries G/C/D/Q/P frames, only some P frames match
`--sub-event-stream`, and only some of those pass the L3 cut.  The output
shows every stage.

**Module cache.** If `cannot import name ... from oscnext_l4.data` appears
after a `git pull`, the module in memory is stale.  Section 0 of the notebook
enables `%autoreload 2`, so it should not recur; if it does, Kernel → Restart.

**Speedups are not automatic.** The default is a single process.  For
parallelism use `run_all(jobs=8, chunk_files=10)`.  cobalt is a shared
machine — `jobs=8` is reasonable, `jobs=64` is not.

## Before committing the notebook

```bash
pip install --user nbstripout && nbstripout --install
```

Output cells take megabytes and produce meaningless diffs.

## Status

- [x] Environment verified, IceTray/lightgbm import layer
- [x] Robust against corrupt input files
- [x] Column names pinned down (14/14 BDT inputs found)
- [x] nue and CORSIKA processed
- [x] Noise classifier trained — 95.9% efficiency at 99% rejection
      (Table 13: ~96%)
- [ ] numu / noise need reprocessing (cut short by corrupt `.i3.zst`)
- [ ] Muon classifier not trained
- [ ] The rewritten variables (VICH, accumulated_time) not verified against
      the reference
- [ ] Noise MC statistics are inadequate — nothing right of 99% is measurable

Details and open risks: `CLAUDE.md`.  Line-by-line comparison with the
technical note: `docs/technical_note_comparison.md`.  Data flow:
`docs/pipeline.md`.
