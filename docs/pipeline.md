# Pipeline — which file runs when

Two stages: **L3 to L4 processing** (needs IceTray, takes hours) and
**training** (no IceTray, takes minutes).  The interface is
`notebooks/oscNext_L4.ipynb`.

## Stage 1 — L3 to L4 processing

```
L3 .i3.zst files
   │
   ├─ scripts/scan_files.py    drop the corrupt files, write a healthy list
   │                           (ONCE per set; then --scan off)
   ▼
scripts/process_L4.py          driver: builds the tray, runs it, books
   │   │
   │   ├─ oscnext_l4.env       EVERY icecube import goes through here
   │   │
   │   ├─ oscnext_l4.variables tray segments:
   │   │      oscNext_L4                the main segment
   │   │      ├─ common      first_hlc, rho_36, FullTimeLengthRatio
   │   │      ├─ muon vars   ToI, iLineFit, VICH, accumulated_time
   │   │      ├─ noise vars  micro_count, fill_ratio
   │   │      └─ hit_stats   cog_z, z_sigma, z_travel, n_hit_doms
   │   │
   │   ├─ oscnext_l4.filescan  the pre-scan for corrupt input
   │   │
   │   └─ oscnext_l4.booker    add_booker: hdfwriter if present, else the
   │                           pytables fallback (for cvmfs)
   ▼
L4_output/hdf5/<sample>/L4_*.hdf5   + <output>.meta.json
```

`oscnext_l4.runner` drives this stage: `configure_runner`, `run_process`,
`run_all`, `run_process_parallel`, plus a live progress bar.  Section 1 of the
notebook calls it.

## Stage 2 — Training

```
L4_*.hdf5
   │
   ├─ oscnext_l4.data          REGISTRY/ALTS, dump_tables, check_registry,
   │                           check_feature_map, load_sample, add_weights
   ▼
numpy arrays  (notebook sections 4-5)
   │
   ▼
L4_output/ds/L4_<tag>_dataset.npz   (notebook section 6)
   │
   ▼
scripts/train_L4_classifier.py      LightGBM, Table 10 hyperparameters
   │
   ▼
L4_output/models/
   L4_<tag>_model.txt          LightGBM native text format
   L4_<tag>_model.json         feature order + metrics + metadata
   <tag>_cuts.png  <tag>_overtrain.png  <tag>_dist.png
   │
   ▼
oscnext_l4.classifier          the L4Classifier tray module
                               add_L4_classifiers(tray, ...) -> I3Double
```

## Tray module order (inside the `oscNext_L4` segment)

1. L3 cut (`l3_cut`) — `IC2018_LE_L3_Full AND Data_quality_bool`
2. `first_hlc` → `L4_FirstHLC`
3. `rho_36`, `FullTimeLengthRatio`
4. muon variables (`iLineFit`, VICH, `accumulated_time`, optional ToI)
5. noise variables (the `micro_count` chain, `fill_ratio`)
6. hit statistics (recomputed, because L3 deletes them)
7. the weight chain (`PropagateGenieInfo`)

## Helpers / diagnostics

| file | when |
|---|---|
| `setup_env.sh` | find the environment, open a shell, run one command, register a Jupyter kernel |
| `oscnext_l4/env.py` | every `icecube` import; `get_I3Tray`, `have_lightgbm` |
| `scripts/diagnose_env.py` | when something "will not import": what is and is not in the environment |
| `scripts/scan_files.py` | scan for corrupt `.i3.zst`, produce a `--good-list` |

## Reference (`reference/`) — never runs, it is source material

| file | what |
|---|---|
| `OscNext_v00.074_pass2_technical_note.pdf` | the official technical note (Tables 10-13) |
| `oscNext_L4_pass2_original.py` | the original L4 tray segment (Tom Stuttard) |
| `pass3_L3_process.py` | the user's actual pass3 L3 processing script |
