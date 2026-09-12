# Figures — what you have, what to fix, what is still missing

Drop files into `presentation/figures/` under **exactly** these names.  A
missing figure is replaced by a box naming it, so the deck compiles at every
stage.

## You already have these four

| name to save as | your plot | slide | note |
|---|---|---|---|
| `rates_pass2_pass3.png` | oscNext Event Rates: pass2 vs pass3 | 3 | **fix before use** — see below |
| `nue_spectrum_pass2_pass3.png` | NuE_CC energy spectrum (Level2) | 3 | **fix labels** |
| `noise_inputs.png` | L4_noise inputs, rate vs variable | 5 | ready |
| `pybdt_model_comparison.png` | six-model efficiency vs rejection | 7 | **re-export** — see below |
| `pybdt_score_dist.png` | BDT score distribution | 7 | **fix before use** |
| `adaboost_vs_lightgbm.png` | the green/dashed overlay (re-exported version) | 8 | ready |
| `lightgbm_score_dist.png` | LightGBM score, train/test, signal/background | 8 | ready -- train and test overlap, which is the overtraining evidence |

### Fixes needed

**`rates_pass2_pass3.png`** — the "IceCube Preliminary" watermark sits on top of
the Pass legend, and "Kategori" is Turkish.  Move the watermark to the upper
left or lower right, and rename the legend title to "Category" (or drop it).

**`nue_spectrum_pass2_pass3.png`** — the title and x label are Turkish
("enerji spektrumu", "Enerji [GeV]").  If the talk is in English, these need to
be "energy spectrum" / "Energy [GeV]".

**`pybdt_score_dist.png`** — as exported it is unusable: the y-axis label is cut
off and there is no legend, so nobody can tell which curve is signal-train and
which is background-test.  Re-export with `bbox_inches="tight"` and a legend.

**`pybdt_model_comparison.png`** — the aspect ratio is squashed flat (it looks
like only the bottom overlay panel got cropped out of the multi-panel figure).
Re-export just that panel at a normal size, e.g.
`figsize=(7, 4.5)`, or crop the full figure less aggressively.

## Still missing — worth producing

Ordered by value per unit effort.  Everything in tier 1 and 2 needs only the
HDF5 files, the `.npz` sets and the trained model you already have — no
reprocessing.

**Nothing is irrecoverable.**  pybdt training was measured to be deterministic
(the same configuration twice gives bit-identical scores), so every pybdt plot
you did not save can be reproduced exactly by re-running the same command.  The
numbers behind them are all in `CLAUDE.md`.

### Tier 1 — minutes of work

**`feature_importance.png` — slide 10.**  Horizontal bar chart of the gain
split.  The numbers are already in `L4_output/models/L4_noise_model.json` under
`importance_gain`.

```python
import json, matplotlib.pyplot as plt
imp = json.load(open("L4_output/models/L4_noise_model.json"))["importance_gain"]
tot = sum(imp.values())
k, v = zip(*sorted(imp.items(), key=lambda x: x[1]))
fig, ax = plt.subplots(figsize=(5, 2.4))
ax.barh(k, [100*x/tot for x in v], color="tab:green")
ax.set_xlabel("gain [%]"); fig.tight_layout()
fig.savefig("feature_importance.png", dpi=160)
```

**`lightgbm_cuts.png` — slide 9.**  Already written by the training script:

```bash
python scripts/train_L4_classifier.py --tag noise \
    --dataset L4_output/ds/L4_noise_dataset.npz --outdir L4_output/models
# -> L4_output/models/noise_cuts.png
```

**`input_correlation.png` — optional, slide 5 or 8.**  Correlation matrix of the
five noise inputs, straight from the `.npz`.  It is the natural companion to the
gain plot: if `iLineFit_speed` only gets 1.1% of the gain because it is
correlated with something else, this shows it, and that is a different
conclusion from "the variable is useless".

```python
import numpy as np, matplotlib.pyplot as plt
z = np.load("L4_output/ds/L4_noise_dataset.npz")
f = [str(x) for x in z["features"]]
X = np.column_stack([z[n] for n in f])
ok = np.isfinite(X).all(axis=1)
C = np.corrcoef(X[ok].T)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(f))); ax.set_xticklabels(f, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(f))); ax.set_yticklabels(f, fontsize=7)
for i in range(len(f)):
    for j in range(len(f)):
        ax.text(j, i, "%.2f" % C[i, j], ha="center", va="center", fontsize=6)
fig.colorbar(im); fig.tight_layout(); fig.savefig("input_correlation.png", dpi=160)
```

### Tier 2 — an hour, and each answers a question you will be asked

**`mc_scaling.png` — slide 9.  The highest-value missing plot.**
Efficiency at fixed rejection versus the fraction of background used.  This is
what turns "we need more vuvuzela MC" from an assertion into a measurement: if
the curve is still rising at 100% of the sample the request is justified, and if
it has flattened it is not.  `pybdt_diagnose.py` on branch
`claude/lightgbm-compare` already does this scan (with pybdt -- say so in the
caption, or swap `learner.train` for `lgb.train`).

**`incremental_features.png` — slide 5 or 8.**
Train with 1, 2, 3, 4, 5 variables (adding them in order of gain) and plot
efficiency at fixed rejection against the number of inputs.  It answers "do you
actually need all five" and, together with the gain plot, tells you whether
`iLineFit_speed` and `micro_count` are carrying anything at all.  The original
pass2 workflow suggested exactly this scan.

**An L4 point on your own rates plot — the best closing image.**
Your `rates_pass2_pass3.png` stops at Level 3.  Adding an "L4 noise cut" column
shows the noise rate falling from 41.4\,mHz toward the note's target of
$<$0.3\,mHz **on axes the audience has already read**.  Label it "noise cut
only" -- the muon BDT is not trained.

### From the technical note — for the comparison you asked about

| what | where | why |
|---|---|---|
| **Figure 13** (noise BDT input distributions) | note p.36-37 | **the single most valuable screenshot.** Put it beside your `noise_inputs.png` -- same five variables, same style. That side-by-side is the strongest validation image in the talk. |
| **Figure 12** | note p.36 | the remaining input distributions, same purpose |
| the L4 noise score distribution / cut figure | note §3.6.2 | compare with your `lightgbm_score_dist.png` |
| DeepCore geometry, or the selection-chain diagram | note §2-3 | `oscnext_levels.png`, slide 2 context |
| **Table 13** | note p.48 | you already quote the numbers; a screenshot is only worth it if someone challenges the rate comparison |

Figure 13 is worth doing properly: crop it to the same five panels in the same
order as your own plot, so the eye can move between them.

## Two context figures (no data needed)

| name | what it should show | how |
|---|---|---|
| `oscnext_levels.png` | the oscNext selection chain, or DeepCore geometry | screenshot from the technical note |
| `bdt_schematic.png` | trees in sequence, outputs summed into a sigmoid | any standard boosting schematic |

Both are optional.  Slide 2 works with its table alone, and slide 6 can lose
its figure — delete the `\plot` line and widen the left column.

## One plot you could add if the L4 result needs a closing image

Your `rates_pass2_pass3.png` stops at Level 3.  **Extending it with an
"L4-noise" point** would show the noise rate falling from 41.4\,mHz toward the
note's target of $<$0.3\,mHz on the same axes the audience has already read.
That is the single clearest way to show the result landing, and it reuses a
figure they have already understood.  It needs only the noise classifier, which
is trained — the muon BDT is not, so label the point "L4 noise cut only".

## Compiling

```bash
cd presentation
pdflatex oscnext_l4_noise_bdt.tex
pdflatex oscnext_l4_noise_bdt.tex     # twice, for the frame numbers
```

Speaker notes are in `\note{}` on every slide.  For note pages, add after
`\documentclass`:

```latex
\setbeameroption{show notes on second screen=right}
```
