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

### Tier 1 — one command

`presentation/make_figures.py` produces all of these.  It needs only numpy and
matplotlib -- no lightgbm, no icetray -- so it runs in a plain python as well
as inside env-shell:

```bash
cd ~/l4/osncnextl4
python presentation/make_figures.py
```

It writes into `presentation/figures/` and reports what it could and could not
find, so it is safe to run before the training has been redone:

| figure | from | slide |
|---|---|---|
| `feature_importance.png` | `L4_noise_model.json` → `importance_gain` | 10 |
| `input_correlation.png` | `L4_noise_dataset.npz` | 5 or 8 |
| `lightgbm_cuts.png` | copied from `noise_cuts.png` | 9 |
| `lightgbm_score_dist.png` | copied from `noise_dist.png` | 8 |

The last two are written by the training, so if they are reported missing:

```bash
python scripts/train_L4_classifier.py --tag noise \
    --dataset L4_output/ds/L4_noise_dataset.npz --outdir L4_output/models
python presentation/make_figures.py          # then copy them across
```

Options: `--tag muon` once the muon classifier exists, `--out-root` if
`OSCNEXT_OUT_ROOT` is not set, `--figures` to write somewhere else.

**On the correlation matrix.**  It uses a **Spearman rank** correlation, not
Pearson, and plots signal and background in separate panels.  Both choices
matter for the question it answers.  Pearson would be misleading here --
`iLineFit_speed` spans three decades, `micro_count` and `NchCleaned` are small
integers, `fill_ratio` piles up near zero -- and a pair can be correlated in one
class while being independent in the other.

Read it next to the gain plot: if `iLineFit_speed` is strongly correlated with a
variable that took most of the gain, its 1.1\% means *redundant*.  If it is
uncorrelated with everything, 1.1\% means *weak*.  Those are different
conclusions and only this plot separates them.

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
