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
| `adaboost_vs_lightgbm.png` | the green/dashed overlay from earlier | 8 | ready |

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

Ordered by how much they add to the talk.

**1. `mc_scaling.png` — slide 9.  The highest-value missing plot.**
Efficiency at fixed rejection versus the *fraction of background used*.  This
is the plot that answers "do we actually need more vuvuzela MC" with a trend
line instead of an assertion — if it is still rising at 100% of the sample, the
request is justified; if it has flattened, it is not.

```bash
git checkout claude/lightgbm-compare
python pybdt_diagnose.py --ds-dir L4_output/ds --tag L4_noise \
    --features NchCleaned,micro_count,iLineFit_speed,fill_ratio,FullTimeLengthRatio
```

`pybdt_diagnose.py` already does this scan.  It trains pybdt, not LightGBM --
mention that in the caption, or port the loop to LightGBM if you have time
(it is the same loop with `lgb.train` in place of `learner.train`).

**2. `lightgbm_cuts.png` — slide 9.**
The LightGBM cut curve with the background-exhaustion lines at 100 / 10 / 1
events.  Already written by the training script:

```bash
python scripts/train_L4_classifier.py --tag noise \
    --dataset L4_output/ds/L4_noise_dataset.npz --outdir L4_output/models
# -> L4_output/models/noise_cuts.png
```

**3. `feature_importance.png` — slide 8.**
A horizontal bar chart of the gain split.  The numbers are already in
`L4_output/models/L4_noise_model.json` under `importance_gain`; the training
script prints them but does not plot them.  Ten lines of matplotlib:

```python
import json, matplotlib.pyplot as plt
imp = json.load(open("L4_output/models/L4_noise_model.json"))["importance_gain"]
tot = sum(imp.values())
k, v = zip(*sorted(imp.items(), key=lambda x: x[1]))
fig, ax = plt.subplots(figsize=(5, 2.6))
ax.barh(k, [100*x/tot for x in v], color="tab:green")
ax.set_xlabel("gain [%]"); fig.tight_layout()
fig.savefig("feature_importance.png", dpi=160)
```

The finding is worth a chart rather than a table: the model leans on
`fill_ratio`, whose `SphericalRadiusMean=1.6` the original code itself says was
"optimised for GRECO and not re-optimised for oscNext".

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
