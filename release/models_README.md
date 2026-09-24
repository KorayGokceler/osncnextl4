# models/

<!-- Template: scripts/make_release.py fills {{STATUS}}. -->

`process_L4.py --apply-cut --model-dir models` runs BOTH classifiers and needs
all four files:

| file | what |
|---|---|
| `L4_noise_model.txt` | the noise classifier, LightGBM's native text format |
| `L4_noise_model.json` | its sidecar: the feature ORDER, the training record, the provenance |
| `L4_muon_model.txt` | the muon classifier |
| `L4_muon_model.json` | its sidecar |

The `.json` is not optional: it fixes the order the inputs are handed to the
model in, and `process_L4.py` refuses to run without it.

They are produced by `scripts/train_L4_classifier.py --outdir models` (README,
section 2): noise first, then muon on the survivors of that noise model.

{{STATUS}}
