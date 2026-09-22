'''
L4 classifier application module -- the replacement for
icecube.oscNext.tools.classifier.I3Classifier.

The oscNext project is absent from the meta-project, so I3Classifier is not
available.  This module does the same job: load the model -> read the
variables out of the frame -> predict -> write an I3Double into the frame.

DEPENDENCIES: lightgbm + numpy, nothing else.
sklearn and joblib do NOT exist in the IceTray environment (py3-v4.4.2), so
the model is read in the native text format:

    L4_noise_model.txt    the LightGBM trees
    L4_noise_model.json   the feature list (ORDER MATTERS) + class map

Usage:

    from oscnext_l4.classifier import L4Classifier, add_L4_classifiers

    tray.Add(L4Classifier, "noise_clf",
             ModelFile="models/L4_noise_model.txt",
             OutputKey="L4_NoiseClassifier_ProbNu")

    # or both at once, with the cut:
    add_L4_classifiers(tray, "L4_cut", model_dir="models")
'''

import os
import json

import numpy as np

import sys
from icecube import icetray, dataclasses


# ---------------------------------------------------------------------------
# Reading variables out of a frame
# ---------------------------------------------------------------------------
#
# FEATURE_MAP maps the model's variable names ("NchCleaned", "cog_z",
# "micro_count", ...) to where they live in the FRAME
# ("IC2018_LE_L3_Vars"["NchCleaned"], ...), and COLUMN_ALTS gives the field
# names that change between versions.
#
# Both come from config/variables.json, the same row that carries the HDF5
# column data.load_sample reads.  That is what keeps training and application
# on the same quantity: they cannot be edited apart, because there is one row.
# Reading one column while the other reads another produces nonsense silently.
# data.check_feature_map() checks it.
#
# The hit-statistics keys carry the pass3 spelling (SRTTWSplitInIcePulsesDC...)
# on a pass2 run too -- deliberate, and explained in config/README.md.
from .varmap import FEATURE_MAP, COLUMN_ALTS                    # noqa: F401


def _get_col(obj, col):
    if hasattr(obj, "keys"):
        return float(obj[col]) if col in obj else None
    if col == "value":
        return float(obj.value)
    if hasattr(obj, col):
        return float(getattr(obj, col))
    return None


def read_feature(frame, name):
    '''Read one variable out of the frame; NaN when it is not there.'''
    loc = FEATURE_MAP.get(name)
    if loc is None:
        return np.nan
    key, col = loc
    if key not in frame:
        return np.nan
    obj = frame[key]
    for candidate in [col] + COLUMN_ALTS.get((key, col), []):
        try:
            v = _get_col(obj, candidate)
        except (KeyError, AttributeError, TypeError, ValueError):
            v = None
        if v is not None:
            return v
    return np.nan


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_file):
    '''
    Load the native LightGBM model plus its JSON sidecar.

    model_file is the .txt path; the .json next to it is found
    automatically.  Returns (booster, features, sidecar).
    '''
    import lightgbm as lgb

    if not os.path.exists(model_file):
        raise IOError("model file not found: %s" % model_file)

    booster = lgb.Booster(model_file=model_file)

    json_file = os.path.splitext(model_file)[0] + ".json"
    if os.path.exists(json_file):
        with open(json_file) as fh:
            sidecar = json.load(fh)
        features = list(sidecar["features"])
    else:
        # No sidecar -> trust the feature names stored in the model itself
        icetray.logging.log_warn(
            "l4_classifier: %s not found, taking the feature order from the booster"
            % json_file)
        sidecar = {}
        features = list(booster.feature_name())

    # Consistency check -- the most likely source of a silent error
    bn = list(booster.feature_name())
    if bn and bn != features:
        raise ValueError(
            "feature order mismatch!\n  booster: %s\n  json   : %s"
            % (bn, features))

    unknown = [f for f in features if f not in FEATURE_MAP]
    if unknown:
        raise KeyError(
            "variable(s) not defined in FEATURE_MAP: %s\n"
            "Add them to config/variables.json." % unknown)

    return booster, features, sidecar


# ---------------------------------------------------------------------------
# Tray modulu
# ---------------------------------------------------------------------------

class L4Classifier(icetray.I3ConditionalModule):
    '''Apply a trained LightGBM model frame by frame.'''

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("ModelFile", "path to L4_<tag>_model.txt", None)
        self.AddParameter("OutputKey", "frame key the I3Double is written to", None)
        self.AddParameter("MissingValue",
                          "value used for a missing variable "
                          "(NaN -> LightGBM handles it itself)", np.nan)
        self.AddParameter("SkipIfIncomplete",
                          "skip the frame when every variable is missing", False)
        self.AddOutBox("OutBox")

    def Configure(self):
        model_file = self.GetParameter("ModelFile")
        self.output_key = self.GetParameter("OutputKey")
        self.missing = self.GetParameter("MissingValue")
        self.skip_incomplete = self.GetParameter("SkipIfIncomplete")

        if not model_file or not self.output_key:
            raise ValueError("ModelFile and OutputKey are required")

        self.booster, self.features, self.sidecar = load_model(model_file)
        self.n_missing = {f: 0 for f in self.features}
        self.n_frames = 0

        print("L4Classifier [%s]" % self.output_key)
        print("  model    : %s" % model_file)
        print("  trees    : %d,  features: %d" %
              (self.booster.num_trees(), len(self.features)))
        if self.sidecar:
            print("  trained  : %s (lightgbm %s)" %
                  (self.sidecar.get("trained", "?"),
                   self.sidecar.get("lightgbm_version", "?")))
            print("  default cut: %s" % self.sidecar.get("default_cut", "?"))

    def Physics(self, frame):
        if self.output_key in frame:
            self.PushFrame(frame)
            return

        x = np.empty((1, len(self.features)), dtype=np.float64)
        n_nan = 0
        for i, f in enumerate(self.features):
            v = read_feature(frame, f)
            if not np.isfinite(v):
                self.n_missing[f] += 1
                n_nan += 1
                v = self.missing
            x[0, i] = v
        self.n_frames += 1

        if self.skip_incomplete and n_nan == len(self.features):
            self.PushFrame(frame)
            return

        # binary objective -> predict returns P(positive class) directly
        prob = float(self.booster.predict(x)[0])
        frame[self.output_key] = dataclasses.I3Double(prob)
        self.PushFrame(frame)

    def Finish(self):
        bad = {f: n for f, n in self.n_missing.items() if n > 0}
        if bad and self.n_frames:
            print("L4Classifier [%s] missing-variable report (%d frames):"
                  % (self.output_key, self.n_frames))
            for f, n in sorted(bad.items(), key=lambda kv: -kv[1]):
                frac = 100.0 * n / self.n_frames
                flag = "  <-- ALWAYS MISSING, the model output is invalid" if frac > 99.9 else ""
                print("    %-30s %7d (%5.1f%%)%s" % (f, n, frac, flag))


# ---------------------------------------------------------------------------
# Convenience segment
# ---------------------------------------------------------------------------

NOISE_KEY = "L4_NoiseClassifier_ProbNu"
MUON_KEY  = "L4_MuonClassifier_Data_ProbNu"
CUT_KEY   = "L4_Cut_Bool"


@icetray.traysegment
def add_L4_classifiers(tray, name, model_dir,
                       noise_cut=0.70, muon_cut=0.65, apply_cut=True):
    '''
    Add both classifiers and compute the combined L4 cut.

    The cut values are the v00.07 reference.  Pick your own model's optimum
    from the efficiency-vs-rejection plot train_L4_classifier.py writes.
    '''
    tray.Add(L4Classifier, name + "_noise",
             ModelFile=os.path.join(model_dir, "L4_noise_model.txt"),
             OutputKey=NOISE_KEY)

    tray.Add(L4Classifier, name + "_muon",
             ModelFile=os.path.join(model_dir, "L4_muon_model.txt"),
             OutputKey=MUON_KEY)

    if apply_cut:
        def overall_cut(frame):
            if NOISE_KEY not in frame or MUON_KEY not in frame:
                frame[CUT_KEY] = icetray.I3Bool(False)
                return True
            ok = (frame[NOISE_KEY].value >= noise_cut and
                  frame[MUON_KEY].value >= muon_cut)
            frame[CUT_KEY] = icetray.I3Bool(bool(ok))
            return True
        tray.Add(overall_cut, name + "_overall_cut")
