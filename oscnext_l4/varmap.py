"""
The variable table: config/variables.json -> the views the code uses.

WHY THIS FILE EXISTS
--------------------
There used to be SIX hand-written tables spread over two modules, all keyed by
the same variable names: REGISTRY / ALTS / AUX / AUX_SCHEMES in data.py and
FEATURE_MAP / COLUMN_ALTS in classifier.py.  Editing one and forgetting the
other is the failure CLAUDE.md calls the critical synchronisation point: if
training reads one column and frame application another, the model produces
wrong predictions and DOES NOT RAISE.

The old defence was a checker that parsed classifier.py with AST (because that
module needs icetray, which a plain kernel does not have) and compared the two
dicts.  Now there is ONE row per variable carrying BOTH sides, so the two
cannot be edited apart -- and the check costs a dict comparison instead of 95
lines of AST.

This module imports nothing but the standard library on purpose: classifier.py
runs inside icetray and data.py needs pytables, so neither can be the home of a
table the other one reads.

Rationale for the individual spellings is in config/README.md.
"""

import os
import json

CONFIG_PATH = os.environ.get("OSCNEXT_L4_VARIABLES") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "variables.json")

with open(CONFIG_PATH) as _fh:
    CONFIG = json.load(_fh)

VARIABLES = CONFIG["variables"]

NOISE_FEATURES = list(CONFIG["bdt_features"]["noise"])
MUON_FEATURES = list(CONFIG["bdt_features"]["muon"])
KIND_TO_SCHEME = dict(CONFIG["kind_to_scheme"])
SCHEMES_WITHOUT_AUX = tuple(CONFIG["schemes_without_aux"])


def _is_aux(entry):
    """A WEIGHT column rather than a BDT input: it declares its schemes."""
    return "schemes" in entry


# --- the HDF5 side (training / loading) ---------------------------------

# variable -> (HDF5 table, column).  Both BDT variables and weight columns are
# in the one file, so they are split here by whether the entry has `schemes`.
REGISTRY = {n: tuple(e["hdf5"]) for n, e in VARIABLES.items() if not _is_aux(e)}
AUX = {n: tuple(e["hdf5"]) for n, e in VARIABLES.items() if _is_aux(e)}

# variable -> every (table, column) that names the SAME quantity, best first.
# Column names differ between meta-project and pass versions; whichever one is
# actually in the file is used.  Only variables that have variants appear.
ALTS = {n: [tuple(e["hdf5"])] + [tuple(p) for p in e["hdf5_alts"]]
        for n, e in VARIABLES.items() if e.get("hdf5_alts")}

# weight column -> which schemes need it (SAMPLES[...]["weight"]).
AUX_SCHEMES = {n: tuple(e["schemes"]) for n, e in VARIABLES.items()
               if _is_aux(e)}


# --- the frame side (tray application) ----------------------------------

FEATURE_MAP = {n: tuple(e["frame"]) for n, e in VARIABLES.items()
               if "frame" in e}

COLUMN_ALTS = {tuple(e["frame"]): list(e["frame_alts"])
               for e in VARIABLES.values() if e.get("frame_alts")}


# --- the consistency check ----------------------------------------------

def check(verbose=True):
    """
    Does each variable's HDF5 side still name the same quantity as its frame
    side, and does every BDT input have both?

    Returns the list of conflicts (empty = consistent).  Needs no icetray and
    no HDF5 file -- it reads the config and nothing else.
    """
    conflict = []
    for name in sorted(VARIABLES):
        entry = VARIABLES[name]
        if "frame" not in entry or _is_aux(entry):
            continue
        h = {tuple(entry["hdf5"])} | {tuple(p) for p in entry.get("hdf5_alts", [])}
        tbl, col = entry["frame"]
        f = {(tbl, col)} | {(tbl, c) for c in entry.get("frame_alts", [])}
        if not (h & f):
            conflict.append((name, tuple(entry["hdf5"]), tuple(entry["frame"])))

    for name in sorted(set(NOISE_FEATURES) | set(MUON_FEATURES)):
        entry = VARIABLES.get(name)
        if entry is None:
            conflict.append((name, "ABSENT from the variable table", None))
        elif "frame" not in entry:
            conflict.append((name, tuple(entry["hdf5"]), "no `frame` side"))

    if verbose:
        if conflict:
            print("[!] CONFLICT -- the two sides of a variable disagree.")
            print("    Training reads one column, frame application the other;")
            print("    the model produces nonsense without raising:")
            for name, a, b in conflict:
                print("    %-22s hdf5=%s  frame=%s" % (name, a, b))
        else:
            both = sum(1 for e in VARIABLES.values()
                       if "frame" in e and not _is_aux(e))
            print("variable table: no conflict (%d variables with both sides, "
                  "%d weight columns)" % (both, len(AUX)))
    return conflict
