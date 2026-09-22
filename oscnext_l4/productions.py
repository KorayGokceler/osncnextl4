"""
Which production the pipeline is pointed at, and how to switch between them.

THE FACTS LIVE IN `config/productions.json`; THIS FILE IS THE LOADER.
Every value there is explained in `config/README.md`, which is where the
reasons went when the tables became JSON -- JSON cannot carry comments, and
several of those values fail SILENTLY when wrong (the vuvuzela weight unit
scales every noise rate by 1e9; the detector-data GCD glob dropped a whole
season). Read the README before editing the JSON, and keep the two in step.

    from oscnext_l4.productions import select
    GCD, SAMPLES = select("pass2", HDF_BASE)     # or "pass3"

Everything downstream -- booking, the registry check, loading, weights,
datasets, training -- is production-independent and needs no change.

A different config:

    export OSCNEXT_L4_CONFIG=/path/to/my_productions.json
"""

import os
import json
import glob


# Repo root / config / productions.json, unless told otherwise.  The package
# may be imported from anywhere, so the path is resolved from THIS file rather
# than from the working directory.
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "productions.json")

CONFIG_PATH = os.environ.get("OSCNEXT_L4_CONFIG") or DEFAULT_CONFIG

# Fields without which a sample cannot be used.  Checked at load time and
# named individually: a missing `weight` would otherwise surface much later as
# "no weighting function for scheme None", and a missing `kind` would make a
# sample silently invisible to every role-based selection.
_SAMPLE_REQUIRED = ("flags", "kind", "weight")
_PRODUCTION_REQUIRED = ("gcd", "samples", "noise_weight_unit")


def _expand_runs(name, spec):
    """
    A sample given as runs + templates -> (expanded spec, {year: runs}).

    The runs come back separately rather than inside the spec; see below.

    Detector data is one pattern PER RUN, and its GCD is another: written out,
    the run numbers would appear in two lists that can drift apart.  One
    substitution -- `{run:08d}` and `{season:02d}`, where season = year - 2000
    -- keeps them in a single place.  This is the only templating the config
    has, and it is deliberately not a language.
    """
    if "runs_by_year" not in spec:
        return spec, None
    out = dict(spec)
    runs = out.pop("runs_by_year")
    try:
        l3_t = out.pop("l3_template")
    except KeyError:
        raise KeyError("sample %r has `runs_by_year` but no `l3_template` "
                       "(%s)" % (name, CONFIG_PATH))
    gcd_t = out.pop("gcd_template", None)

    pairs = [(int(y), int(r)) for y in sorted(runs, key=int) for r in runs[y]]
    out["l3"] = [l3_t.format(season=y - 2000, run=r) for y, r in pairs]
    if gcd_t:
        out["gcd"] = [gcd_t.format(season=y - 2000, run=r) for y, r in pairs]
    # The runs do NOT go back into the sample spec.  That spec is handed
    # straight to the runner and out of select(), and an extra key there is an
    # extra key everything downstream sees -- the expansion should leave no
    # trace beyond the lists it produced.  The caller keeps them separately.
    return out, {int(y): tuple(runs[y]) for y in runs}


def load_config(path=None):
    """Read and validate the config -> the parsed dict, samples expanded."""
    path = path or CONFIG_PATH
    if not os.path.exists(path):
        raise IOError(
            "production config not found: %s\n"
            "Set OSCNEXT_L4_CONFIG, or check out config/productions.json."
            % path)
    with open(path) as fh:
        cfg = json.load(fh)

    for key in ("productions", "bdt_roles"):
        if key not in cfg:
            raise KeyError("%s has no top-level %r" % (path, key))

    for pname, prod in cfg["productions"].items():
        for key in _PRODUCTION_REQUIRED:
            if key not in prod:
                raise KeyError("production %r has no %r (%s)"
                               % (pname, key, path))
        prod.setdefault("cleaned_pulses", None)
        for sname, spec in list(prod["samples"].items()):
            spec, runs = _expand_runs("%s.%s" % (pname, sname), spec)
            if runs:
                prod.setdefault("runs_by_year", {})[sname] = runs
            missing = [k for k in _SAMPLE_REQUIRED if k not in spec]
            if "l3" not in spec:
                missing.append("l3 (or runs_by_year + l3_template)")
            if missing:
                raise KeyError("sample %s.%s has no %s (%s)"
                               % (pname, sname, ", ".join(missing), path))
            prod["samples"][sname] = spec
    return cfg


CONFIG = load_config()

PRODUCTIONS = CONFIG["productions"]

# What a noise-BDT run needs, stated as ROLES rather than as names: every
# signal set, plus the noise background.  See config/README.md -- this used to
# be a name list and it silently dropped pass2's NuTau.
NOISE_BDT_ROLES = tuple(CONFIG["bdt_roles"]["noise"])
MUON_BDT_ROLES = tuple(CONFIG["bdt_roles"]["muon"])

# Names kept for the modules that already import them (oscnext_l4/pass2.py).
PASS2_CLEANED_PULSES = PRODUCTIONS["pass2"]["cleaned_pulses"]
PASS2_SAMPLES_ALL = PRODUCTIONS["pass2"]["samples"]
PASS2_GCD = PRODUCTIONS["pass2"]["gcd"]
PASS3_GCD = PRODUCTIONS["pass3"]["gcd"]
PASS3_SAMPLES = PRODUCTIONS["pass3"]["samples"]

# The note's 18 detector-data runs, by year -- quoted and explained in
# config/README.md.
PASS2_MUON_DATA_RUNS_BY_YEAR = PRODUCTIONS["pass2"].get(
    "runs_by_year", {}).get("data", {})


def select(production, hdf_base, samples=None, roles=None, count_files=True):
    """
    Point the pipeline at one production -> (GCD, SAMPLES).

    production : "pass3" or "pass2"
    hdf_base   : where the HDF5 output goes; each sample gets its own
                 subdirectory, so two productions never overwrite each other.
    roles      : keep only samples with these `kind`s -- the portable way to
                 say what a run needs, since the roles are the same in both
                 productions while the names are not:

                     select("pass2", HDF_BASE, roles=NOISE_BDT_ROLES)

                 picks up pass2's NuTau automatically, where a name list would
                 have to be edited per production.
    samples    : an explicit name list, when you really do mean these sets.
                 Combined with `roles` it is an intersection.
    count_files: glob each pattern to report how many L3 files were found.
                 Turn it off on a slow filesystem.

    Sets the noise weight unit as part of switching, and says so.  That is the
    point of the function: the unit is a per-production fact that does not
    raise when wrong.
    """
    from .data import set_noise_weight_unit

    if production not in PRODUCTIONS:
        raise ValueError("production must be one of %s, not %r"
                         % (sorted(PRODUCTIONS), production))
    spec = PRODUCTIONS[production]

    names = list(samples) if samples else list(spec["samples"])
    unknown = [n for n in names if n not in spec["samples"]]
    if unknown:
        raise ValueError("%s has no sample(s) %s -- it has %s"
                         % (production, unknown, sorted(spec["samples"])))

    if roles:
        roles = tuple(roles)
        known = {c.get("kind") for c in spec["samples"].values()}
        unknown_roles = [r for r in roles if r not in known]
        if unknown_roles:
            raise ValueError("%s has no sample with role(s) %s -- it has %s"
                             % (production, unknown_roles, sorted(known)))
        names = [n for n in names if spec["samples"][n].get("kind") in roles]

    out = {}
    for name in names:
        cfg = dict(spec["samples"][name])
        cfg["flags"] = list(cfg["flags"])
        if spec["cleaned_pulses"]:
            cfg["flags"] += ["--cleaned-pulses", spec["cleaned_pulses"]]
        cfg["hdf5"] = os.path.join(hdf_base, name, "L4_%s.hdf5" % name)
        out[name] = cfg

    print("production: %s" % production)
    if roles:
        print("  roles          : %s" % ", ".join(roles))
    print("  GCD            : %s" % spec["gcd"])
    if spec["cleaned_pulses"]:
        print("  cleaned pulses : %s" % spec["cleaned_pulses"])
    set_noise_weight_unit(spec["noise_weight_unit"])
    print("  HDF5 base      : %s" % hdf_base)
    print("")

    for name, cfg in out.items():
        if count_files:
            n = 0
            for pat in (cfg["l3"] if isinstance(cfg["l3"], (list, tuple))
                        else [cfg["l3"]]):
                n += len(glob.glob(pat))
            cfg["n_l3_files"] = n
            print("  %-8s %-10s %6d L3 files" % (name, cfg["kind"], n))
        else:
            print("  %-8s %-10s" % (name, cfg["kind"]))

    missing = [n for n, c in out.items() if c.get("n_l3_files") == 0]
    if missing:
        print("\n  [!] no L3 file found for: %s" % ", ".join(missing))
        print("      check the paths before running anything.")
    return spec["gcd"], out
