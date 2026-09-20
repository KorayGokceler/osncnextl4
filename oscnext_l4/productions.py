"""
The productions this pipeline can be pointed at, and how to switch between
them.

WHY THIS EXISTS
---------------
The notebook used to carry the pass3 GCD and sample paths inline.  Running the
same pipeline over pass2 needs three things changed at once -- the GCD, the
paths, and the cleaned pulse series -- plus one that is easy to forget and
expensive to get wrong: **the unit of the vuvuzela noise weight**.  pass3
stores it in 1/ns, pass2 already in Hz.  Nothing raises when that is wrong; it
scales every noise rate by 1e9 and the only symptom is an absurd number much
later, in the training.

So the switch is one call, and it sets the unit as part of switching.  There is
no supported way to select pass2 and forget.

USAGE (notebook section 1)

    from oscnext_l4.productions import select
    GCD, SAMPLES = select("pass2", HDF_BASE)     # or "pass3"

Everything downstream -- booking, the registry check, loading, weights,
datasets, training -- is production-independent and needs no change.

WHAT DIFFERS BETWEEN THE TWO
----------------------------
Verified against real files and the technical note; the full account is in
`docs/pass2_verification.md`.

| | pass3 | pass2 |
|---|---|---|
| cleaned pulses | `SRTTWSplitInIcePulsesDC` | `SRTTWOfflinePulsesDC` |
| uncleaned pulses | `SplitInIcePulses` | the same |
| noise weight unit | 1/ns | Hz |
| L3 cut | `L3_oscNext_bool` (data quality ANDed by the pass3 script) | `L3_oscNext_bool` too -- `oscNext_L3.py` folds data quality INTO `IC2018_LE_L3_Full` and copies it there, so both branches of `l3_cut` agree and the cut is applied either way |
| `I3GenieInfo` | present | **absent** -- every event falls back to `NEvents * 70/30` |
| muon background | CORSIKA | MuonGun (there is no CORSIKA in these paths) |

The last two matter for weights, not for the BDT inputs, and neither touches
the noise classifier.  An earlier version of this table claimed pass2 drops
the data-quality cut; that was read out of the technical note and is wrong --
the production L3 script settles it.  See CLAUDE.md, "Running on pass2".
"""

import os
import glob


# ---------------------------------------------------------------------------
# pass3
# ---------------------------------------------------------------------------

PASS3_GCD = ("/cvmfs/icecube.opensciencegrid.org/data/GCD/"
             "GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz")

PASS3_SAMPLES = {
    "nue":     dict(l3="/data/ana/LE/oscNext/pass3/genie/level3/23800/*.i3.zst",
                    flags=["--mc", "--genie"], kind="signal", weight="genie"),
    "numu":    dict(l3="/data/ana/LE/oscNext/pass3/genie/level3/23799/*.i3.zst",
                    flags=["--mc", "--genie"], kind="signal", weight="genie"),
    "corsika": dict(l3="/data/ana/LE/oscNext/pass3/corsika/level3/23694/*.i3.zst",
                    flags=["--corsika"], kind="muon_bg", weight="corsika"),
    "noise":   dict(l3="/data/ana/LE/oscNext/pass3/noise/level3/23813/*.i3.zst",
                    flags=["--noise"], kind="noise_bg", weight="noise"),
}


# ---------------------------------------------------------------------------
# pass2
# ---------------------------------------------------------------------------

# The GCD that sits with the pass2 files.  That it is the RIGHT one is not
# assumed: cog_z / z_sigma / z_travel, which we compute from it and pass2
# stored from theirs, agree to the last bit over 8144 events.
PASS2_GCD = ("/data/ana/LE/oscNext/pass2/genie/level4/121122/"
             "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise"
             ".i3.gz")

PASS2_CLEANED_PULSES = "SRTTWOfflinePulsesDC"

_P2 = "/data/ana/LE/oscNext/pass2"
_GENIE_L3 = _P2 + "/genie/level3/%s/oscNext_genie_level3_v02.00_pass2.%s.*.i3.zst"

# NuE and NuMu are each TWO datasets at pass2, so `l3` is a list here.
#
# `weight` names the weighting SCHEME, not the sample -- that is what makes one
# code serve both productions.  Both productions have a `muon_bg`, but pass3's
# is CORSIKA and pass2's is MuonGun, and they need different input columns and
# a different formula.  Keying anything downstream on the SAMPLE NAME breaks the
# moment a production names its sets differently; keying on the scheme does not.
# ---------------------------------------------------------------------------
# The muon classifier's background: DETECTOR DATA, and exactly which runs
# ---------------------------------------------------------------------------
#
# Technical note v00.07 sec. 3.6.3 (p.39) names them, so this is not a choice
# of ours -- it is the production's list, quoted:
#
#   "For this classifier, detector data (following L4 noise cut) is used as the
#    background sample used for training, as it is 99% muons at this stage and
#    is considered more robust than using MuonGun MC at this processing level.
#    The signal training sample is still GENIE MC.  This also allows
#    unsimulated event populations (such as muon bundles) to be removed by the
#    classifier.  Note that a classifier was also trained using MuonGun MC for
#    the background sample, which achieved similar performance but was not used
#    in the sample."
#
#   "The background sample is comprised of the following data runs, which were
#    selected to cover years roughly equally in order to avoid strong
#    dependence of the muon rejection on the specific season due to muon flux
#    seasonal variations."
#
# Three per year, 2012-2017.  The even coverage is the POINT of the list: the
# atmospheric muon flux varies seasonally, so a background drawn from one part
# of the year teaches the classifier that season rather than the muon.
#
# TWO CONDITIONS COME WITH IT, and neither is optional:
#
#   1. "following L4 noise cut" -- the background is data that has ALREADY
#      passed the noise classifier at 0.7.  That is what makes it 99% muon.
#      Training on data that has not been through the noise cut trains the muon
#      BDT partly on noise, which the noise BDT has already removed.  The
#      note's own figures say the same in their cut box: Figures 19-21 all
#      carry `L4_NoiseClassifier_ProbNu > 0.7`.
#   2. The signal side stays GENIE MC.  This is a data-vs-MC classifier by
#      construction, not a data/data or MC/MC one.
#
# The fridge's `L4_model_data.py` says "one run per month 2012-2018", which is
# a LATER and larger list than the note's 18.  Where they differ the note is
# what the published pass2 numbers (Table 13) were produced with; the fridge
# script is the state of the code at a later date.
PASS2_MUON_DATA_RUNS_BY_YEAR = {
    2012: (120200, 120700, 121650),
    2013: (122650, 123250, 124650),
    2014: (125150, 125700, 126300),
    2015: (126850, 127400, 127850),
    2016: (128000, 128550, 129050),
    2017: (129650, 130150, 130700),
}

PASS2_MUON_DATA_RUNS = tuple(
    r for year in sorted(PASS2_MUON_DATA_RUNS_BY_YEAR)
    for r in PASS2_MUON_DATA_RUNS_BY_YEAR[year])

# Detector data L3 -- the layout is VERIFIED on disk, not assumed:
#
#   .../data/level3/IC86.12/Run00120200/
#       Level2pass2_IC86.2012_data_Run00120200_0527_1_20_GCD.i3.zst
#       oscNext_data_IC86.12_level3_v02.00_pass2_Run00120200_Subrun00000000.i3.zst
#       oscNext_data_..._Subrun00000000.hdf5
#       oscNext_data_..._Subrun00000000.json
#
# one directory per season, one per run inside it, and the season number is
# the year: IC86.NN holds the runs taken in 20NN.  A season carries about a
# thousand runs (IC86.12 has 1010); the note picked three of them per year.
#
# THE PATTERN CANNOT BE `*.i3.zst`.  Each subrun contributes THREE files and
# the run's GCD is `*_GCD.i3.zst`, so that glob would hand I3Reader a GCD as
# if it were an input and count three entries per subrun.  It also makes the
# file counts three times what they are: a directory listing of 634 entries is
# about 211 L3 files, and the note's 18 runs come to roughly 4,400 rather than
# the 13,125 a naive `ls | wc -l` suggests.
_P2_DATA_L3 = (_P2 + "/data/level3/IC86.%02d/Run%08d/"
                     "oscNext_data_*_level3_*_Run%08d_Subrun*.i3.zst")

# The GCD lives in the run directory, one per run -- detector data cannot use
# the averaged MC GCD, because the dead DOMs and the calibration are what
# changes from run to run.  Its name carries fields that vary per run
# (`_0527_1_20_` above), so it is GLOBBED rather than constructed.
_P2_DATA_GCD = _P2 + "/data/level3/IC86.%02d/Run%08d/*_GCD.i3.zst"

PASS2_MUON_DATA_L3 = [
    _P2_DATA_L3 % (year - 2000, run, run)
    for year in sorted(PASS2_MUON_DATA_RUNS_BY_YEAR)
    for run in PASS2_MUON_DATA_RUNS_BY_YEAR[year]]

PASS2_MUON_DATA_GCD = [
    _P2_DATA_GCD % (year - 2000, run)
    for year in sorted(PASS2_MUON_DATA_RUNS_BY_YEAR)
    for run in PASS2_MUON_DATA_RUNS_BY_YEAR[year]]


PASS2_SAMPLES_ALL = {
    "nue":     dict(l3=[_GENIE_L3 % ("121122", "121122"),
                        _GENIE_L3 % ("121291", "121291")],
                    flags=["--mc", "--genie"], kind="signal", weight="genie"),
    "numu":    dict(l3=[_GENIE_L3 % ("141154", "141154"),
                        _GENIE_L3 % ("141292", "141292")],
                    flags=["--mc", "--genie"], kind="signal", weight="genie"),
    "noise":   dict(l3=_P2 + "/noise/level3/888003/"
                            "oscNext_noise_level3_v02.00_pass2.888003.*.i3.zst",
                    flags=["--noise"], kind="noise_bg", weight="noise"),
    "nutau":   dict(l3=_GENIE_L3 % ("160511", "160511"),
                    flags=["--mc", "--genie"], kind="signal", weight="genie"),
    "muongun": dict(l3=_P2 + "/muongun/level3/139008/"
                            "oscNext_muongun_level3_v02.00_pass2.139008.*.i3.zst",
                    flags=["--mc", "--muongun"], kind="muon_bg",
                    weight="muongun"),
    # Real detector data -- the production's ACTUAL muon background, and the
    # only sample here with NO MC flag: no truth, no MC weight dict, nothing
    # to propagate.  process_L4.py needs no change for it; an empty `flags`
    # is exactly what "this is data" means to build_key_list.
    #
    # It shares the muon_bg role with MuonGun ON PURPOSE, so that a muon-BDT
    # run picks up whichever the production offers -- but the two must never
    # be STACKED: one is measured and one is simulated, and a model trained on
    # the union learns the difference between them as much as the physics.
    # Section 6 of the notebook picks one and says which.
    # `gcd` is a LIST here, one per L3 pattern, and it is the only sample with
    # the field: every MC set shares one averaged GCD, detector data cannot.
    # Nothing consumes it yet -- see the note below on how a run is processed.
    "data":    dict(l3=PASS2_MUON_DATA_L3, gcd=PASS2_MUON_DATA_GCD,
                    flags=[], kind="muon_bg", weight="data"),
}

# What a noise-BDT run needs, stated as ROLES rather than as names: every
# signal set, plus the noise background.  The muon background is the only thing
# it can skip.
#
# This used to be the tuple ("nue", "numu", "noise"), which was written when
# pass3 was the only production and pass3 has no NuTau set.  At pass2 NuTau
# (160511) EXISTS and is signal -- the production's own L4_model_data.py
# harvests 12xxxx, 14xxxx and 16xxxx alike -- so a name list silently dropped
# it.  Roles do not have that failure mode: a production that adds a signal set
# gets it, one that does not have it is unaffected.
NOISE_BDT_ROLES = ("signal", "noise_bg")
MUON_BDT_ROLES = ("signal", "muon_bg")


PRODUCTIONS = {
    "pass3": dict(gcd=PASS3_GCD, samples=PASS3_SAMPLES,
                  cleaned_pulses=None,          # the code's own default
                  noise_weight_unit="per_ns"),
    "pass2": dict(gcd=PASS2_GCD, samples=PASS2_SAMPLES_ALL,
                  cleaned_pulses=PASS2_CLEANED_PULSES,
                  noise_weight_unit="hz"),
}


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
