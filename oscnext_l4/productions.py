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
}

# What a noise-BDT run needs: the two signal sets and the noise background.
# Kept for a deliberately narrow run; it is NOT the pass2 default any more.
# The muon BDT is now reachable at pass2 (VICH is verified at 100.00% and
# MuonGun has a weighter), so a pass2 run loads everything unless asked not to.
PASS2_NOISE_BDT = ("nue", "numu", "noise")


PRODUCTIONS = {
    "pass3": dict(gcd=PASS3_GCD, samples=PASS3_SAMPLES,
                  cleaned_pulses=None,          # the code's own default
                  noise_weight_unit="per_ns"),
    "pass2": dict(gcd=PASS2_GCD, samples=PASS2_SAMPLES_ALL,
                  cleaned_pulses=PASS2_CLEANED_PULSES,
                  noise_weight_unit="hz"),
}


def select(production, hdf_base, samples=None, count_files=True):
    """
    Point the pipeline at one production -> (GCD, SAMPLES).

    production : "pass3" or "pass2"
    hdf_base   : where the HDF5 output goes; each sample gets its own
                 subdirectory, so two productions never overwrite each other.
    samples    : which samples to include (default: all of them).  Pass
                 PASS2_NOISE_BDT to leave out the sets the noise classifier
                 does not use.
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

    out = {}
    for name in names:
        cfg = dict(spec["samples"][name])
        cfg["flags"] = list(cfg["flags"])
        if spec["cleaned_pulses"]:
            cfg["flags"] += ["--cleaned-pulses", spec["cleaned_pulses"]]
        cfg["hdf5"] = os.path.join(hdf_base, name, "L4_%s.hdf5" % name)
        out[name] = cfg

    print("production: %s" % production)
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
