'''
oscNext Level 4 variable computation -- a working version.

The original tray segment (oscNext_level4.py) was commented out in full and
depended on three old projects:

    tau_bdt.I3CutL7Module        -> the VICH variables
    analysis.event_selection     -> the Dunkman variables
    slc-veto SmallQ_Box          -> the QR box

None of these is generally present in a modern IceTray meta-project.  The VICH
and Dunkman variables are rewritten here in pure Python from their definitions
in the technical note; the QR box is left optional (it is not a BDT input).

Reference: oscNext technical note v00.07, sections 3.4-3.6, Tables 11-12.
'''

import os
import sys
import numpy as np

from .env import (require_icetray, optional_project, require_project,
                  load_deserialization_libs, deepcore_doms,
                  load_lib)

require_icetray()
from icecube import dataclasses, icetray

# --- Opsiyonel projeler -----------------------------------------------------
# Deliberately NOT imported hard at module level: one missing project (say
# tensor_of_inertia absent from your own build) must not make the whole
# repository unimportable.  A missing one raises an explicit error when the
# segment that produces its variable is called; everything else keeps working.
DomTools          = optional_project("DomTools")
linefit           = optional_project("linefit")
tensor_of_inertia = optional_project("tensor_of_inertia")
fill_ratio        = optional_project("fill_ratio")

# Required for deserialisation (even though they are not used directly)
load_deserialization_libs()


# ---------------------------------------------------------------------------
# Output frame object names  (identical to the original script)
# ---------------------------------------------------------------------------

L4_CUT_BOOL_KEY = "L4_Cut_Bool"

L4_FIRST_HLC_KEY      = "L4_first_hlc"
L4_FIRST_HLC_RHO_KEY  = L4_FIRST_HLC_KEY + "_rho"

L4_TOI_KEY            = "L4_ToI"
L4_LINEFIT_KEY        = "L4_iLineFit"
L4_QRBOX_KEY          = "L4_QR_Box"

L4_VICH_NCH_KEY       = "L4_VICH_nch"
L4_VICH_NPULSES_KEY   = "L4_VICH_npulses"
L4_VICH_QTOT_KEY      = "L4_VICH_qtot"

L4_SEP_IN_COGS_KEY    = "L4_separation_in_cogs"
L4_ACC_TIME_KEY       = "L4_accumulated_time"

L4_FTLR_KEY           = "L4_FullTimeLengthRatio"
L4_NFLUX_KEY          = "L4_n_flux_events"   # I3GenieInfo'dan P frame'e tasinir
L4_MICROCOUNT_KEY     = "L4_micro_count"
L4_FILL_RATIO_KEY     = "L4_fill_ratio"

L4_NOISE_STRAIGHT_CUT_KEY         = "L4_NoiseStraightCuts_Bool"
L4_NOISE_MODEL_PREDICTION_KEY     = "L4_NoiseClassifier_ProbNu"
L4_MUON_MODEL_PREDICTION_DATA_KEY = "L4_MuonClassifier_Data_ProbNu"

# ---------------------------------------------------------------------------
# Pulse serisi isimleri
#
# oscNext L3 (online_filterscripts .. grecovariables.DeepCoreCleaning) girdi
# olarak "SplitInIcePulses" alir ve "SRTTWSplitInIcePulsesDC" uretir.
# (The old pass1 name was "SRTTWOfflinePulsesDC" -- NOT that any more.)
# ---------------------------------------------------------------------------
UNCLEANED_PULSES_DEFAULT = "SplitInIcePulses"
CLEANED_PULSES_DEFAULT   = "SRTTWSplitInIcePulsesDC"

# HitStatistics / HitMultiplicity (the muon BDT's cog_z, z_sigma, z_travel)
#
# NOTE: L3 computes these and then DELETES them (the cleanup list entry
# "<name>_DeepCoreCutsSRTTWSplitInIcePulsesDCHitStatistics").  So the
# oscNext_L4_hit_statistics segment RECOMPUTES them -- same pulse series, same
# module, therefore the same result.
HITSTAT_KEY  = CLEANED_PULSES_DEFAULT + "HitStatistics"
HITMULT_KEY  = CLEANED_PULSES_DEFAULT + "HitMultiplicity"

# Static/dynamic time window parameters used for micro count
STW_MINUS = 3500.
STW_PLUS  = 4000.
DTW       = 200
# The original script's format string is kept verbatim -> "STW_m3500p4000_DTW200"
# (Older GRECO code looked for "STW7500_DTW200" for the same variable -- they
# do not match.)
MICROCOUNT_SUBKEY = "STW_m%ip%i_DTW%i" % (STW_MINUS, STW_PLUS, DTW)

# Fill ratio radius multiplier -- optimised for GRECO and NEVER re-optimised
# for oscNext.  A parameter worth revisiting: it carries ~61% of the noise
# model's gain.
FILL_RATIO_SPHERICAL_RADIUS_MEAN = 1.6

# The VICH speed window of technical note sec. 3.4 is NOT part of the
# algorithm -- see _vich.  That passage describes the Level 2 DeepCore Filter.
# Kept only because oscnext_l4/fit_pass2.py imports them while it still exists.
VICH_SPEED_MIN = 0.25
VICH_SPEED_MAX = 0.40

# Position of string 36 (the DeepCore centre) -- fallback for calc_rho_36
STRING36_X = 46.29
STRING36_Y = -34.88


L4_HDF5_KEYS = [
    L4_CUT_BOOL_KEY,
    L4_FIRST_HLC_KEY, L4_FIRST_HLC_RHO_KEY,
    L4_TOI_KEY, L4_TOI_KEY + "Params",
    L4_LINEFIT_KEY, L4_LINEFIT_KEY + "Params",
    L4_QRBOX_KEY,
    L4_VICH_NCH_KEY, L4_VICH_NPULSES_KEY, L4_VICH_QTOT_KEY,
    L4_SEP_IN_COGS_KEY, L4_ACC_TIME_KEY,
    L4_MICROCOUNT_KEY, L4_FILL_RATIO_KEY, L4_FTLR_KEY, L4_NFLUX_KEY,
    L4_NOISE_STRAIGHT_CUT_KEY,
    L4_NOISE_MODEL_PREDICTION_KEY,
    L4_MUON_MODEL_PREDICTION_DATA_KEY,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LC_FLAG = dataclasses.I3RecoPulse.PulseFlags.LC


def iter_map(pulse_map):
    '''
    Iterate over (omkey, pulses) pairs.

    CAREFUL: depending on the IceTray version, iterating an
    I3RecoPulseSeriesMap directly may yield KEYS rather than pairs.  Then
    "for omkey, pulses in pmap" fails with "too many values to unpack
    (expected 2)", because an OMKey unpacks into three components
    (string, om, pmt).  .items() is correct in every version.
    '''
    if pulse_map is None:
        return []
    try:
        return pulse_map.items()
    except AttributeError:
        return iter(pulse_map)


def get_pulses(frame, key):
    '''Pulse serisini al; mask/union ise frame'e uygula.'''
    if key not in frame:
        return None
    obj = frame[key]
    if hasattr(obj, "apply"):
        try:
            return obj.apply(frame)
        except Exception:
            return None
    return obj


def calc_rho_36(x, y):
    '''
    Horizontal radial distance from string 36 (the DeepCore centre).

    VERBATIM from the official project, oscNext/frame_objects/geom.py:

        return np.sqrt( (x-46.29) ** 2 + (y+34.88) ** 2 )

    The constants were already right; the SQUARE ROOT was not.  We used
    np.hypot, which is a different algorithm (it rescales to avoid overflow)
    and disagrees with the naive form in the last bit.  Measured against pass2:
    our rho matched bitwise in 80.6% of 8144 events and to 1e-6 in 99.85% --
    the gap was this, not a different definition.  Using their form makes it
    exact.
    '''
    return float(np.sqrt((x - STRING36_X) ** 2 + (y - STRING36_Y) ** 2))


def iter_hits(pulse_map, geometry, first_pulse_only=False):
    '''
    (omkey, pos, time, charge) uzerinde iterasyon.
    first_pulse_only=True ise DOM basina sadece ilk pulse.
    '''
    omgeo = geometry.omgeo
    for omkey, pulses in iter_map(pulse_map):
        if omkey not in omgeo:
            continue
        pos = omgeo[omkey].position
        if first_pulse_only:
            if len(pulses):
                p = pulses[0]
                yield omkey, pos, p.time, p.charge
        else:
            for p in pulses:
                yield omkey, pos, p.time, p.charge


def charge_weighted_cog(hits):
    '''Charge weighted centre of gravity (position + time).'''
    xs, ys, zs, ts, qs = [], [], [], [], []
    for _, pos, t, q in hits:
        xs.append(pos.x); ys.append(pos.y); zs.append(pos.z)
        ts.append(t); qs.append(max(q, 0.0))
    if not xs:
        return None
    q = np.asarray(qs, dtype=float)
    if q.sum() <= 0:
        q = np.ones_like(q)
    w = q / q.sum()
    return (float(np.dot(w, xs)), float(np.dot(w, ys)),
            float(np.dot(w, zs)), float(np.dot(w, ts)))


def check_object_exists(frame, object_key):
    '''Drop the frame when the object is absent (the original's
    data_quality.check_object_exists).'''
    return object_key in frame


# ===========================================================================
# 1. COMMON VARIABLES
# ===========================================================================

def _first_hlc(frame, pulses_key, output_key, geometry_key="I3Geometry"):
    '''
    Find the earliest HLC hit in the cleaned series and write it as an
    I3Particle.

    The original script used the "FirstHLC<I3RecoPulse>" C++ module.  That
    module is not in every meta-project, so the same job is done in Python:
    HLC hits carry the I3RecoPulse.PulseFlags.LC flag.
    '''
    if output_key in frame:
        return True
    pmap = get_pulses(frame, pulses_key)
    if pmap is None or geometry_key not in frame:
        return True
    omgeo = frame[geometry_key].omgeo

    best = None   # (time, pos)
    for omkey, pulses in iter_map(pmap):
        if omkey not in omgeo:
            continue
        for p in pulses:
            if not (p.flags & _LC_FLAG):
                continue
            if best is None or p.time < best[0]:
                best = (p.time, omgeo[omkey].position)
            break   # DOM basina ilk HLC pulse yeterli

    if best is None:
        return True

    part = dataclasses.I3Particle()
    part.pos = best[1]
    part.time = best[0]
    part.shape = dataclasses.I3Particle.ParticleShape.Cascade
    part.fit_status = dataclasses.I3Particle.FitStatus.OK
    frame[output_key] = part
    return True


def _add_rho_36(frame, particle_key, output_key):
    if particle_key in frame and output_key not in frame:
        pos = frame[particle_key].pos
        frame[output_key] = dataclasses.I3Double(calc_rho_36(pos.x, pos.y))
    return True


def _full_time_length_ratio(frame, output_key,
                            l3_key="IC2018_LE_L3_Vars",
                            cleaned_pulses=None, uncleaned_pulses=None):
    '''
    Ratio of cleaned to uncleaned event duration.  A noise BDT input.

    Technical note Table 11: "measure the total duration of the event in both
    the cleaned and the uncleaned pulse series (duration = max pulse time -
    min pulse time) and take their ratio."

    Direction: cleaned / uncleaned, i.e. within [0, 1] -- consistent with the
    x axis of Figure 13.

    MEASURED BEHAVIOUR (on one L3 file each: 126 nue, 17 noise).  The physics
    story in the earlier comment was WRONG and is corrected here:

                          ratio (median)   cleaned     uncleaned
        nue                        0.16     1626 ns     10100 ns
        noise (passing L3)         0.27     2780 ns     10290 ns

      * The ratio never approaches 1.  The uncleaned duration is ~10 us in
        every event, because SplitInIcePulses spans the whole readout window
        and noise hits are everywhere.  So the variable is effectively
        "cleaned duration / 10 us".
      * The separating direction is the OPPOSITE of what one expects: the
        cleaned series of noise events is LONGER than that of nue (a low
        energy cascade is compact in time, whereas a noise event that survived
        L3 is a few hits spread out in time).  It does separate, but not as
        "real event ~1 / noise ~0".
        (17 noise events is few -- the ordering is what this file shows, while
        the magnitude finding is structural and solid.)

    The pass3 L3 output has CleanedFullTimeLength and UncleanedFullTimeLength
    SEPARATELY inside IC2018_LE_L3_Vars but NOT their ratio, which is why it is
    computed here.  When they are absent from the L3 map it is measured
    directly from the pulse series.
    '''
    if output_key in frame:
        return True

    cleaned = uncleaned = None

    if l3_key in frame:
        v = frame[l3_key]
        if "CleanedFullTimeLength" in v and "UncleanedFullTimeLength" in v:
            cleaned = float(v["CleanedFullTimeLength"])
            uncleaned = float(v["UncleanedFullTimeLength"])

    if cleaned is None and cleaned_pulses and uncleaned_pulses:
        def duration(key):
            pmap = get_pulses(frame, key)
            if pmap is None:
                return None
            times = [p.time for _, pulses in iter_map(pmap) for p in pulses]
            return (max(times) - min(times)) if times else None
        cleaned = duration(cleaned_pulses)
        uncleaned = duration(uncleaned_pulses)

    if cleaned is None or uncleaned is None or uncleaned <= 0:
        return True

    # `uncleaned <= 0` prevents division by zero but does NOT prevent NaN:
    # NaN <= 0 is False, so a NaN denominator passed the check and silently
    # wrote NaN into the result.  Check the result itself -- inf and NaN are
    # both rejected, the variable is not written, and it shows up as missing
    # in the HDF5 (better than a wrong number).
    ratio = float(cleaned) / float(uncleaned)
    if not np.isfinite(ratio):
        return True

    # The cleaned series is a subset of the uncleaned one, so the ratio must
    # be <= 1.  If it exceeds that, the two durations are being measured from
    # different base series -- do not let it pass in silence.
    if ratio > 1.0 and not getattr(_full_time_length_ratio, "_warned", False):
        _full_time_length_ratio._warned = True
        icetray.logging.log_warn(
            "FullTimeLengthRatio > 1 (%.3f): the cleaned duration (%.1f ns) "
            "exceeds the uncleaned one (%.1f ns).  The two may not be measured "
            "from the same base series." % (ratio, cleaned, uncleaned))

    frame[output_key] = dataclasses.I3Double(ratio)
    return True


@icetray.traysegment
def oscNext_L4_common_variables(tray, name, cleaned_pulses, uncleaned_pulses=None):
    '''First HLC hit, rho, and FullTimeLengthRatio.'''

    tray.Add(_first_hlc, name + "_FirstHLC",
             pulses_key=cleaned_pulses,
             output_key=L4_FIRST_HLC_KEY)

    tray.Add(_add_rho_36, name + "_FirstHLCRho",
             particle_key=L4_FIRST_HLC_KEY,
             output_key=L4_FIRST_HLC_RHO_KEY)

    tray.Add(_full_time_length_ratio, name + "_FTLR",
             output_key=L4_FTLR_KEY,
             cleaned_pulses=cleaned_pulses,
             uncleaned_pulses=uncleaned_pulses)


class PropagateGenieInfo(icetray.I3Module):
    '''
    Write I3GenieInfo.n_flux_events into every Physics frame as an I3Double.

    WHY: I3GenieInfo appears once per file, in a NON-Physics frame (S/M).
    HDF5 booking is per event, so unless the value is carried into every P
    frame it cannot be used in the weight calculation.

    Weight convention (the same as the existing oscnext_rates.py):
        weight [Hz] = OneWeight * flux(E) / n_flux
        n_flux      = I3GenieInfo.n_flux_events                (when present)
                    = NEvents * 0.7 (nu) or 0.3 (nubar)        (otherwise)
    '''

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter("OutputKey", "frame key to write", L4_NFLUX_KEY)
        self.AddOutBox("OutBox")

    def Configure(self):
        self.output_key = self.GetParameter("OutputKey")
        self.n_flux = None
        self.warned = False
        self.n_seen = 0          # how many I3GenieInfo seen (= how many L3 files)

    def _grab(self, frame):
        '''
        Try to read I3GenieInfo.  A deserialisation error (genie_icetray not
        imported) or a missing field must NOT kill the job -- the weight
        calculation falls back to NEvents.

        CAREFUL: every L3 FILE has its own I3GenieInfo and one tray processes
        several files (--chunk-files).  The value must be UPDATED at every new
        I3GenieInfo.  It used to be read once and frozen, so the first file's
        n_flux_events was applied to the events of every later file and their
        weights came out wrong.
        '''
        if not frame.Has("I3GenieInfo"):
            return
        try:
            new_val = float(frame["I3GenieInfo"].n_flux_events)
            self.n_seen += 1
            if self.n_flux is not None and new_val != self.n_flux:
                icetray.logging.log_info(
                    "PropagateGenieInfo: n_flux_events changed %g -> %g "
                    "(file %d) -- updating"
                    % (self.n_flux, new_val, self.n_seen))
            self.n_flux = new_val
            icetray.logging.log_info(
                "PropagateGenieInfo: n_flux_events = %g" % self.n_flux)
        except Exception as e:
            if not self.warned:
                icetray.logging.log_warn(
                    "PropagateGenieInfo: could not read I3GenieInfo (%s: %s). "
                    "Was genie_icetray imported?  The weight calculation will "
                    "fall back to NEvents * nu/nubar fractions."
                    % (type(e).__name__, e))
                self.warned = True

    # NOTE: because Process() is overridden, stream methods such as DAQ() and
    # Simulation() are NEVER CALLED -- the Process below does the dispatch and
    # calls _grab on every stream.  (Both used to exist here; DAQ and
    # Simulation were dead code.)
    def Process(self):
        frame = self.PopFrame()
        if frame.Stop != icetray.I3Frame.Physics:
            self._grab(frame)
            self.PushFrame(frame)
            return
        self._grab(frame)
        if self.n_flux is not None and self.output_key not in frame:
            frame[self.output_key] = dataclasses.I3Double(self.n_flux)
        elif self.n_flux is None and not self.warned:
            icetray.logging.log_warn(
                "PropagateGenieInfo: no I3GenieInfo found -- the weight "
                "calculation will fall back to NEvents * nu/nubar fractions")
            self.warned = True
        self.PushFrame(frame)


# ===========================================================================
# 2. MUON REJECTION VARIABLES
# ===========================================================================

def _accumulated_time(frame, pulses_key, output_key, fraction=0.75,
                      before_crossing=False):
    '''
    Time [ns] taken to reach 75% of the event's total charge.

    Technical note Table 12: "Time to reach 75% of an event's charge in the
    cleaned pulse series."  One of the few charge-dependent variables in the
    selection.  The original took it from analysis/event_selection's
    CalculateVariables module; it is computed directly here.

    before_crossing=False (DEFAULT, the note): the time of the first pulse
        whose cumulative charge REACHES the fraction -- i.e. the time at which
        the event has 75% of its charge, which is what Table 12 describes.

    before_crossing=True: the pulse BEFORE that one.  This reproduces the
        pass2 production, and it is an off-by-one: at that pulse the event has
        LESS than 75% of its charge, so it is not "the time to reach 75%".

    THE PASS2 BEHAVIOUR IS MEASURED, NOT GUESSED.  On 8144 pass2 L4 events
    with both values in the same frame, this variable was fitted against the
    stored L4_accumulated_time:

        all pulses, fraction 0.75, the pulse BEFORE the crossing   99.40%
        per-DOM charge, same rule                                  54.25%
        all pulses, fraction 0.70, at the crossing                 44.35%
        all pulses, fraction 0.75, at the crossing (the note)       0.98%

    with a median difference of 0 for the winner.  It was found by inverting
    the question rather than guessing: the charge fraction accumulated at
    pass2's stored time sits just BELOW 0.75 in at least 84% of events
    (median 0.7258, 84th pct 0.7449) and never above it, which is exactly one
    entry's charge share short -- the signature of an off-by-one.  The median
    shortfall, 0.024, is the share of an average pulse at this hit multiplicity.

    The DEFAULT stays with the note, as it does for micro_count (open risk 5b):
    we follow the description, not the original's slip.  Use
    before_crossing=True only to reproduce pass2 numbers.
    '''
    if output_key in frame:
        return True
    pmap = get_pulses(frame, pulses_key)
    if pmap is None:
        return True

    times, charges = [], []
    for _, pulses in iter_map(pmap):
        for p in pulses:
            times.append(p.time)
            charges.append(max(p.charge, 0.0))
    if not times:
        return True

    t = np.asarray(times); q = np.asarray(charges)
    # stable: pulses sharing a time must keep their order, or the index the
    # cumulative sum lands on is not reproducible run to run.  The fit that
    # settled before_crossing used a stable sort, so production must too.
    order = np.argsort(t, kind="stable")
    t, q = t[order], q[order]
    total = q.sum()
    if total <= 0:
        return True
    cum = np.cumsum(q) / total
    idx = int(np.searchsorted(cum, fraction))
    idx = min(idx, len(t) - 1)
    if before_crossing:
        idx = max(idx - 1, 0)
    frame[output_key] = dataclasses.I3Double(float(t[idx] - t[0]))
    return True


def _separation_in_cogs(frame, pulses_key, output_key, geometry_key="I3Geometry"):
    '''
    Split the event into two halves in time, compute each half's charge
    weighted COG, and write the distance between them.

    A track moves on   -> the COG shifts -> large separation.
    A cascade is local -> small separation.

    CAREFUL: the exact definition of the original Dunkman implementation could
    not be verified (the project is unavailable).  If you have reference files,
    compare this variable against them.  It is not a BDT input, so it is not
    critical.
    '''
    if output_key in frame:
        return True
    pmap = get_pulses(frame, pulses_key)
    if pmap is None or geometry_key not in frame:
        return True

    hits = list(iter_hits(pmap, frame[geometry_key]))
    if len(hits) < 4:
        return True
    hits.sort(key=lambda h: h[2])          # sort by time
    half = len(hits) // 2
    c1 = charge_weighted_cog(hits[:half])
    c2 = charge_weighted_cog(hits[half:])
    if c1 is None or c2 is None:
        return True
    d = np.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2 + (c2[2]-c1[2])**2)
    frame[output_key] = dataclasses.I3Double(float(d))
    return True


def _vich(frame, uncleaned_pulses,
          nch_key, npulses_key, qtot_key, geometry_key="I3Geometry",
          trigger_key="I3TriggerHierarchy", config_ids=(1010, 1011)):
    '''
    Veto Identified Causal Hits -- REPRODUCES pass2 EXACTLY (100.00%).

    THE ALGORITHM, AND HOW IT WAS FOUND
    -----------------------------------
    This is `VetoCausalHits` from the pythonic reimplementation of the
    LowEnVariables algorithms used by the GRECO online filter
    (`grecovariables.py`).  The GRECO tray calls it twice on the same
    uncleaned series, extracted pulse by pulse:

        VetoCausalCharge = VetoCausalHits(..., useCharge=True)    -> qtot
        VetoCausalHits   = VetoCausalHits(..., useCharge=False)   -> npulses

    For every trigger with a matching config id, the hit CLOSEST IN TIME to
    that trigger is taken as the reference, and with `dt = t_ref - t_hit` and
    `d` the distance to it, a hit counts when

        d  < 750 m
        dt > -5 d + 500
        dt < d / 0.3 + 150
        dt > d / 0.3 - 1850

    Only the FIRST matching trigger is used, and `TriggerConfigIDs` defaults to
    **[1010, 1011]**, which the L4 tray does not override.

    CONFIRMED AGAINST THE MODULE ITSELF.  The algorithm was found by fitting
    candidate definitions against pass2 and reaching 100.00%; the oscNext_meta
    V01-00-07 build that produced pass2 then turned out to carry
    `tau_bdt/python/I3CutL7Module.py` -- Python, not C++, and exactly what
    `from icecube.tau_bdt import I3CutL7Module` imports.  It settles every
    detail this function had to assume:

      - `break` after the first trigger whose config id is in the list;
      - `nch` counts DOMs with at least one selected pulse (`foundOne`);
      - `npulses` counts the selected pulses;
      - `nVetoHitsTotalPE += pulse.charge`, with no clamping of negatives;
      - the reference is the pulse closest in time to the trigger, ties going
        to the first in map order (a strict `<`, as `np.argmin` does).

    One deviation is deliberate.  The original seeds its reference search with
    `refPulseTime = 0` and `refPulsePos = (0,0,0)`, so if NO pulse is closer to
    the trigger than t=0 is, that sentinel survives and the whole calculation
    is done against the origin at t=0.  With a trigger at ~10 us and pulses in
    the readout window this cannot happen, and it did not in any of the 8144
    verified events; reproducing it would only copy a latent bug.

    A second, structurally identical implementation exists as `CausalTrackVeto`
    in `analysis/python/event_selection/I3CutL7Module_JP_Matt.py`:

        def mightBeBackground(distance, timeDiff):
            return ((distance < 750) and (timeDiff > (-5*distance + 500)) and
                    ((distance/0.3 - 1850) < timeDiff < (distance/0.3 + 150)))

    Its docstring names the physics: "charge that might have caused the
    DeepCore trigger ... direct or scattered light that could have come from a
    muon track approaching the trigger position".  It restricts itself to
    config id 1011 and returns charge only, so it corroborates the bands but
    is NOT the module L4 runs; where the two differ, I3CutL7Module.py wins.
    (GRECO's `VetoCausalHits` is a third implementation of the same bands, and
    it loops over every matching trigger with no `break` -- which is why the
    trigger question had to be settled from the module itself rather than by
    majority.)

    MEASURED against the real pass2 L4 files, 8144 events, all three outputs:
    nch 100.00%, npulses 100.00%, qtot 100.00%, median difference 0.

    WHAT THIS REPLACED, AND WHY THE OLD VERSION COULD NOT WORK
    ---------------------------------------------------------
    The first implementation followed technical note sec. 3.4: for every hit in
    the veto region, the speed between it and the event's COG vertex, kept when
    that speed fell in [0.25, 0.40] m/ns.  It reproduced pass2 in ~12% of
    events with the median difference pinned at 2 DOMs.

    Reading sec. 3.4 in full (p.26-27) showed why: that passage describes the
    LEVEL 2 DeepCore Filter, a different algorithm, which uses the window to
    DISCARD hits.  The note's only description of `L4_VICH_nch` is Table 12's
    one line -- no window, no region, no reference point.

    Five hypotheses were then retired by measurement, all landing at ~12%:
    the speed window (a 36-cell sweep of both edges peaked at 18.0%, with the
    median difference pinned at 2 in EVERY cell -- the signature of sweeping
    the wrong knob), the veto region (closed by construction: DOMS.DOMS("IC86")
    gives 554 fiducial + 4606 veto DOMs, disjoint, 5160 = the whole in-ice
    detector, so DeepCoreVetoDOMs already IS "not fiducial"), the COG
    (unweighted 12.0%, non-fiducial 11.9%), the reference point (first HLC hit,
    11.7%) and the input series (cleaned, 2.8%).

    Three structural differences separate the real algorithm from that one, and
    none of them is reachable by tuning a parameter:

      1. There is NO speed window.  Selection is four conditions in the
         (distance, dt) plane -- a 750 m sphere and three causality bands.
      2. There is NO veto DOM list.  The region is implicit in the bands, so
         the whole series is scanned.  Restricting to the veto DOMs drops
         agreement from 100% to 19.4%.
      3. The reference is the hit nearest the TRIGGER, not a charge-weighted
         fiducial COG.

    Hence this function no longer takes `cleaned_pulses` or `fiducial_cog`:
    neither plays any part.  The uncleaned series was already verified --
    `reference/oscNext_L4_pass2_original.py` passes
    `InputPulses=uncleaned_pulses  # Use uncleaned hits` to `I3CutL7Module`.

    Originally done by tau_bdt.I3CutL7Module, which was never found.
    '''
    if nch_key in frame:
        return True
    unc = get_pulses(frame, uncleaned_pulses)
    if unc is None or geometry_key not in frame:
        return True
    geo = frame[geometry_key]

    trigger_times = []
    if trigger_key in frame:
        for trig in frame[trigger_key]:
            # MERGED and THROUGHPUT triggers carry no config id; only the
            # SIMPLE_MULTIPLICITY one does.  Skipping them matters: reading
            # config_id unguarded raises and loses every trigger in the event.
            cid = getattr(getattr(trig, "key", None), "config_id", None)
            if cid is not None and int(cid) in config_ids:
                trigger_times.append(float(trig.time))

    # Pulse by pulse, as GetHitInformation(..., hitMode=0) does.
    xs, ys, zs, ts, qs, doms = [], [], [], [], [], []
    for omkey, pulses in iter_map(unc):
        if omkey not in geo.omgeo:
            continue
        pos = geo.omgeo[omkey].position
        key = (omkey.string, omkey.om)
        for pulse in pulses:
            xs.append(pos.x); ys.append(pos.y); zs.append(pos.z)
            ts.append(pulse.time); qs.append(pulse.charge); doms.append(key)

    if not ts or not trigger_times:
        frame[nch_key]     = dataclasses.I3Double(0.0)
        frame[npulses_key] = dataclasses.I3Double(0.0)
        frame[qtot_key]    = dataclasses.I3Double(0.0)
        return True

    x = np.asarray(xs); y = np.asarray(ys); z = np.asarray(zs)
    t = np.asarray(ts); q = np.asarray(qs)

    # ONLY THE FIRST matching trigger, per the module's own `break`.  An
    # earlier version looped over every matching trigger and accumulated.  It
    # still agreed with pass2 in 100% of 8144 events, because those events
    # carry exactly one matching trigger (next to a MERGED and a THROUGHPUT
    # one, which have no config id at all) -- so the bug was invisible in the
    # data that verified it, and an event with two would have been double
    # counted.  Found by reading the original, not by measurement; the same
    # goes for the [1010, 1011] default, which pass2 simulation never exercises
    # because 1010 does not appear in it.
    trigger_time = trigger_times[0]
    i = int(np.argmin(np.abs(t - trigger_time)))
    d = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2 + (z - z[i]) ** 2)
    dt = t[i] - t
    sel = d < 750.0
    sel &= dt > (-5.0 * d + 500.0)
    sel &= dt < (d / 0.3 + 150.0)
    sel &= dt > (d / 0.3 - 1850.0)

    n_pulses = float(sel.sum())
    # The original sums pulse.charge as it is -- no clamping of negatives.
    qtot = float(q[sel].sum())
    n_doms = len({doms[j] for j in np.flatnonzero(sel)})

    frame[nch_key]     = dataclasses.I3Double(float(n_doms))
    frame[npulses_key] = dataclasses.I3Double(float(n_pulses))
    frame[qtot_key]    = dataclasses.I3Double(float(qtot))
    return True


@icetray.traysegment
def oscNext_L4_atm_muon_classifier_variables(tray, name,
                                             uncleaned_pulses,
                                             cleaned_pulses,
                                             run_qr_box=False,
                                             run_optional=False,
                                             accumulated_time_pass2=False):
    '''
    Inputs of the L4 atmospheric muon rejection classifier.

    run_optional=False (the default) skips the computations that are NOT BDT
    inputs (I3TensorOfInertia and separation_in_cogs).  Neither appears in
    Table 12; they are kept as candidates/legacy but cost real time per event.
    Turn them on with process_L4.py --run-optional.
    '''

    # Projects this segment needs -- raise an explicit error HERE if absent
    # (not at import time, so the rest of the repository stays importable).
    require_project("linefit")

    # --- Tensor of inertia (not a BDT input; candidate/legacy) ---
    if run_optional:
        require_project("tensor_of_inertia")
        tray.AddModule("I3TensorOfInertia", name + "_ToI",
                       AmplitudeOption=1,
                       AmplitudeWeight=1,
                       InputReadout=cleaned_pulses,
                       InputSelection="",
                       MinHits=3,
                       Name=L4_TOI_KEY)

    # --- improved LineFit ---
    # The field used by the BDT is the speed: L4_iLineFitParams.LFVel
    tray.AddSegment(linefit.simple, name + "_iLineFit",
                    inputResponse=cleaned_pulses,
                    fitName=L4_LINEFIT_KEY)

    # --- QR box (slc-veto; optional, not a BDT input) ---
    if run_qr_box:
        try:
            if not load_lib("slc-veto"):
                raise RuntimeError("the slc-veto library is not in this build")
            tray.AddModule("SmallQ_Box", name + "_QRBox",
                           BoxName=L4_QRBOX_KEY,
                           RecoPulsesKey=cleaned_pulses)
        except Exception as e:
            icetray.logging.log_warn("QR box skipped (no slc-veto): %s" % e)

    # --- Dunkman variables (Python rewrite) ---
    tray.Add(_accumulated_time, name + "_AccTime",
             pulses_key=cleaned_pulses,
             output_key=L4_ACC_TIME_KEY,
             before_crossing=accumulated_time_pass2)

    # Not a BDT input (absent from Table 12) -- pure Python, real per-event cost
    if run_optional:
        tray.Add(_separation_in_cogs, name + "_SepCOG",
                 pulses_key=cleaned_pulses,
                 output_key=L4_SEP_IN_COGS_KEY)

    # --- VICH (tau_bdt rewrite) ---
    tray.Add(_vich, name + "_VICH",
             uncleaned_pulses=uncleaned_pulses,
             nch_key=L4_VICH_NCH_KEY,
             npulses_key=L4_VICH_NPULSES_KEY,
             qtot_key=L4_VICH_QTOT_KEY)


# ===========================================================================
# 3. NOISE REJECTION VARIABLES
# ===========================================================================

def _micro_count(frame, pulses_key, output_key, subkey):
    '''Write the number of hit DOMs in the cleaned+windowed series as an
    I3MapStringInt.'''
    values = dataclasses.I3MapStringInt()
    pmap = get_pulses(frame, pulses_key)
    # DOM count: len(map) works in every version, whereas .keys() sometimes
    # returns a view/iterator rather than a list.
    if pmap is None:
        n_doms = 0
    else:
        try:
            n_doms = len(pmap)
        except TypeError:
            n_doms = sum(1 for _ in iter_map(pmap))
    values[subkey] = int(n_doms)
    if output_key not in frame:
        frame[output_key] = values
    return True


@icetray.traysegment
def oscNext_L4_noise_cut_variables(tray, name,
                                   fill_ratio_vertex,
                                   cleaned_pulses,
                                   micro_count_pulses=None):
    '''
    Inputs of the L4 pure noise rejection classifier.

    micro_count_pulses: the pulse series the micro_count chain STARTS from.
        None (default) -> cleaned_pulses, i.e. what Table 11 of the technical
        note says ("Start with the cleaned pulse series").
        Pass the uncleaned series to reproduce the original pass2 code exactly
        -- see CLAUDE.md, open risk 5b.  fill_ratio uses cleaned_pulses either
        way (as the original does).
    '''
    if micro_count_pulses is None:
        micro_count_pulses = cleaned_pulses

    #
    # Micro count
    #
    # Technical note, Table 11 (p.36) -- the description VERBATIM:
    #   "Start with the cleaned pulse series.  Look at pulses occurring within
    #    [-3.5 us, +4 us] from the trigger time.  Slide a time window of 200 ns
    #    that maximizes the number of triggered DOMs in it.  Get the number of
    #    triggered DOMs in that sliding time window"
    #
    # Chain:  cleaned -> static TW [-3500,+4000] ns -> DeepCore fiducial
    #         -> 200 ns dynamic window -> count DOMs
    #
    # FIX (see CLAUDE.md, "Booking/read audit"): this chain used to start
    # from the UNCLEANED series with an I3SeededRTCleaning in the middle -- but
    # that module's output (L4_SRTTWPulses) was read NOWHERE: I3OMSelection
    # took the StaticTWC output as its input, not the SeededRT one.  So the
    # only noise-cleaning step in the chain was effectively disabled and
    # micro_count was counted over raw (uncleaned) hits.
    #
    # It now starts from the CLEANED series (SRTTWSplitInIcePulsesDC) as the
    # note says.  The SeededRT block was also removed: L3 already applied SRT
    # cleaning to that series (that is the "SRT" in its name), so applying it
    # again would be double cleaning.
    #
    # Measured afterwards, the difference is small: the two chains agree on 91%
    # of nue and 94% of noise events, because the closing 200 ns window is what
    # actually decides.
    #
    # L3's own microcount (STW9000_DTW300Hits, [-4,+5] us / 300 ns) uses
    # different parameters -- the two are not fully correlated, and the BDT
    # extracts information from both.

    require_project("DomTools")
    require_project("fill_ratio")
    if not load_lib("static-twc"):
        raise RuntimeError(
            "the C++ library 'static-twc' is not in this build -- micro_count "
            "cannot be computed.  Check whether DomTools/static-twc were "
            "built (python scripts/diagnose_env.py).")

    tw_pulses = "L4_TWPulses"
    tray.AddModule("I3StaticTWC<I3RecoPulseSeries>", name + "_StaticTWC_DC",
                   InputResponse=micro_count_pulses,
                   OutputResponse=tw_pulses,
                   TriggerConfigIDs=[1010, 1011],
                   TriggerName="I3TriggerHierarchy",
                   WindowMinus=STW_MINUS,
                   WindowPlus=STW_PLUS)

    # The classic IC86 DeepCore fiducial volume
    dom_list = deepcore_doms("IC86")
    tw_fid_pulses = tw_pulses + "_DCFid"

    tray.AddModule("I3OMSelection<I3RecoPulseSeries>", name + "_DCFidPulses",
                   selectInverse=True,
                   InputResponse=tw_pulses,
                   OutputResponse=tw_fid_pulses,
                   OmittedKeys=dom_list.DeepCoreFiducialDOMs)

    dtw_pulses = tw_fid_pulses + ("_DTW%i" % DTW)
    tray.AddModule("I3TimeWindowCleaning<I3RecoPulse>", name + "_DynamicTW",
                   InputResponse=tw_fid_pulses,
                   OutputResponse=dtw_pulses,
                   TimeWindow=DTW)

    tray.Add(_micro_count, name + "_MicroCount",
             pulses_key=dtw_pulses,
             output_key=L4_MICROCOUNT_KEY,
             subkey=MICROCOUNT_SUBKEY)

    #
    # Fill ratio
    #
    # GRECO used a different pulse series; the standard cleaned series is used
    # here (the original comment says "works well, leaving it like this for
    # simplicity").

    tray.AddModule("I3FillRatioModule", name + "_FillRatio",
                   RecoPulseName=cleaned_pulses,
                   ResultName=L4_FILL_RATIO_KEY,
                   SphericalRadiusMean=FILL_RATIO_SPHERICAL_RADIUS_MEAN,
                   VertexName=fill_ratio_vertex)


# ===========================================================================
# 4. HIT STATISTICS  (the muon BDT's cog_z / z_sigma / z_travel inputs)
# ===========================================================================

@icetray.traysegment
def oscNext_L4_hit_statistics(tray, name, cleaned_pulses):
    '''
    Compute hit statistics and multiplicity with common_variables.
    Skip this segment if L3 already computes them.
    '''
    require_project("common_variables")
    from icecube.common_variables import hit_statistics, hit_multiplicity

    tray.AddSegment(hit_statistics.I3HitStatisticsCalculatorSegment,
                    name + "_HitStatistics",
                    PulseSeriesMapName=cleaned_pulses,
                    OutputI3HitStatisticsValuesName=HITSTAT_KEY,
                    BookIt=False,
                    If=lambda f: HITSTAT_KEY not in f)

    tray.AddSegment(hit_multiplicity.I3HitMultiplicityCalculatorSegment,
                    name + "_HitMultiplicity",
                    PulseSeriesMapName=cleaned_pulses,
                    OutputI3HitMultiplicityValuesName=HITMULT_KEY,
                    BookIt=False,
                    If=lambda f: HITMULT_KEY not in f)


# ===========================================================================
# 5. KESIMLER
# ===========================================================================

def L4_noise_straight_cuts(frame, output_key=L4_NOISE_STRAIGHT_CUT_KEY):
    '''
    Loose straight cuts as an alternative to the classifier.  Produced but not
    used in the cut -- kept in case the classifier turns out to be a problem.
    Also a useful reference for which direction each variable separates in.
    '''
    try:
        keep = (
            frame[HITMULT_KEY].n_hit_doms >= 8 and
            frame["IC2018_LE_L3_Vars"]["STW9000_DTW300Hits"] >= 2 and
            frame[L4_MICROCOUNT_KEY][MICROCOUNT_SUBKEY] >= 2 and
            frame[L4_FILL_RATIO_KEY].fill_ratio_from_mean >= 0.03 and
            frame[HITSTAT_KEY].z_sigma >= 8. and
            frame[HITSTAT_KEY].z_travel >= -50.
        )
    except (KeyError, AttributeError):
        keep = False
    frame[output_key] = icetray.I3Bool(bool(keep))
    return True


@icetray.traysegment
def compute_L4_cut(tray, name, classifier_model_dir,
                   noise_cut=0.70, muon_cut=0.65):
    '''
    Apply the trained classifiers and compute the L4 cut.

    NOTE: the oscNext project (icecube.oscNext.tools.classifier.I3Classifier)
    is NOT in this meta-project.  oscnext_l4.classifier is used instead -- it
    does the same job and depends only on lightgbm + numpy (sklearn and joblib
    do not exist in the IceTray environment).

    Skip this segment while the models are NOT YET TRAINED: the variables have
    to be booked without a cut first, so the classifiers can be trained.
    '''
    from .classifier import add_L4_classifiers

    tray.Add(L4_noise_straight_cuts, name + "_straight_cuts")

    tray.Add(add_L4_classifiers, name + "_classifiers",
             model_dir=classifier_model_dir,
             noise_cut=noise_cut,
             muon_cut=muon_cut,
             apply_cut=True)


# ===========================================================================
# 6. ANA SEGMENT
# ===========================================================================

@icetray.traysegment
def oscNext_L4(tray, name,
               uncleaned_pulses=UNCLEANED_PULSES_DEFAULT,
               cleaned_pulses=CLEANED_PULSES_DEFAULT,
               apply_l3_cut=True,
               is_genie=False,
               compute_hit_statistics=True,
               run_optional=False,
               apply_cut=False,
               classifier_model_dir=None,
               micro_count_uncleaned=True,
               accumulated_time_pass2=True):
    '''
    The main oscNext L4 tray segment.

    apply_cut=False (default): compute the variables only.  Run in this mode
    before the models are trained -- every event is booked without a cut, so
    both the noise and the muon training set come out of one pass.

    accumulated_time_pass2=True (DEFAULT): accumulated_time takes the pulse
    BEFORE the cumulative charge crosses 75%, reproducing pass2 at 99.40% over
    8144 events.  Set False to follow the technical note instead, which says
    "time to reach 75%" and so takes the pulse AT the crossing -- that agrees
    with pass2 in 0.98% of events.

    micro_count_uncleaned=True (DEFAULT): the micro_count chain starts from the
    uncleaned series, as the original pass2 code does.  Set False to follow
    Table 11, which says to start from the cleaned series.  The two differ in
    about 9% of events.

    BOTH DEFAULTS REPRODUCE THE PRODUCTION, NOT THE NOTE.  That is a deliberate
    reversal: where the note and the production disagree, matching the numbers
    the collaboration actually produced is what this pipeline is for, and a
    model trained on variables that differ from the production's is training on
    a different quantity.  The note's reading stays one argument away in both
    cases, and both deviations are documented where they are implemented
    (_accumulated_time's docstring, and open risk 5b in CLAUDE.md).
    '''

    # I3GenieInfo -> into every P frame, BEFORE the L3 cut so that the S
    # frames are unaffected by it
    if is_genie:
        tray.Add(PropagateGenieInfo, name + "_genie_info")

    if apply_l3_cut:
        # The L3 script does NOT drop events, it only writes a bool -> apply
        # the cut here.
        #
        # Order of preference:
        #   1) L3_oscNext_bool  = IC2018_LE_L3_Full AND Data_quality_bool
        #      (data quality included: no LID errata AND did not pass the
        #       SLOP filter)
        #   2) IC2018_LE_L3_bools.IC2018_LE_L3_Full  (data quality EXCLUDED)
        #
        # SLOP triggers can satisfy the DeepCore criteria by accident thanks
        # to their very wide time window (a few ms), but they are not simulated
        # properly -- hence (1) is preferred.
        def l3_cut(frame):
            if "L3_oscNext_bool" in frame:
                return bool(frame["L3_oscNext_bool"].value)
            if "IC2018_LE_L3_bools" in frame:
                return bool(frame["IC2018_LE_L3_bools"]["IC2018_LE_L3_Full"])
            return False
        tray.Add(l3_cut, name + "_L3_cut")

    tray.Add(oscNext_L4_common_variables, name + "_common",
             cleaned_pulses=cleaned_pulses,
             uncleaned_pulses=uncleaned_pulses)

    if compute_hit_statistics:
        tray.Add(oscNext_L4_hit_statistics, name + "_hitstats",
                 cleaned_pulses=cleaned_pulses)

    tray.Add(oscNext_L4_noise_cut_variables, name + "_noise_vars",
             fill_ratio_vertex=L4_FIRST_HLC_KEY,
             cleaned_pulses=cleaned_pulses,
             micro_count_pulses=(uncleaned_pulses if micro_count_uncleaned
                                 else None))

    tray.Add(oscNext_L4_atm_muon_classifier_variables, name + "_muon_vars",
             uncleaned_pulses=uncleaned_pulses,
             cleaned_pulses=cleaned_pulses,
             run_optional=run_optional,
             accumulated_time_pass2=accumulated_time_pass2)

    if apply_cut:
        if classifier_model_dir is None:
            raise ValueError("apply_cut=True requires classifier_model_dir")
        tray.Add(compute_L4_cut, name + "_cut",
                 classifier_model_dir=classifier_model_dir)
