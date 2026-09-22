'''
The oscNext Level 4 tray segments.

This file is the working counterpart of the production's
`oscNext/python/selection/oscNext_L4.py`, whose body is commented out in full
(`#TODO migrate`).  It keeps that file's structure -- the same L4_* key names,
the same segments in the same order -- so the two can be read side by side.

The production calls three projects a modern meta-project does not carry:

    SimpleVertex   FirstHLC<I3RecoPulse>   -> the first HLC vertex
    tau_bdt        I3CutL7Module           -> the VICH variables
    analysis       CalculateVariables      -> the Dunkman variables

Each is one `tray.AddModule` line there and a section of `rewritten.py` here,
verified against the real pass2 L4 output.  The slc-veto QR box has no rewrite
and stays optional (it is not a BDT input).

Everything that is not a segment lives beside this file:

    frame_objects.py  calc_rho_36, iter_map, get_pulses, PropagateGenieInfo
                      -- our version of the production's frame_objects/, whose
                      weighting it runs from its master script, not from L4
    l3vars.py         FullTimeLengthRatio  (an L3 variable pass3's L3 does not store)
    rewritten.py      first_hlc, dunkman, vich

Reference: oscNext technical note v00.07, sections 3.4-3.6, Tables 11-12 --
but where the note and the production disagree, the production wins; see the
`oscNext_L4` docstring.
'''

from .env import (require_icetray, optional_project, require_project,
                  load_deserialization_libs, deepcore_doms,
                  load_lib)

# Helpers and rewrites live beside this file so that the segments below read
# next to the original script.  Importing them here also keeps them module
# attributes, so `from oscnext_l4.variables import _vich` still works.
from .frame_objects import calc_rho_36, iter_map, get_pulses
from .frame_objects import PropagateGenieInfo, L4_NFLUX_KEY
from .l3vars import _full_time_length_ratio, L4_FTLR_KEY
from .rewritten import _first_hlc, _accumulated_time, _separation_in_cogs, _vich

require_icetray()
from icecube import dataclasses, icetray

# --- Optional projects -----------------------------------------------------
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

L4_MICROCOUNT_KEY     = "L4_micro_count"
L4_FILL_RATIO_KEY     = "L4_fill_ratio"

L4_NOISE_STRAIGHT_CUT_KEY         = "L4_NoiseStraightCuts_Bool"
L4_NOISE_MODEL_PREDICTION_KEY     = "L4_NoiseClassifier_ProbNu"
L4_MUON_MODEL_PREDICTION_DATA_KEY = "L4_MuonClassifier_Data_ProbNu"

# ---------------------------------------------------------------------------
# Pulse serisi isimleri
#
# oscNext L3 (online_filterscripts .. grecovariables.DeepCoreCleaning) takes
# "SplitInIcePulses" as its input and produces "SRTTWSplitInIcePulsesDC".
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


# ===========================================================================
# 1. COMMON VARIABLES
# ===========================================================================

# The sentinel the original vertex search starts from.  It is not a guard:
# FirstHLC.cxx calls GetFirstHLCHit, IGNORES the bool it returns, and writes
# the I3Particle unconditionally -- so an event with no HLC hit gets a vertex
# at (1000, 1000, 1000) with time 1e10, and a first_hlc_rho of 1407.3.  Both
# the production's fill_ratio and its muon BDT saw that number, so we write it
# too.  (The original's own "was one found" check compares against 10000.0
# rather than 1000.0 -- an extra zero -- but nothing reads the result.)

def _add_rho_36(frame, particle_key, output_key):
    if particle_key in frame and output_key not in frame:
        pos = frame[particle_key].pos
        frame[output_key] = dataclasses.I3Double(calc_rho_36(pos.x, pos.y))
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


# ===========================================================================
# 2. MUON REJECTION VARIABLES
# ===========================================================================


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
