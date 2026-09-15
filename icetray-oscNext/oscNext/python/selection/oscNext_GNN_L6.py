'''
The oscNext GNN level 6 event selection traysegment.
This is where final level reconstruction is done.

Tom Stuttard, Kayla Leonard, Rasmus Orsoe
'''

import math, resource, copy, os, collections

import numpy as np

from icecube import dataclasses
from icecube import icetray
from icecube.icetray import I3Units
# from icecube import lilliput
# import icecube.lilliput.segments

from icecube.oscNext.tools.data_quality import check_object_exists
from icecube.oscNext.selection.globals import *
from icecube.oscNext.selection.oscNext_cuts import oscNext_cut
from icecube.oscNext.frame_objects.pulses import MCPULSE_KEY, pulse_info, prereco_pulse_cleaning, record_reco_pulse_truth_flags


#
# Variables
#

L6_HDF5_KEYS = []

# Pulse cleaning
L6_PRE_RECO_CLEANED_PULSES_KEY = "L6_oscNext_prereco_cleaned_pulses"
L6_HDF5_KEYS.append( L6_PRE_RECO_CLEANED_PULSES_KEY )

# SANTA
# L6_SANTA_KEY = "L6_SANTA"
# L6_SANTA_PULSES_KEY = L6_SANTA_KEY + "_DirectPulses"
# L6_SANTA_BEST_KEY = L6_SANTA_KEY + "_sel_Particle"
# L6_HDF5_KEYS.extend([L6_SANTA_KEY, L6_SANTA_BEST_KEY])

# Add DynEdge predictions to HDF5 keys
DYNEDGE_PREFIX = "L6_DynEdge"
L6_DYNEDGE_SUCCESS_KEY = DYNEDGE_PREFIX + "_success"
L6_DYNEDGE_ENERGY_KEY = DYNEDGE_PREFIX + "_energy_pred"
# L6_DYNEDGE_TRACK_ENERGY_KEY = DYNEDGE_PREFIX + "_energy_track_pred"
# L6_DYNEDGE_CASCADE_ENERGY_KEY = DYNEDGE_PREFIX + "_energy_cascade_pred"
L6_DYNEDGE_ZENITH_KEY = DYNEDGE_PREFIX + "_zenith_pred"
L6_DYNEDGE_ZENITH_KAPPA_KEY = DYNEDGE_PREFIX + "_zenith_kappa_pred"
L6_DYNEDGE_AZIMUTH_KEY = DYNEDGE_PREFIX + "_azimuth_pred"
L6_DYNEDGE_AZIMUTH_KAPPA_KEY = DYNEDGE_PREFIX + "_azimuth_kappa_pred"
L6_DYNEDGE_LENGTH_KEY = DYNEDGE_PREFIX + "_length_pred"
L6_DYNEDGE_VERTEX_X_KEY = DYNEDGE_PREFIX + "_position_x_pred"
L6_DYNEDGE_VERTEX_Y_KEY = DYNEDGE_PREFIX + "_position_y_pred"
L6_DYNEDGE_VERTEX_Z_KEY = DYNEDGE_PREFIX + "_position_z_pred"
L6_DYNEDGE_TIME_KEY = DYNEDGE_PREFIX + "_interaction_time_pred"
L6_DYNEDGE_PID_KEY = DYNEDGE_PREFIX + "_track_pred"

L6_HDF5_KEYS.extend([
    L6_DYNEDGE_ENERGY_KEY,
    # L6_DYNEDGE_TRACK_ENERGY_KEY,
    # L6_DYNEDGE_CASCADE_ENERGY_KEY,
    L6_DYNEDGE_ZENITH_KEY,
    L6_DYNEDGE_ZENITH_KAPPA_KEY,
    L6_DYNEDGE_AZIMUTH_KEY,
    L6_DYNEDGE_AZIMUTH_KAPPA_KEY,
    L6_DYNEDGE_LENGTH_KEY,
    L6_DYNEDGE_VERTEX_X_KEY,
    L6_DYNEDGE_VERTEX_Y_KEY,
    L6_DYNEDGE_VERTEX_Z_KEY,
    L6_DYNEDGE_TIME_KEY,
    L6_DYNEDGE_PID_KEY,
])

# Cut
L6_HDF5_KEYS.append( L6_CUT_BOOL_KEY )


#
# Pulse cleaning settings
#

# See https://drive.google.com/file/d/19xQb5sBxttNhQZZM1Pl89isuR-PFtVVT/view for details of cleaning settings
PULSE_CLEANING_SETTINGS = dict(
    signal_time_window_only=True,
    fiducial_doms_only=True,
    dom_stats=True,
    apply_srt=True,
    srt_ic_r=150.*I3Units.m,
    srt_ic_t=1000.*I3Units.ns,
    srt_dc_r=150.*I3Units.m,
    srt_dc_t=500.*I3Units.ns,
    srt_self_coincidence=False,
    srt_dust_layer_correction=False,
    srt_max_n_iterations=-1,
    srt_hlccore_seeding=False,
    no_charge=True,
    round_time=True,
)


#
# DynEdge config
#

# DynEdge input data configuration
DYNEDGE_PULSEMAP = L6_PRE_RECO_CLEANED_PULSES_KEY
DYNEDGE_FEATURES = ['dom_x', 'dom_y', 'dom_z', 'dom_time']
DYNEDGE_FEATURE_EXTRACTOR = "I3FeatureExtractorIceCube86"

# DynEdge model definitions
DYNEDGE_MODELS = collections.OrderedDict() # Each entry is [task_name] = (target_labels, prediction_labels)
DYNEDGE_MODELS["energy_reco"] = ( ["energy"], ["energy_pred"] )
DYNEDGE_MODELS["energy_tc_reco"] = ( ["energy_track", "energy_cascade"], ["energy_track_pred", "energy_cascade_pred"] )
DYNEDGE_MODELS["length_reco"] = ( ["length"], ["length_pred"] )
DYNEDGE_MODELS["direction_reco"] = ( ["direction"], ["direction_x_pred", "direction_y_pred", "direction_z_pred", "direction_kappa_pred"] )
DYNEDGE_MODELS["length_reco"] = ( ["length"], ["length_pred"] )
DYNEDGE_MODELS["zenith_reco"] = ( ["zenith"], ["zenith_pred", "zenith_kappa_pred"] )
DYNEDGE_MODELS["azimuth_reco"] = ( ["azimuth"], ["azimuth_pred", "azimuth_kappa_pred"] )
DYNEDGE_MODELS["position_reco"] = ( ["position_x", "position_y", "position_z"], ["position_x_pred", "position_y_pred", "position_z_pred"] )
DYNEDGE_MODELS["vertex_reco"] = ( ["position_x", "position_y", "position_z", "interaction_time"], ["position_x_pred", "position_y_pred", "position_z_pred", "interaction_time_pred"] )
DYNEDGE_MODELS["pid"] = ( ["track"], ["track_pred"] )
DYNEDGE_MODELS["muon_classifier"] = ( ["neutrino"], ["neutrino_pred"] )

# Paths to trained DynEdge models
DYNEDGE_MODEL_ROOT_DIR = "$I3_SRC/oscNext/resources/models/level6_GNN"
DYNEDGE_MODEL_PATHS = collections.OrderedDict()
DYNEDGE_MODEL_PATHS["energy_reco"] = os.path.join(f"{DYNEDGE_MODEL_ROOT_DIR}/energy_reco/v02.00/0000/{DYNEDGE_PULSEMAP}")
DYNEDGE_MODEL_PATHS["zenith_reco"] = os.path.join(f"{DYNEDGE_MODEL_ROOT_DIR}/zenith_reco/v02.00/0000/{DYNEDGE_PULSEMAP}")
DYNEDGE_MODEL_PATHS["azimuth_reco"] = os.path.join(f"{DYNEDGE_MODEL_ROOT_DIR}/azimuth_reco/v02.00/0000/{DYNEDGE_PULSEMAP}")
DYNEDGE_MODEL_PATHS["position_reco"] = os.path.join(f"{DYNEDGE_MODEL_ROOT_DIR}/position_reco/v01.34/0000/{DYNEDGE_PULSEMAP}")
DYNEDGE_MODEL_PATHS["length_reco"] = os.path.join(f"{DYNEDGE_MODEL_ROOT_DIR}/length_reco/v02.01/0000/{DYNEDGE_PULSEMAP}")
DYNEDGE_MODEL_PATHS["pid"] = os.path.join(f"{DYNEDGE_MODEL_ROOT_DIR}/pid/v02.01/0000/{DYNEDGE_PULSEMAP}")


#
# Tray segments
#

@icetray.traysegment
def oscNext_prereco_pulse_cleaning( tray, name, input_pulses_key ):
    '''
    Run custom pulse cleaning ahead of reconstruction
    '''

    #
    # Run cleaning
    #

    # Add cleaning to tray
    tray.Add(
        prereco_pulse_cleaning,
        L6_PRE_RECO_CLEANED_PULSES_KEY,
        input_pulse_map_key=input_pulses_key,
        output_key=L6_PRE_RECO_CLEANED_PULSES_KEY, 
        **PULSE_CLEANING_SETTINGS
    )


    #
    # Also generate extra information relating to the new pulse maps
    #

    # Add the pulse stats/info/etc segment 
    tray.Add(
        pulse_info,
        L6_PRE_RECO_CLEANED_PULSES_KEY+"_pulse_info",
        pulses=L6_PRE_RECO_CLEANED_PULSES_KEY,
    )

    # Add truth flags calculation
    # Use filter to only do this when the required truth info is present (it is not in older MC)
    mcpulse_truth_map_key = MCPULSE_KEY + "_TruthFlags"
    tray.Add(
        record_reco_pulse_truth_flags,
        L6_PRE_RECO_CLEANED_PULSES_KEY+"_record_reco_pulse_truth_flags",
        reco_pulse_map_key=L6_PRE_RECO_CLEANED_PULSES_KEY,
        mcpulse_map_key=MCPULSE_KEY,
        mcpulse_truth_map_key=mcpulse_truth_map_key,
        If=lambda f : f.Has(mcpulse_truth_map_key),
    )


@icetray.traysegment
def oscNext_DynEdge_reco( tray, name, gcd_file ):
    '''
    Run DynEdge (GNN) reconstruction
    '''

    from graphnet.data.constants import FEATURES
    from graphnet.deployment.i3modules import I3InferenceModule, GraphNeTI3Module
    from graphnet.data.extractors import i3featureextractor

    #TODO Prefer to directly load OscNextDynEdgeModel class and add some "get_i3_inference_module" method, but for now sticking to existing tools 


    #
    # Configuration
    #

    # Pulse map
    pulsemap = DYNEDGE_PULSEMAP

    # Input features for each pulse
    features = DYNEDGE_FEATURES

    # Use standard IceCube pulse feature extractor
    pulsemap_extractor = getattr(i3featureextractor, DYNEDGE_FEATURE_EXTRACTOR)(pulsemap=pulsemap)


    #
    # Add inference modules
    #

    # Loop over models
    for task_name, model_path in DYNEDGE_MODEL_PATHS.items() :

        # Get prediction names
        assert task_name in DYNEDGE_MODELS, "Could not find DynEdge task definition"
        prediction_names = DYNEDGE_MODELS[task_name][1]
        assert isinstance(prediction_names, list)
        assert all([ isinstance(n, str) for n in prediction_names ])
        print(f"DynEdge {task_name} : Prediction names = {prediction_names}")

        # Get paths to model files
        model_config = os.path.expandvars( f"{model_path}/model_config.yml" )
        state_dict = os.path.expandvars( f"{model_path}/state_dict.pth" )
        assert os.path.isfile(model_config), "Could not find model config file : %s" % model_config
        assert os.path.isfile(state_dict), "Could not find state dict file : %s" % state_dict

        # Configure inference/deployment module
        deployment_module = I3InferenceModule(
            pulsemap=pulsemap,
            features=features,
            pulsemap_extractor=pulsemap_extractor,
            model_config=model_config,
            state_dict=state_dict,
            gcd_file=gcd_file,                            #TODO ideally I3InferenceModule should grab GCD data from the frame, e.g. no need to load GCD again (also risky, user could pass a differnt file accidentally)
            prediction_columns=prediction_names, 
            model_name=DYNEDGE_PREFIX, # Used as prefix for the output frame keys
        )

        # Add to tray
        tray.AddModule(deployment_module, DYNEDGE_PREFIX+"_"+task_name) #TODO filter events with 0 pulses


def oscNext_DynEdge_reco_checks(frame) :
    '''
    Check the results of the DynEdge reco
    '''

    # Init flag
    reco_success = True

    # Loop over models
    for model_name in DYNEDGE_MODEL_PATHS.keys() :

        # Loop over predictons for this model
        assert model_name in DYNEDGE_MODELS
        for prediction_name in DYNEDGE_MODELS[model_name][1] :

            # Get the prediction
            prediction_key = DYNEDGE_PREFIX + "_" + prediction_name
            assert prediction_key in frame
            prediction_val = frame[prediction_key].value

            # Check the value is finite
            if not np.isfinite(prediction_val) :
                reco_success = False

            # Check physical bounds, where applicable
            if prediction_name in [ "energy_pred", "energy_track_pred", "energy_cascade_pred", "length_pred", "zenith_kappa_pred", "azimuth_kappa_pred", "direction_kappa_pred" ] :
                if prediction_val < 0. : # Some variable cannot be negative
                    reco_success = False
            if prediction_name == "zenith_pred" :
                if (prediction_val < 0.) or (prediction_val > np.pi) : # Zenith is within [0, pi]
                    reco_success = False
            if prediction_name == "azimuth_pred" :
                if (prediction_val < 0.) or (prediction_val > (2.*np.pi)) : # Azimuth is within [0, 2pi]
                    reco_success = False
            if prediction_name in ["direction_x_pred", "direction_y_pred", "direction_z_pred"]  :
                if (prediction_val < -1.) or (prediction_val > +1.) : # Unit vectors in are within [-1, +1]
                    reco_success = False
            if prediction_name in ["track_pred", "neutrino_pred"]  :
                if (prediction_val < 0.) or (prediction_val > +1.) : # Classifiers are within [0, 1]
                    reco_success = False

    frame[L6_DYNEDGE_SUCCESS_KEY] = icetray.I3Bool(reco_success)



# @icetray.traysegment
# def oscNext_SANTA(tray, name):
#     '''
#     SANTA reconstruction.
#     Extracted from oscNext Direct L6 segment, used for verification sample.
#     Using the new cleaned pulse map from this L6 however as input
#     '''

#     from icecube import santa

#     santa_settings = {
#         'SStoMSseed'              : None, # this causes default seeds to be computed
#         'SingleStringSeed'        : None,
#         'MultiStringSeed'         : None,
#         # 'Interactive'             : False, # as above, make this available for the running script
#         # 'SuppressPlots'           : False,
#         'D0scaling'               : 7,
#         'RobustResidualScaling'   : 3.,
#         'RobustResidualScalingMS' : 3.,
#         'FirstPulseThreshold'     : 0.1,
#         'MinimumStringDist'       : 4.,
#         'MinimumStringHits'       : 3,
#         'FixMinimumStringDist'    : False,
#         'UseCharge'               : True,
#         'UsePMTOrientation'       : True,
#         'LossFunction'            : 'cauchy',
#         'LossFunctionMS'          : 'cauchy',
#     }

#     def has_hits(frame, pulsesname, minhits=5):
#         if frame.Has(pulsesname):
#             if type(frame[pulsesname]) == dataclasses.I3RecoPulseSeriesMap:
#                 hit_map = frame.Get(pulsesname)
#             elif type(frame[pulsesname]) == dataclasses.I3RecoPulseSeriesMapMask:
#                 hit_map = frame[pulsesname].apply(frame)
#             return len(hit_map) >= minhits
#         else:
#             return False

#     def SANTA_picker(frame , basename):
#         SS_found = frame.Has(basename + "_FitTrack_SS_Contained_Particle")
#         MS_found = frame.Has(basename + "_FitTrack_MS_Contained_Particle")
#         if (not SS_found) and (not MS_found):
#             santa_fittype = 0
#         elif MS_found :
#             santa_fittype = 2
#             santa_particle  = frame[basename + "_FitTrack_MS_Contained_Particle" ]
#             santa_pid       = frame[basename + "_PID_MS"]
#             santa_fitresult = frame[basename + "_FitTrack_MS"]
#         elif SS_found :
#             santa_fittype = 1
#             santa_particle  = frame[basename + "_FitTrack_SS_Contained_Particle" ]
#             santa_pid       = frame[basename + "_PID_SS"]
#             santa_fitresult = frame[basename + "_FitTrack_SS"]
#         frame[basename+'_FitType'] = icetray.I3Int(santa_fittype)
#         if santa_fittype!=0:
#             frame[basename + "_sel_Particle"] = santa_particle
#             frame[basename + "_sel"]          = santa_fitresult
#             frame[basename + "_sel_PID"]      = santa_pid
#         return True


#     tray.AddSegment(
#         santa.SANTA_fitSegment, 
#         'SANTA_fit',
#         UncleanedPulses   = L6_PRE_RECO_CLEANED_PULSES_KEY,
#         Interactive       = False,
#         SuppressPlots     = True,
#         SANTAPulseSeries  = L6_SANTA_PULSES_KEY,
#         OutputBasename    = L6_SANTA_KEY,
#         If=lambda f: has_hits(f, L6_SANTA_PULSES_KEY),
#         **santa_settings
#     )

#     tray.AddModule(SANTA_picker, basename=L6_SANTA_KEY)


def compute_L6_cut(frame):
    '''
    Compute the actual reco stage cut boolean
    '''

    # Enforce at least 6 hit DOMs in the cleaned pulse map computed here
    # Required due to earlier similar cut used on SRTTWOfflinePulsesDC pulsemap (L5)
    hit_multiplicity_key = L6_PRE_RECO_CLEANED_PULSES_KEY + "HitMultiplicity"
    assert frame.Has(hit_multiplicity_key)
    num_hit_doms_cut = frame[hit_multiplicity_key].n_hit_doms >= 6

    # Check reco was succesful
    reco_success = frame[L6_DYNEDGE_SUCCESS_KEY].value

    # Build the final cut
    keep_event = num_hit_doms_cut & reco_success

    # Add to frame
    frame[L6_CUT_BOOL_KEY] = icetray.I3Bool(keep_event)
    

@icetray.traysegment
def oscNext_GNN_L6( tray, name, uncleaned_pulses, cleaned_pulses, gcd_file ):
# def oscNext_GNN_L6( tray, name, uncleaned_pulses, cleaned_pulses ):
    '''
    Main traysegment for running the oscNext GNN recos
    '''

    #
    # Apply L5 cut
    #

    # Only keep frames passing L5
    # This is actually also run by `oscNext_direct_L6` so could ingore this,
    # but doubling up for safety in case that tray segment changes in the future
    tray.Add( oscNext_cut, "L5_cut_high_stats", processing_level=5 ) # Differentiating from "L5_cut" also run as part of verification sample L6, which is also called by this code


    #
    # Run pre-reco pulse cleaning
    #

    # Run the pulse cleaning
    tray.Add(
        oscNext_prereco_pulse_cleaning, 
        "oscNext_prereco_pulse_cleaning",
        input_pulses_key=uncleaned_pulses,
    )

    # Also add the extra pulse info for the uncleaned pulses
    tray.Add(
        pulse_info,
        uncleaned_pulses+"_pulse_info",
        pulses=uncleaned_pulses,
    )


    #
    # DynEdge
    #

    # DynEdge reconstruction
    tray.Add(
        oscNext_DynEdge_reco, 
        "oscNext_DynEdge_reco",
        gcd_file=gcd_file,
    )

    # Check the results
    tray.Add(
        oscNext_DynEdge_reco_checks, 
        "oscNext_DynEdge_reco_checks",
    )


    #
    # Done
    #

    # Add the cut
    tray.Add(
        compute_L6_cut, 
        "compute_L6_cut",
    )

    # Can dump frame for debugging
    if False :
        tray.Add("Dump","Dump")

    return
