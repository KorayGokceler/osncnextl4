'''
The oscNext GNN level 7 event selection traysegment.
'''


import math, os, copy

import numpy as np

from icecube.icetray import I3Bool
from icecube import dataclasses
from icecube import icetray
from icecube.icetray import I3Units
from icecube import lilliput
import icecube.lilliput.segments
# icetray.logging.console() #TODO Breaks some stuff, think about this
from icecube.oscNext.tools.data_quality import check_object_exists
from icecube.oscNext.selection.globals import *
from icecube.oscNext.selection.oscNext_cuts import oscNext_cut
from icecube.oscNext.selection.oscNext_GNN_L6 import *
from icecube.oscNext.frame_objects.pulses import compute_extra_pulse_map_info
from icecube.oscNext.frame_objects.reco import calc_sigma_from_kappa
from icecube.oscNext.frame_objects.geom import compute_event_vertex_geometry_params
from icecube.common_variables.track_characteristics import I3TrackCharacteristicsCalculator


#
# Globals
#

# Define all output frame objects
L7_HDF5_KEYS = []

# Cut
L7_HDF5_KEYS.append( L7_CUT_BOOL_KEY )

# Reco post-processing HDF5 keys
L7_DYNEDGE_PARTICLE_KEY = DYNEDGE_PREFIX + "_particle"
L7_DYNEDGE_VERTEX_RHO36_KEY = DYNEDGE_PREFIX + "_position_rho36_pred"
L7_DYNEDGE_VERTEX_TIME_KEY = DYNEDGE_PREFIX + "_interaction_time_pred"
L7_DYNEDGE_STOPPING_POINT_KEY = DYNEDGE_PREFIX + "_stopping_point_pred"
L7_DYNEDGE_STOPPING_POINT_RHO36_KEY = DYNEDGE_PREFIX + "_stopping_point_rho36_pred"
L7_DYNEDGE_ZENITH_SIGMA_KEY = DYNEDGE_PREFIX + "_zenith_sigma_pred"
L7_DYNEDGE_AZIMUTH_SIGMA_KEY = DYNEDGE_PREFIX + "_azimuth_sigma_pred"
L7_HDF5_KEYS.extend([
    L7_DYNEDGE_PARTICLE_KEY,
    L7_DYNEDGE_ZENITH_SIGMA_KEY,
    L7_DYNEDGE_AZIMUTH_SIGMA_KEY,
    L7_DYNEDGE_VERTEX_RHO36_KEY,
    L7_DYNEDGE_VERTEX_TIME_KEY,
    L7_DYNEDGE_STOPPING_POINT_KEY,
    L7_DYNEDGE_STOPPING_POINT_RHO36_KEY
])

# Containment HDF5 keys
L7_DYNEDGE_DC_STARTING_CONTAINMENT_KEY = DYNEDGE_PREFIX + "_dc_starting_containment_pred"
L7_DYNEDGE_IC_STARTING_CONTAINMENT_KEY = DYNEDGE_PREFIX + "_ic_starting_containment_pred"
L7_DYNEDGE_DC_STOPPING_CONTAINMENT_KEY = DYNEDGE_PREFIX + "_dc_stopping_containment_pred"
L7_DYNEDGE_IC_STOPPING_CONTAINMENT_KEY = DYNEDGE_PREFIX + "_ic_stopping_containment_pred"
L7_HDF5_KEYS.extend([
    L7_DYNEDGE_DC_STARTING_CONTAINMENT_KEY, 
    L7_DYNEDGE_IC_STARTING_CONTAINMENT_KEY,
    L7_DYNEDGE_DC_STOPPING_CONTAINMENT_KEY,
    L7_DYNEDGE_IC_STOPPING_CONTAINMENT_KEY,
])

# Corridor cut (narrow) + angular correlation
L7_NARROW_CORRIDOR_CUT_PULSES_KEY = "L7_NarrowCorridorCutPulses"
L7_NARROW_CORRIDOR_CUT_COUNT_KEY = "L7_NarrowCorridorCutCount"
L7_NARROW_CORRIDOR_CUT_TRACK_KEY  = "L7_NarrowCorridorCutTrack"
L7_HDF5_KEYS.extend([ L7_NARROW_CORRIDOR_CUT_COUNT_KEY, L7_NARROW_CORRIDOR_CUT_TRACK_KEY ])
L7_HDF5_KEYS.extend([ 
    L7_NARROW_CORRIDOR_CUT_PULSES_KEY, 
    L7_NARROW_CORRIDOR_CUT_PULSES_KEY+"HitMultiplicity", 
    L7_NARROW_CORRIDOR_CUT_PULSES_KEY+"HitStatistics", 
    L7_NARROW_CORRIDOR_CUT_PULSES_KEY+"TimeCharacteristics",
])
L7_NARROW_CORRIDOR_TRACK_ANGLE_DIFF_KEY = L7_NARROW_CORRIDOR_CUT_TRACK_KEY + "_" + L7_DYNEDGE_PARTICLE_KEY + "_angles"
L7_HDF5_KEYS.append( L7_NARROW_CORRIDOR_TRACK_ANGLE_DIFF_KEY )

# Cuts/classifiers
L7_MUON_CLASSIFIER_MODEL_FILE = "$I3_SRC/oscNext/resources/models/level7_GNN/L7_GNN_muon_model_v00.04.joblib"
L7_MUON_MODEL_PREDICTION_KEY = "L7_MuonClassifier_ProbNu"
L7_HDF5_KEYS.extend([L7_MUON_MODEL_PREDICTION_KEY ])

# Coincident muons
L7_COINCIDENT_REJECTION_VARS_KEY = "L7_CoincidentMuon_Variables"
L7_COINCIDENT_REJECTION_BOOL_KEY = "L7_CoincidentMuon_bool"
L7_HDF5_KEYS.extend([L7_COINCIDENT_REJECTION_VARS_KEY, L7_COINCIDENT_REJECTION_BOOL_KEY])

# Photon speed
for suffix in ["PhotonSpeedMetrics", "PhotonDisplacement", "PhotonTimeTaken", "PhotonSpeed"] :
    for prefix in ["AllPhotons", "PhysicalPhotonsOnly"] :
        L7_HDF5_KEYS.append( "L7_%s_%s" % (prefix, suffix) )

# Vertex geom params
L7_VERTEX_GEOM_PARAMS_KEY = "L7_VertexGeomParams"
L7_STOPPING_POINT_GEOM_PARAMS_KEY = "L7_StoppingPointGeomParams"
L7_HDF5_KEYS.extend([ L7_VERTEX_GEOM_PARAMS_KEY, L7_STOPPING_POINT_GEOM_PARAMS_KEY ])

# Containment
L7_CONTAINMENT_METRICS_KEY = "L7_ContainmentMetrics"
L7_HDF5_KEYS.append( L7_CONTAINMENT_METRICS_KEY )

# SANTA direct hits
L7_SANTA_PULSES_KEY = "L7_SANTA_DirectPulses"
L7_HDF5_KEYS.extend([ L7_SANTA_PULSES_KEY, L7_SANTA_PULSES_KEY+"HitMultiplicity", L7_SANTA_PULSES_KEY+"HitStatistics", L7_SANTA_PULSES_KEY+"TimeCharacteristics" ])

# CommonVariables direct hits
L7_DIRECT_HITS_PREFIX = "L7_DynEdge_DirectHits"
L7_DIRECT_HITS_STD_KEY = L7_DIRECT_HITS_PREFIX + "_std"
L7_DIRECT_HITS_SHORT_KEY = L7_DIRECT_HITS_PREFIX + "_short"
L7_HDF5_KEYS.extend([L7_DIRECT_HITS_STD_KEY, L7_DIRECT_HITS_SHORT_KEY])

# Track characteristics
L7_TRACK_CHARACTERISTICS_KEY = "L7_DynEdge_TrackCharacteristics"
L7_HDF5_KEYS.append(L7_TRACK_CHARACTERISTICS_KEY)

# Track light profile
L7_TRACK_LIGHT_PROFILE_PREFIX = "L7_TrackLightProfile"
L7_TRACK_LIGHT_PROFILE_SEGMENTS_KEY = L7_TRACK_LIGHT_PROFILE_PREFIX + "_Segments"
L7_TRACK_LIGHT_PROFILE_TRACK_START_KEY = L7_TRACK_LIGHT_PROFILE_PREFIX + "_TrackStart"
L7_TRACK_LIGHT_PROFILE_TRACK_STOP_KEY = L7_TRACK_LIGHT_PROFILE_PREFIX + "_TrackStop"
L7_TRACK_LIGHT_PROFILE_TRACK_DIR_KEY = L7_TRACK_LIGHT_PROFILE_PREFIX + "_TrackDir"
L7_TRACK_LIGHT_PROFILE_TRACK_LENGTH_KEY = L7_TRACK_LIGHT_PROFILE_PREFIX + "_TrackLength"
L7_TRACK_LIGHT_PROFILE_DN_DX_KEY = L7_TRACK_LIGHT_PROFILE_PREFIX + "_dN_dx"
L7_HDF5_KEYS.extend([
    L7_TRACK_LIGHT_PROFILE_SEGMENTS_KEY,
    L7_TRACK_LIGHT_PROFILE_TRACK_START_KEY,
    L7_TRACK_LIGHT_PROFILE_TRACK_STOP_KEY,
    L7_TRACK_LIGHT_PROFILE_TRACK_DIR_KEY,
    L7_TRACK_LIGHT_PROFILE_TRACK_LENGTH_KEY,
    L7_TRACK_LIGHT_PROFILE_DN_DX_KEY,
])


#
# Reco
#

def oscNext_GNN_L7_DynEdge_postprocessing(frame) :

    from icecube.dataclasses import I3Double, I3MapStringDouble
    # from retro.i3processing.retro_recos_to_i3files import const_en2len, const_en_to_gms_en, GMS_LEN2EN
    from icecube.oscNext.frame_objects.geom import calc_particle_stopping_point, get_deepcore_containment, get_icecube_containment, calc_rho_36


    #
    # Calibrate track length
    #

    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO
    #TODO

    #TODO also energy?



    #
    # Create reco I3Particle
    #

    reco_particle = dataclasses.I3Particle()

    reco_particle.pos = dataclasses.I3Position(
        frame[L6_DYNEDGE_VERTEX_X_KEY].value, 
        frame[L6_DYNEDGE_VERTEX_Y_KEY].value, 
        frame[L6_DYNEDGE_VERTEX_Z_KEY].value,
    )

    reco_particle.dir = dataclasses.I3Direction(
        frame[L6_DYNEDGE_ZENITH_KEY].value, 
        frame[L6_DYNEDGE_AZIMUTH_KEY].value,
    )

    reco_particle.time = np.NaN # frame[L6_DYNEDGE_VERTEX_TIME_KEY].value  #TODO
    reco_particle.energy = frame[L6_DYNEDGE_ENERGY_KEY].value
    reco_particle.length = frame[L6_DYNEDGE_LENGTH_KEY].value        #TODO calibrated version

    reco_particle.shape = reco_particle.ParticleShape.StartingTrack
    reco_particle.fit_status = reco_particle.FitStatus.OK
    reco_particle.location_type = reco_particle.LocationType.InIce

    frame[L7_DYNEDGE_PARTICLE_KEY] = reco_particle


    #
    # Compute derived position information
    #

    # Stopping point
    reco_stopping_point = calc_particle_stopping_point(particle=reco_particle)
    frame[L7_DYNEDGE_STOPPING_POINT_KEY] = reco_stopping_point

    # Vertex & stopping point rho
    frame[L7_DYNEDGE_VERTEX_RHO36_KEY] = dataclasses.I3Double( calc_rho_36(x=frame[L6_DYNEDGE_VERTEX_X_KEY].value, y=frame[L6_DYNEDGE_VERTEX_Y_KEY].value) )
    frame[L7_DYNEDGE_STOPPING_POINT_RHO36_KEY] = dataclasses.I3Double( calc_rho_36(x=reco_stopping_point.x, y=reco_stopping_point.y) )


    #
    # Containment
    #

    # Starting containment
    dc_starting_containment = get_deepcore_containment(x=reco_particle.pos.x, y=reco_particle.pos.y, z=reco_particle.pos.z) 
    ic_starting_containment = get_icecube_containment(x=reco_particle.pos.x, y=reco_particle.pos.y, z=reco_particle.pos.z) 

    # Stopping containment
    dc_stopping_containment = get_deepcore_containment(x=reco_stopping_point.x, y=reco_stopping_point.y, z=reco_stopping_point.z) 
    ic_stopping_containment = get_icecube_containment(x=reco_stopping_point.x, y=reco_stopping_point.y, z=reco_stopping_point.z) 

    frame[L7_DYNEDGE_DC_STARTING_CONTAINMENT_KEY] = I3Bool(bool(dc_starting_containment))
    frame[L7_DYNEDGE_IC_STARTING_CONTAINMENT_KEY] = I3Bool(bool(ic_starting_containment))
    frame[L7_DYNEDGE_DC_STOPPING_CONTAINMENT_KEY] = I3Bool(bool(dc_stopping_containment))
    frame[L7_DYNEDGE_IC_STOPPING_CONTAINMENT_KEY] = I3Bool(bool(ic_stopping_containment))


    #
    # Uncertainty
    #

    # Convert kappa -> sigma
    frame[L7_DYNEDGE_ZENITH_SIGMA_KEY] = dataclasses.I3Double( calc_sigma_from_kappa(frame[L6_DYNEDGE_ZENITH_KAPPA_KEY].value) )
    frame[L7_DYNEDGE_AZIMUTH_SIGMA_KEY] = dataclasses.I3Double( calc_sigma_from_kappa(frame[L6_DYNEDGE_AZIMUTH_KAPPA_KEY].value) )

    #TODO overall direction sigma



    #
    # Track/cascade length/energy
    #

    #TODO length -> track_energy, then also cascade energy


    #
    # Correct energy bias
    #

    #TODO


    #
    # Inelasticity
    #

    # This is the fraction of energy transferred to the nucleus in the interaction,
    # which for our purpose is the track/total energy (for numu CC events at least,
    # can't determine this for cascade signatures)

    #TODO

    # inelasticity = cascade_energy / total_energy
    # frame[L7_FINAL_RECO_INELASTICITY_KEY] = I3Double(inelasticity)



@icetray.traysegment
def oscNext_GNN_L7_compute_direct_hits(
    tray,
    name,
) :
    '''
    Check which pulses are "direct" (unscattered), given the final level track reco.
    These pulses should be less affected by ice propeties.

    This function was taken from Maria Liubarska's L8 selection level than she applied on 
    top of this L7, for her oscNext inelasticity analysis.
    '''

    from icecube.santa import SANTA_cleanSegment
    from icecube.common_variables import direct_hits
    from icecube.oscNext.frame_objects.pulses import pulse_info


    #
    # Compute direct hits using SANTA
    #

    # Run SANTA cleaning on the new cleaned pulse map
    # Using asme settings as L5 SANTA cleaning, just with different input pulse map

    # Define settings
    santa_cleaning_settings = {
        'LoopLevels'              : 6,
        'AmplitudeCut_HS'         : 1.,
        'FirstPulseThreshold'     : 0.1,
        'TimeDelay'               : 20. * icetray.I3Units.ns,
        'DC_only'                 : False,
        'CausalityOnly'           : True,
        'MinStringHits'           : 3,
        # 'Interactive'             : False,
        'Debugging'               : False,
        'UsePreCleaner'           : True, #XXX: this has to be turned off after Wavedeform fix
    }

    # Add cleaning segment
    tray.AddSegment(
        SANTA_cleanSegment, 
        L7_SANTA_PULSES_KEY,
        OutputPulseSeries=L7_SANTA_PULSES_KEY,
        InputPulseSeries=L6_PRE_RECO_CLEANED_PULSES_KEY,
        Interactive=False,
        **santa_cleaning_settings)

    # Add additional pulse info/stat calculators
    tray.Add(pulse_info, "oscNext_pulse_info_%s"%L7_SANTA_PULSES_KEY, pulses=L7_SANTA_PULSES_KEY)  


    #
    # Compute direct hits using CommonVariables package
    #

    #TODO Removed this for now since the module fails because there is no time reconsutrction in DynEdge currently. Can replace this once time is aded to our reco.

    # # Direct hits definitions
    # # These values were chosen by Maria, have not revisited them myself
    # dh_defs = [
    #     direct_hits.I3DirectHitsDefinition("_std", -10*I3Units.ns, 300*I3Units.ns),
    #     direct_hits.I3DirectHitsDefinition("_short", -2.5*I3Units.ns, 100*I3Units.ns),
    # ]

    # # Add the module
    # tray.AddModule(
    #     direct_hits.I3DirectHitsCalculator,
    #     L7_DIRECT_HITS_PREFIX,
    #     DirectHitsDefinitionSeries=dh_defs,
    #     ParticleName=L7_DYNEDGE_PARTICLE_KEY,
    #     PulseSeriesMapName=L6_PRE_RECO_CLEANED_PULSES_KEY,
    #     OutputI3DirectHitsValuesBaseName=L7_DIRECT_HITS_PREFIX,
    # )

    return


@icetray.traysegment
def oscNext_GNN_L7_compute_track_light_profile(
    tray,
    name,
):
    '''
    Reconstruct light profile along track
    '''

    from icecube.oscNext.frame_objects.reco import calc_track_light_profile

    #TODO Add a smoothness metric?

    tray.AddModule(
        calc_track_light_profile,
        L7_TRACK_LIGHT_PROFILE_PREFIX, 
        pulse_map_name=L6_PRE_RECO_CLEANED_PULSES_KEY,
        particle_key=L7_DYNEDGE_PARTICLE_KEY,
        max_distance_backward=200., # [m] 
        max_distance_forward=750., # [m] 
        num_segments=95, # 10 m segments
        output_key=L7_TRACK_LIGHT_PROFILE_PREFIX,
        start_margin=0.2, # Ignore first 20% of track for dN/dx calculation (account for uncertainty in vertex and avoiding light from break up of nucelus)
        stop_margin=0.1, # Ignore last 10% of track for dN/dx calculation (account for uncertainty in stop point)
    )

    return


def oscNext_GNN_L7_compute_vertex_geom_params(
    frame,
):
    '''
    Calculate various variables characterising the reco vertex and stipping positions wr.t. sensor positions
    '''

    cleaned_pulses = L6_PRE_RECO_CLEANED_PULSES_KEY

    # Calc for vertex
    compute_event_vertex_geometry_params(
        frame=frame,
        pulse_series_name=cleaned_pulses,
        vertex=L7_DYNEDGE_PARTICLE_KEY,
        output_key=L7_VERTEX_GEOM_PARAMS_KEY,
    )

    # Calc for stopping point
    compute_event_vertex_geometry_params(
        frame=frame,
        pulse_series_name=cleaned_pulses,
        vertex=L7_DYNEDGE_STOPPING_POINT_KEY,
        output_key=L7_STOPPING_POINT_GEOM_PARAMS_KEY,
    )


#
# Coicident events
#

def calculate_coincident_cut_containment(frame, pulsesname, outputname):
    '''Calculate containment variables for a given hit map.
    These variables are used for co-incident muon rejection.
    '''
    # taken from http://wiki.icecube.wisc.edu/index.php/Deployment_order
    ic86 = [21, 29, 39, 38, 30, 40, 50, 59, 49, 58, 67, 66, 74, 73, 65, 72, 78, 48, 57, 47,
            46, 56, 63, 64, 55, 71, 70, 76, 77, 75, 69, 60, 68, 61, 62, 52, 44, 53, 54, 45,
            18, 27, 36, 28, 19, 20, 13, 12, 6, 5, 11, 4, 10, 3, 2, 83, 37, 26, 17, 8, 9, 16,
            25, 85, 84, 82, 81, 86, 35, 34, 24, 15, 23, 33, 43, 32, 42, 41, 51, 31, 22, 14,
            7, 1, 79, 80]
    deep_core_strings = [79, 80, 81, 82, 83, 84, 85, 86]
    outer_strings = [31, 41, 51, 60, 68, 75, 76, 77, 78, 72, 73, 74, 67, 59, 50,
                     40, 30, 21, 13, 6, 5, 4, 3, 2, 1, 7, 14, 22]
    vars = dataclasses.I3MapStringDouble()
    vars['n_top15'] = 0.
    vars['n_top10'] = 0.
    vars['n_top5'] = 0.
    vars['z_travel_top15'] = np.NaN # NaN if not enough pulses to compute it
    vars['n_outer'] = 0.
    if type(frame[pulsesname]) == dataclasses.I3RecoPulseSeriesMap:
        hit_map = frame.Get(pulsesname)
    elif type(frame[pulsesname]) == dataclasses.I3RecoPulseSeriesMapMask:
        hit_map = frame[pulsesname].apply(frame)
    omgeo = frame['I3Geometry'].omgeo
    z_pulses = []
    t_pulses = []
    for om in hit_map.keys():
        if om.string in ic86 and om.string not in deep_core_strings:
            if om.om <= 15:
                vars['n_top15'] += 1.
                z_pulses.append(omgeo[om].position.z)
                t_pulses.append(np.min([p.time for p in hit_map[om]]))
            if om.om <= 10:
                vars['n_top10'] += 1.
            if om.om <= 5:
                vars['n_top5'] += 1.
        if om.string in outer_strings:
            vars['n_outer'] += 1.
    z_pulses = np.array(z_pulses)
    z_pulses = z_pulses[np.argsort(t_pulses)]
    if len(z_pulses) >= 4:
        len_quartile = np.floor(len(z_pulses)/4)
        mean_first_quartile = np.mean(z_pulses[:int(len_quartile)])
        vars['z_travel_top15'] = np.mean(z_pulses - mean_first_quartile)
    frame[outputname] = vars
    return True


@icetray.traysegment
def oscNext_GNN_L7_CoincidentCut(
    tray,
    name,
):
    '''
    Calculate co-incident muon rejection variables and a boolean corresponding to
    optimized cut values.
    '''

    tray.AddModule(
        calculate_coincident_cut_containment, "containment_vars_cleaned",
        pulsesname=CLEANED_PULSES, # Using old pulse cleaning here still since it doesn't impose a fiducial region cut
        outputname=L7_COINCIDENT_REJECTION_VARS_KEY,
    )

    def coincident_cut(frame) :
        passed_cut = frame[L7_COINCIDENT_REJECTION_VARS_KEY]['z_travel_top15'] >= 0.
        passed_cut *= frame[L7_COINCIDENT_REJECTION_VARS_KEY]['n_outer'] < 8
        frame[L7_COINCIDENT_REJECTION_BOOL_KEY] = icetray.I3Bool(passed_cut)

    tray.Add(coincident_cut, "L7_coincident_cut")



@icetray.traysegment
def oscNext_GNN_L7_CorridorCuts( 
    tray,
    name, 

) :
    #
    # Corridor cut (narrow)
    #

    # Compute a version of the corridor cut with much tighter settings than used at L5
    # This is more like LEESARD/DRAGON samples

    from icecube.oscNext.frame_objects.corridor_cut import CorridorCut
    from icecube.oscNext.frame_objects.pulses import pulse_info
    # from icecube.oscNext.selection.oscNext_L5 import AngleCorrelation

    # Run the module
    tray.AddModule(
        CorridorCut,
        "L7_NarrowCorridorCut",
        InputPulseSeries = UNCLEANED_PULSES,
        #SANTAFit = "", #TODO?
        NoiselessPulseSeries = L6_PRE_RECO_CLEANED_PULSES_KEY,
        OutputPulseSeries    = L7_NARROW_CORRIDOR_CUT_PULSES_KEY,
        HitCounter           = L7_NARROW_CORRIDOR_CUT_COUNT_KEY,
        OutputTrack          = L7_NARROW_CORRIDOR_CUT_TRACK_KEY,
        Radius               = 75. * I3Units.m, # LEESARD/DRAGON-like settings
        WindowMinus          = -150. * I3Units.ns, # LEESARD/DRAGON-like settings
        WindowPlus           = +250. * I3Units.ns, # LEESARD/DRAGON-like settings
        ZenithSteps          = 0.02,
    )
    
    # Calc hit info for the corridor pulses
    tray.Add( pulse_info, "oscNext_pulse_info_%s"%L7_NARROW_CORRIDOR_CUT_PULSES_KEY, pulses=L7_NARROW_CORRIDOR_CUT_PULSES_KEY  )

    '''
    # Calculating correlation between corridor track and reconstructed direction
    tray.AddModule(
        AngleCorrelation, 
        particleAname = L7_NARROW_CORRIDOR_CUT_TRACK_KEY, 
        particleBname = RETRO_RECO_PARTICLE_KEY, 
        outdictname   = L7_NARROW_CORRIDOR_TRACK_ANGLE_DIFF_KEY
    )
    '''
    #TODO also recompute wide angle correlation with retro reco direction??

    return


#
# Photon speed
#

@icetray.traysegment
def oscNext_GNN_L7_photon_speed_metrics(
    tray,
    name,
) :
    '''
    Compute the superluminal photon variables for oscNext L7
    '''

    from icecube.oscNext.frame_objects.photons import photon_speed_metrics, SPEED_OF_LIGHT_ICE

    # Get the vertex
    vertex_x    = L6_DYNEDGE_VERTEX_X_KEY
    vertex_y    = L6_DYNEDGE_VERTEX_Y_KEY
    vertex_z    = L6_DYNEDGE_VERTEX_Z_KEY
    vertex_time = L6_DYNEDGE_VERTEX_TIME_KEY
    
    # Use cleaned pulses
    pulses = L6_PRE_RECO_CLEANED_PULSES_KEY

    # Choose the superluminal threshold
    superluminal_threshold = SPEED_OF_LIGHT_ICE

    # Trying both all pulses (useful for noise) and only those with physical speeds (useful for track ID)
    for trim_unphysical, key in zip([False, True], ["AllPhotons", "PhysicalPhotonsOnly"]) :

        # Add module
        tray.Add(
            photon_speed_metrics,
            "photon_speed_metrics_%s"%key,
            pulses=pulses, 
            vertex_x=vertex_x,
            vertex_y=vertex_y,
            vertex_z=vertex_z,
            vertex_time=vertex_time,
            output_prefix="L7_%s"%key,
            superluminal_threshold=superluminal_threshold,
            trim_unphysical=trim_unphysical,
        )


#
# 
#

@icetray.traysegment
def oscNext_GNN_L7_containment_metrics(
    tray,
    name,
) :
    '''
    Compute some non-reco based containment variables, based on comparing 
    fiducial vs non-fiducial DOM hits.
    '''

    #
    # Re-run pulse cleaning with non-fiducial DOMs included
    #

    # Use default cleaning settings, but with disabling the "fiducial-only" settings
    from icecube.oscNext.selection.oscNext_GNN_L6 import PULSE_CLEANING_SETTINGS
    cleaning_kwargs = copy.deepcopy(PULSE_CLEANING_SETTINGS)
    cleaning_kwargs["fiducial_doms_only"] = False

    # Define new pulse map
    new_pulse_map = L6_PRE_RECO_CLEANED_PULSES_KEY + "_with_non_fid"

    # Add cleaning to tray
    tray.Add(
        prereco_pulse_cleaning,
        new_pulse_map,
        input_pulse_map_key="SplitInIcePulses",
        output_key=new_pulse_map, 
        **cleaning_kwargs
    )


    #
    # Variables based on fiducial vs non-fiducial hits
    #

    def compare_fid_with_non_fid_pulses(frame, pulse_map_key, output_key) : #TODO move to geom.py ?

        from icecube.oscNext.frame_objects.geom import get_dom_geometry, calc_rho_36, is_dom_in_deepcore_fiducial

        # Checks
        assert frame.Has(pulse_map_key)
        assert not frame.Has(output_key)

        # Load cleaned pulse map that includes non-fiducial pulses
        pulse_map = frame[pulse_map_key]

        # Init vars
        num_hit_doms_fid, num_hit_doms_non_fid = 0, 0

        # Loop over OMs
        for om_key, pulses in pulse_map.items() :

            # Skip DOM if empty
            if len(pulses) == 0 :
                continue

            # Get DOM geometry
            om_geom = get_dom_geometry(frame=frame, om_key=om_key)
            om_z = om_geom.position.z
            om_rho = calc_rho_36(x=om_geom.position.x, y=om_geom.position.y)

            # Check fiducial status (geometric, not DOM list)
            fid = is_dom_in_deepcore_fiducial(frame=frame, om_key=om_key, use_dom_list=False)

            # Apply a larger containment around DeepCore. Anything outside of this is unlikely to be associated with a 
            # DeepCore event (e.g. just noise or coincidences), so ignore these
            if ( om_z > 200. ) or ( om_rho > 300. ) :
                continue

            # Update counters
            if fid :
                num_hit_doms_fid += 1
            else :
                num_hit_doms_non_fid += 1

            #TODO consider causality requirement

        # Post-process
        fraction_non_fid_hit_doms = float(num_hit_doms_non_fid) / float(num_hit_doms_fid + num_hit_doms_non_fid)

        # Write to frame
        out = dataclasses.I3MapStringDouble()
        out["num_hit_doms_fid"] = float(num_hit_doms_fid)
        out["num_hit_doms_non_fid"] = float(num_hit_doms_non_fid)
        out["fraction_non_fid_hit_doms"] = fraction_non_fid_hit_doms
        frame[output_key] = out


    # Add the module to the tray
    tray.Add(
        compare_fid_with_non_fid_pulses,
        "compare_fid_with_non_fid_pulses",
        pulse_map_key=new_pulse_map, 
        output_key=L7_CONTAINMENT_METRICS_KEY,
    )



#
# Other stuff
#

@icetray.traysegment
def oscNext_GNN_L7_compute_cut( tray, name, ):
    '''
    Calculate the cut(s) to apply at this processing level
    '''

    from icecube.oscNext.tools.classifier import I3Classifier


    #
    # Atmospheric muon classifier
    #

    # Add muon classifiers to the tray
    muon_model_file = os.path.expandvars(L7_MUON_CLASSIFIER_MODEL_FILE)
    muon_classifier = I3Classifier(model_file=muon_model_file, class_key="neutrino", output_key=L7_MUON_MODEL_PREDICTION_KEY)
    tray.Add(muon_classifier, "oscNext_GNN_L7_muon_classifier")


    #
    # Make the overall cut
    #

    #TODO

    # # Create function defining the cut
    # def overall_cut(frame) :

    #     # Data quality
    #     # Remove coincident events (not simulated)
    #     data_quality_cut = frame[L7_COINCIDENT_REJECTION_BOOL_KEY].value

    #     # Muons
    #     # Cut on classifier prediction (0.7 gives a muon rate that is ~ the same as the nutau rate)
    #     # muon_cut = frame[L7_MUON_MODEL_PREDICTION_KEY_FULLSKY].value >= 0.7

    #     # Put it all together and add to frame
    #     frame[L7_CUT_BOOL_KEY] = icetray.I3Bool(data_quality_cut & reco_success)

    # # Add to the tray
    # tray.Add(overall_cut,"L7_overall_cut")

    return


#
# Top-level traysegment
#

@icetray.traysegment
def oscNext_GNN_L7( 
    tray, name, 
    uncleaned_pulses,
    cleaned_pulses,
):
    '''
    This is the main oscNext L7 traysegment
    '''

    #
    # Apply cuts on input data
    #

    # Only keep frames passing L6
    tray.Add(oscNext_cut, "L6_cut", processing_level=6)
    

    #
    # Compute L7 variables
    #

    # Reco post-processing
    tray.Add(
        oscNext_GNN_L7_DynEdge_postprocessing,
        'oscNext_GNN_L7_DynEdge_postprocessing',
    )

    # Find coincident events (unsimulated)
    tray.Add(
        oscNext_GNN_L7_CoincidentCut,
        'oscNext_GNN_L7_CoincidentCut',
    )

    # Geometry variables
    tray.AddModule(
        oscNext_GNN_L7_compute_vertex_geom_params,
        'oscNext_GNN_L7_compute_vertex_geom_params',
    )

    # Calculate direct pulses from the reco track
    tray.Add(
        oscNext_GNN_L7_compute_direct_hits,
        'oscNext_GNN_L7_compute_direct_hits',
    )

    # Track characteristics
    tray.AddModule( I3TrackCharacteristicsCalculator, 
        L7_TRACK_CHARACTERISTICS_KEY, 
        PulseSeriesMapName=L6_PRE_RECO_CLEANED_PULSES_KEY, # Using cleaned pulses (these are the input to the reco) 
        ParticleName=L7_DYNEDGE_PARTICLE_KEY, # Using the post-processed particle
        TrackCylinderRadius=150.*I3Units.m, #TODO what value?
        OutputI3TrackCharacteristicsValuesName=L7_TRACK_CHARACTERISTICS_KEY, 
    )

    # Reconstruct track light profile
    tray.Add( 
        oscNext_GNN_L7_compute_track_light_profile,
        'oscNext_GNN_L7_compute_track_light_profile',
    )

    # Corridor cut
    # Compute a version of the corridor cut with much narrower corridor definitions compared to L5
    # This is more like what was done in DRAGON/GRECO
    tray.Add(
        oscNext_GNN_L7_CorridorCuts,
        'oscNext_GNN_L7_NarrowCorrCuts',
    )

    # Photon speeds     #TODO replace this once have aded reco time model
    # tray.Add(
    #     oscNext_GNN_L7_photon_speed_metrics,
    #     "oscNext_GNN_L7_photon_speed_metrics",
    # )

    # Containment metrics
    tray.Add(
        oscNext_GNN_L7_containment_metrics,
        'oscNext_GNN_L7_containment_metrics',
    )


    #
    # L7 cut
    #

    # Run classifier predictions and compute cut
    tray.Add(
        oscNext_GNN_L7_compute_cut, 
        "oscNext_GNN_L7_cut", 
    )


    #
    # Done
    #

    # Can dump frame for debugging
    if False :
        tray.Add("Dump","Dump")

    return