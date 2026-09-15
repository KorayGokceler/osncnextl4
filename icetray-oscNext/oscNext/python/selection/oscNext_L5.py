'''
The oscNext level 5 event selection traysegment

Tom Stuttard, Andrii Terliuk
'''

import math

from icecube import dataclasses
from icecube import icetray
from icecube.icetray import I3Units
from icecube import lilliput
import icecube.lilliput.segments

from icecube.oscNext.tools.data_quality import check_object_exists
from icecube.oscNext.selection.globals import *
from icecube.oscNext.selection.oscNext_cuts import oscNext_cut
from icecube.oscNext.frame_objects.geom import calc_rho_36


#
# Globals
#

# Define all output frame objects
L5_HDF5_KEYS = []

# Cut
L5_HDF5_KEYS.append( L5_CUT_BOOL_KEY )

# Containment
L5_BRIGHT_STR_POS_KEY = "L5_BrStrPos"
L5_BRIGHT_STR_RHO_KEY = "L5_BrStrRho"
L5_HDF5_KEYS.extend([L5_BRIGHT_STR_POS_KEY, L5_BRIGHT_STR_RHO_KEY])

L5_VERTEX_GUESS_RHO_KEY = "L5_VertexGuessRho" 
L5_HDF5_KEYS.append( L5_VERTEX_GUESS_RHO_KEY ) 

# SPE fit
n_spe_iterations = 11
L5_SPE_FIT_KEY = "L5_SPEFit%i" % n_spe_iterations
L5_HDF5_KEYS.append( L5_SPE_FIT_KEY )


# Corridor cut
# Include hit stats, etc for the output pulse series
L5_WIDE_CORRIDOR_CUT_PULSES_KEY = "L5_WideCorridorCutPulses"
L5_WIDE_CORRIDOR_CUT_COUNT_KEY = "L5_WideCorridorCutCount"
L5_WIDE_CORRIDOR_CUT_TRACK_KEY  = "L5_WideCorridorCutTrack"
L5_HDF5_KEYS.extend([ L5_WIDE_CORRIDOR_CUT_COUNT_KEY, L5_WIDE_CORRIDOR_CUT_TRACK_KEY ])
L5_HDF5_KEYS.extend([ L5_WIDE_CORRIDOR_CUT_PULSES_KEY, L5_WIDE_CORRIDOR_CUT_PULSES_KEY+"HitMultiplicity", L5_WIDE_CORRIDOR_CUT_PULSES_KEY+"HitStatistics", L5_WIDE_CORRIDOR_CUT_PULSES_KEY+"TimeCharacteristics" ])

# Corridor cut - SPE fit correlation
L5_WIDE_CORRIDOR_TRACK_ANGLE_DIFF_KEY = L5_WIDE_CORRIDOR_CUT_TRACK_KEY + "_" + L5_SPE_FIT_KEY + "_angles"
L5_HDF5_KEYS.append( L5_WIDE_CORRIDOR_TRACK_ANGLE_DIFF_KEY )

# Hit cleaning
# Note that have one output per confiration passed (defaults to 4)
L5_DIRECT_HITS_KEY  = "L5_DirectHits"
L5_HDF5_KEYS.extend([ L5_DIRECT_HITS_KEY+x for x in ["A","B","C","D"] ])

# SANTA cleaning
# Include hit stats, etc for the output pulse series
L5_SANTA_PULSES_KEY = 'L5_SANTA_DirectPulses'
L5_HDF5_KEYS.extend([ L5_SANTA_PULSES_KEY, L5_SANTA_PULSES_KEY+"HitMultiplicity", L5_SANTA_PULSES_KEY+"HitStatistics", L5_SANTA_PULSES_KEY+"TimeCharacteristics" ])



#TODO Migrate


# #
# # Useful functions
# #

# def AngleCorrelation(
#     frame, 
#     particleAname,
#     particleBname, 
#     outdictname = None,
# ):
#     '''
#     Compute the angualr correlation between two directions, of two I3Particle instances
#     Can compare two reco resulrs for example
#     '''

#     #TODO move somewhere common

#     from numpy import pi, isnan

#     dirA =  frame[particleAname].dir
#     dirB =  frame[particleBname].dir
#     outdict = dataclasses.I3MapStringDouble()
#     outdict['total_cos_diff']  = dirA*dirB
#     if isnan(outdict['total_cos_diff']):
#         outdict['total_cos_diff'] = -1.1 ## Avoid undefined behaviour when cutting
#     outdict['zenith_diff']     = dirA.zenith - dirB.zenith
#     if isnan(outdict['zenith_diff']):
#         outdict['zenith_diff'] = 4.0 ## Avoid undefined behaviour when cutting        
#     outdict['abs_zenith_diff'] = abs(outdict['zenith_diff'] )
#     outdict['azimuth_diff']     = dirA.azimuth - dirB.azimuth 
#     if outdict['azimuth_diff']  > pi : 
#         outdict['azimuth_diff'] = outdict['azimuth_diff']  - pi 
#     elif outdict['azimuth_diff']  < - pi : 
#         outdict['azimuth_diff'] = outdict['azimuth_diff']  + pi 
#     outdict['abs_azimuth_diff'] = abs(outdict['azimuth_diff'])
#     if isnan(outdict['azimuth_diff']):        
#         outdict['azimuth_diff'] = 10.0 
#     if isnan(outdict['abs_azimuth_diff'] ):
#         outdict['abs_azimuth_diff'] =  10.0
#     #### 
#     frame[outdictname] = outdict


# #
# # Tray segments
# #

# def compute_L5_cut(frame):
#     '''
#     Compute the actual L5 cut boolean
#     '''
#     radial_containment = ( (frame[L5_BRIGHT_STR_RHO_KEY].value < 150.0) *
#                            (frame["L4_first_hlc_rho"].value  < 150.0  ) * 
#                            (frame[L5_VERTEX_GUESS_RHO_KEY].value < 150.0 )   )

#     upper_veto         = ( (frame["L4_first_hlc"].pos.z < -220.0 )* 
#                            (frame['IC2018_LE_L3_Vars']['VertexGuessZ'] < -220.0 ) )

#     bottom_veto        = ( (frame["L4_first_hlc"].pos.z > -490.0 )* 
#                            (frame['IC2018_LE_L3_Vars']['VertexGuessZ'] > -490.0 ) )

#     #TODO This corridor cut is WRONG. Should be only rejectimg events with ( N > 2 and total_cos_diff > 0.7 ). Should fix this is re-process.
#     corridor_cut       = ( (frame[L5_WIDE_CORRIDOR_CUT_COUNT_KEY].value < 2.01 ) * 
#                            (frame[L5_WIDE_CORRIDOR_TRACK_ANGLE_DIFF_KEY]['total_cos_diff'] < 0.7) ) 

#     tighten_L4         = ( (frame['L4_MuonClassifier_Data_ProbNu'].value > 0.90 ) *
#                            (frame['L4_NoiseClassifier_ProbNu'].value     > 0.85) )

#     frame[L5_CUT_BOOL_KEY] = icetray.I3Bool(radial_containment * 
#                                             upper_veto * 
#                                             bottom_veto * 
#                                             corridor_cut * 
#                                             tighten_L4)
    
#     return True


# @icetray.traysegment
# def oscNext_L5_atm_muon_variables( 
#     tray,
#     name, 
#     uncleaned_pulses,
#     cleaned_pulses,
# ) :
#     '''
#     Create all variables to be used by the L5 muon classifier
#     This is mainly stuff that was too slow for L4
#     '''

#     #
#     # Finding brighthest string for cleaned pulses
#     # 

#     # This is used to make some loose containment cuts

#     from icecube.oscNext.frame_objects.pulses import find_brightest_string

#     tray.AddModule(
#         find_brightest_string, 
#         pulsesname   = cleaned_pulses, 
#         position_key = L5_BRIGHT_STR_POS_KEY, 
#         rho_key      = L5_BRIGHT_STR_RHO_KEY,
#     ) 


#     #
#     # Finding vertex guess rho value
#     #

#     # This is used to make some loose containment cuts


#     def VertexGuessRho(frame, key = ""):
#         rho_val = calc_rho_36(x=frame['IC2018_LE_L3_Vars']['VertexGuessX'], y=frame['IC2018_LE_L3_Vars']['VertexGuessY'])
#         frame[key] = dataclasses.I3Double(rho_val)
#         return True
#     tray.AddModule(VertexGuessRho, key = L5_VERTEX_GUESS_RHO_KEY)
   


#     #
#     # SPE reconstruction
#     #

#     # Run a medium speed reconstruction now the event rate is lower
#     # Can use this for cuts, and also as a seed to final level reconstructions

#     tray.AddSegment( 
#         lilliput.segments.I3IterativePandelFitter, 
#         L5_SPE_FIT_KEY,
#         pulses = cleaned_pulses,
#         n_iterations = n_spe_iterations,
#         seeds = ['SPEFit2_DC'], #TODO Line fit?
#         # OutputName=L5_SPE_FIT_KEY,
#     )     


#     #
#     # Corridor cut (wide)
#     #

#     # This is used to identify muons entering via corridors
#     # Using much looser/wider setting that previous samples
#     # Example older settings are (LEESARD/DRAGON-like):     
#     #     Radius        =    75. * I3Units.m
#     #     WindowMinus   =  -150. * I3Units.ns
#     #     WindowPlus    =  +250. * I3Units.ns
#     #     ZenithSteps   =  0.02,

#     from icecube.veto_tools.CorridorCut import CorridorCut
#     from icecube.oscNext.frame_objects.pulses import pulse_info

#     # Run the module
#     tray.AddModule(
#         CorridorCut,
#         "L5_WideCorridorCut",
#         InputPulseSeries = uncleaned_pulses,
#         #SANTAFit = "", #TODO?
#         NoiselessPulseSeries = cleaned_pulses,
#         OutputPulseSeries = L5_WIDE_CORRIDOR_CUT_PULSES_KEY,
#         HitCounter = L5_WIDE_CORRIDOR_CUT_COUNT_KEY,
#         OutputTrack = L5_WIDE_CORRIDOR_CUT_TRACK_KEY,
#         Radius        =   250. * I3Units.m,
#         WindowMinus   = -1000. * I3Units.ns,
#         WindowPlus    =  1000. * I3Units.ns,
#         ZenithSteps   =  0.02,
#     )

#     # Calc hit info for the corridor pulses
#     tray.Add( pulse_info, "oscNext_pulse_info_%s"%L5_WIDE_CORRIDOR_CUT_PULSES_KEY, pulses=L5_WIDE_CORRIDOR_CUT_PULSES_KEY  )


#     #
#     # Calculating correlation between corridors and SPEFit
#     #

#     # Check if track found by corridor cut is in thr same direction as the SPE fit track
#     # If it isn't, the corridor pulse may well be noise rather than a sign of a muon sneaking through
#     # Can reduce the amount of neutrinos we loose using this
    
#     # Compute for wide version
#     tray.AddModule(AngleCorrelation, 
#                     particleAname = L5_WIDE_CORRIDOR_CUT_TRACK_KEY, 
#                     particleBname = L5_SPE_FIT_KEY, 
#                     outdictname   = L5_WIDE_CORRIDOR_TRACK_ANGLE_DIFF_KEY
#                     )
    

#     #
#     # Direct hits from SPEFit11
#     #

#     # Calculate direct hits
#     # Use best reco so far as particle hypothesis (currently SPE11)

#     #TODO This is currently untested, use with caution
    
#     from icecube.common_variables import direct_hits

#     tray.AddModule( 
#         direct_hits.I3DirectHitsCalculator, 
#         L5_DIRECT_HITS_KEY, 
#         # DirectHitsDefinitionSeries=XXX, # Can specify parameters here, and multiple sets of them if desired. Currently just using defaults until we have time to study later...
#         ParticleName=L5_SPE_FIT_KEY,
#         PulseSeriesMapName=cleaned_pulses, 
#         OutputI3DirectHitsValuesBaseName=L5_DIRECT_HITS_KEY,
#         # PyLogLevel=XXX, #TODO What value?
#     )

    
#     # 
#     # Calculating SANTA direct hit cleaning
#     #

#     # These are not currently used by the cut, but can be used for down-stream stages
    
#     from icecube import santa

#     santa_cleaning_settings = {
#         'OutputPulseSeries'       : L5_SANTA_PULSES_KEY,
#         'LoopLevels'              : 6,
#         'AmplitudeCut_HS'         : 1.,
#         'FirstPulseThreshold'     : 0.1,
#         'TimeDelay'               : 20. * icetray.I3Units.ns,
#         'DC_only'                 : False,
#         'CausalityOnly'           : True,
#         'MinStringHits'           : 3,
#         # 'Interactive'             : False,
#         'Debugging'               : False,
#         'UsePreCleaner'           : True, #XXX: this has to be turned off after Wavedeform fix
#     }

#     tray.AddSegment(santa.SANTA_cleanSegment, 'SANTADirectPulses',
#                     InputPulseSeries=cleaned_pulses,
#                     Interactive=False,
#                     **santa_cleaning_settings)


#     # Calculate hit stats for the SANTA cleaned pulses
#     tray.Add(pulse_info, "oscNext_pulse_info_%s"%L5_SANTA_PULSES_KEY, pulses=L5_SANTA_PULSES_KEY)  
    

#     return



# @icetray.traysegment
# def oscNext_L5( tray, name, uncleaned_pulses, cleaned_pulses ):
#     '''
#     This is the main oscNext L5 traysegment
#     '''

#     raise Exception("L5 not yet migrated to GitHub IceTray")


#     #
#     # Apply L4 cut
#     #

#     # Only keep frames passing L4
#     tray.Add(oscNext_cut,"L4_cut",processing_level=4)


#     #
#     # Calculate variables
#     #

#     tray.Add(
#         oscNext_L5_atm_muon_variables, 
#         "oscNext_L5_atm_muon_variables", 
#         uncleaned_pulses=uncleaned_pulses,
#         cleaned_pulses=cleaned_pulses,
#     )


#     #
#     # Define cuts
#     #

#     # Compute the cut boolean
#     tray.AddModule(compute_L5_cut,"compute_L5_cut")
    

#     #
#     # Done
#     #

#     # Can dump frame for debugging
#     if False :
#         tray.Add("Dump","Dump")

#     return
