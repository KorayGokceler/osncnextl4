'''
The oscNext level 4 event selection tray segment

Tom Stuttard
'''

import math, os

from icecube import dataclasses
from icecube import icetray
from icecube import DomTools
from icecube import linefit
# from icecube import SeededRTCleaning
from icecube import tensor_of_inertia
from icecube.icetray import I3Units
from icecube import DeepCore_Filter
from icecube.DeepCore_Filter import DOMS

from icecube.oscNext.tools.data_quality import check_object_exists
from icecube.oscNext.selection.globals import *
from icecube.oscNext.selection.oscNext_cuts import oscNext_cut
from icecube.oscNext.frame_objects.geom import calc_rho_36


#
# Output frame objects
#

# Define all output frame objects
L4_HDF5_KEYS = []

# Cut
L4_HDF5_KEYS.append( L4_CUT_BOOL_KEY )

# Common
L4_FIRST_HLC_KEY = "L4_first_hlc"
L4_FIRST_HLC_RHO_KEY = L4_FIRST_HLC_KEY + "_rho"
L4_HDF5_KEYS.extend([ L4_FIRST_HLC_KEY, L4_FIRST_HLC_RHO_KEY ])

# Muon rejection
L4_TOI_KEY = "L4_ToI"
L4_HDF5_KEYS.extend([ L4_TOI_KEY, L4_TOI_KEY+"Params" ])

L4_LINEFIT_KEY = "L4_iLineFit"
L4_HDF5_KEYS.extend([ L4_LINEFIT_KEY, L4_LINEFIT_KEY+"Params" ])

L4_QRBOX_KEY = "L4_QR_Box"
L4_HDF5_KEYS.append( L4_QRBOX_KEY )

L4_VICH_NCH_KEY = "L4_VICH_nch"
L4_VICH_NPULSES_KEY = "L4_VICH_npulses"
L4_VICH_QTOT_KEY = "L4_VICH_qtot"
L4_HDF5_KEYS.extend([ L4_VICH_NCH_KEY, L4_VICH_NPULSES_KEY, L4_VICH_QTOT_KEY ])

L4_SEP_IN_COGS_KEY = "L4_separation_in_cogs"
L4_ACC_TIME_KEY = "L4_accumulated_time"
L4_HDF5_KEYS.extend([ L4_SEP_IN_COGS_KEY, L4_ACC_TIME_KEY ])

# Noise rejection
L4_MICROCOUNT_KEY = "L4_micro_count"
L4_FILL_RATIO_KEY = "L4_fill_ratio"
L4_HDF5_KEYS.extend([ L4_MICROCOUNT_KEY, L4_FILL_RATIO_KEY ])

# Cuts/classifiers
L4_NOISE_STRAIGHT_CUT_KEY = "L4_NoiseStraightCuts_Bool"
L4_NOISE_MODEL_PREDICTION_KEY = "L4_NoiseClassifier_ProbNu"
L4_MUON_MODEL_PREDICTION_DATA_KEY = "L4_MuonClassifier_Data_ProbNu"
L4_MUON_MODEL_PREDICTION_MUONGUN_KEY = "L4_MuonClassifier_MuonGun_ProbNu"
L4_HDF5_KEYS.extend([ L4_NOISE_STRAIGHT_CUT_KEY, L4_NOISE_MODEL_PREDICTION_KEY, L4_MUON_MODEL_PREDICTION_DATA_KEY, L4_MUON_MODEL_PREDICTION_MUONGUN_KEY ])

# Legacy
L4_GRECO_BDT_KEY = "GRECO_L4_BDT"
L5_GRECO_BDT_KEY = "GRECO_L5_BDT"
L4_HDF5_KEYS.extend([ L4_GRECO_BDT_KEY, L5_GRECO_BDT_KEY ])


#TODO migrate

# #
# # Tray segments
# #

# @icetray.traysegment
# def oscNext_L4_common_variables( 
#     tray,
#     name, 
#     cleaned_pulses,
# ) :
#     '''
#     Compute some common variables required by both the noise and muon rejection segments
#     '''

#     #
#     # First HLC
#     #

#     # Get the position of the first HLC hit
#     # Note that this is also part of the Dunkman variables, but also doing this manually here in anticipation of ditching the Dunkman stuff 
#     # Compute the rho value as well as the cartesian coords

#     tray.AddModule(
#         "FirstHLC<I3RecoPulse>",
#         "FirstHLC",
#         HitSeriesName=cleaned_pulses,
#         OutputName=L4_FIRST_HLC_KEY,
#     )

#     # Also get the rho value
#     def add_rho_36_to_frame(frame,ParticleName,OutputName) :
#         if ParticleName in frame :
#             if OutputName not in frame :
#                 pos = frame[ParticleName].pos
#                 rho = calc_rho_36(x=pos.x, y=pos.y)
#                 frame[OutputName] = dataclasses.I3Double(rho)

#     tray.AddModule(
#         add_rho_36_to_frame,
#         "FirstHLCRho",
#         ParticleName=L4_FIRST_HLC_KEY,
#         OutputName=L4_FIRST_HLC_RHO_KEY,
#     )

#     return



# @icetray.traysegment
# def oscNext_L4_atm_muon_classifier_variables( 
#     tray,
#     name, 
#     uncleaned_pulses,
#     cleaned_pulses,
# ) :
#     '''
#     Create all variables to be used by the L4 atmospheric rejection classifier
#     '''

#     #
#     # Tensor of inertia
#     #

#     # http://software.icecube.wisc.edu/documentation/projects/tensor-of-inertia/

#     tray.AddModule( 
#         "I3TensorOfInertia",  #"libtensor-of-inertia",
#         L4_TOI_KEY,
#         AmplitudeOption  =  1,
#         AmplitudeWeight  =  1,
#         InputReadout     =  cleaned_pulses,
#         InputSelection   =  "",
#         MinHits          =  3,
#         Name             =  L4_TOI_KEY,
#     )


#     #
#     # Line fit
#     #

#     # http://software.icecube.wisc.edu/documentation/projects/linefit/

#     tray.AddSegment(
#         linefit.simple, 
#         L4_LINEFIT_KEY,
#         inputResponse=cleaned_pulses,
#         fitName=L4_LINEFIT_KEY,
#     )


#     #
#     # QR box
#     #

#     #TODO Think this is degenerate with an L3 variable, check...

#     icetray.load('slc-veto', False)

#     tray.AddModule( 
#         "SmallQ_Box",
#         L4_QRBOX_KEY,
#         BoxName        =  L4_QRBOX_KEY,
#         RecoPulsesKey  =  cleaned_pulses
#     )


#     #
#     # Dunkman variables
#     #

#     # Running an old project from DRAGON to compute a bunch of analysis variables
#     #TODO eventually extract the code for the few variables we need and ditch this dependency

#     from icecube import analysis
#     from icecube.analysis import event_selection
#     #from icecube import tau_bdt

#     dunk_prefix = "L4_Dunkman_"
#     dunk_vars_key = "%s%s_Variables" % (dunk_prefix,cleaned_pulses)

#     # Compute Dunkman variables
#     dunk_pulses = cleaned_pulses
#     tray.AddModule(
#         "CalculateVariables", 
#         "DunkVars",
#         PulseSeries=dunk_pulses,
#         OutputPrefix=dunk_prefix,
#     )

#     # Extract the variables of interest from the compound structure
#     def ExtractDunk(frame):
#         if frame.Has(dunk_vars_key) :
#             frame[L4_SEP_IN_COGS_KEY] = dataclasses.I3Double(frame[dunk_vars_key].separation)
#             frame[L4_ACC_TIME_KEY] = dataclasses.I3Double(frame[dunk_vars_key].accumulated_time)
#     tray.AddModule( ExtractDunk, "L4_ExtractDunk")


#     #
#     # Veto Identified Causal Hits
#     #

#     from icecube.tau_bdt import I3CutL7Module

#     tray.AddModule( 
#         I3CutL7Module, 
#         "L4_VICH", 
#         InputPulses=uncleaned_pulses, # Use uncleaned hits
#         OutputNChannelName=L4_VICH_NCH_KEY,
#         OutputNPulsesName=L4_VICH_NPULSES_KEY,
#         OutputChargeName=L4_VICH_QTOT_KEY,
#     )


# @icetray.traysegment
# def oscNext_L4_noise_cut_variables(
#     tray,
#     name, 
#     fill_ratio_vertex,
#     uncleaned_pulses,
#     cleaned_pulses,
# ):
#     '''
#     Create all variables to be used by the L4 pure noise event rejection classifier
#     '''

#     #TODO Martin has some additional final level noise cuts that might be useful, see here: https://drive.google.com/file/d/0B3rKx6zbxaQxQUlSMnR4N1ExMlU/view


#     #
#     # Microcount
#     #

#     # This microcount cleaning is lifted wholesale from GREDCO without any re-optimisation.
#     # It performs the microcount cut using a pulse series cleaned with different parameters 
#     # c.f. the microcount cut made at L3.
#     # I don't know the history as to why these settings were chosen, but we see that using 
#     # this microcount variable is complimentary to the L3 one when making cuts / training 
#     # classifiers, so we keep it here.

#     # Load the static time window cleaning (without this get a factory error message)
#     icetray.load("libstatic-twc")

#     # Static time window cleaning
#     tw_pulses = "L4_TWPulses"
#     stw_minus = 3500.
#     stw_plus = 4000.

#     tray.AddModule(
#         'I3StaticTWC<I3RecoPulseSeries>', 'L4_StaticTWC_DC',
#         InputResponse    = uncleaned_pulses,
#         OutputResponse   = tw_pulses,
#         TriggerConfigIDs = [1010, 1011],
#         TriggerName      = "I3TriggerHierarchy",
#         WindowMinus      = stw_minus,
#         WindowPlus       = stw_plus,               
#     )

#     tray.Add(check_object_exists,object_key=tw_pulses)

#     # Seeded RT cleaning
#     from icecube import STTools
#     from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService

#     tauL4_seededRTConfigService_nodust = I3DOMLinkSeededRTConfigurationService( 
#         useDustlayerCorrection  = False,
#         dustlayerUpperZBoundary = 0*I3Units.m,
#         dustlayerLowerZBoundary = -150*I3Units.m,
#         ic_ic_RTTime            = 1000*I3Units.ns, #TODO (Tom) there is also a DC version, which should I use?
#         ic_ic_RTRadius          = 150*I3Units.m #TODO (Tom) there is also a DC version, which should I use?
#     )

#     srt_tw_pulses = "L4_SRTTWPulses" #TODO Is this actually used?

#     tray.AddModule(
#         "I3SeededRTCleaning_RecoPulse_Module", "L4_SeededRTCleaning_DC_STTools",
#         AllowNoSeedHits = False,
#         InputHitSeriesMapName = tw_pulses,     # Name of input pulse series
#         OutputHitSeriesMapName = srt_tw_pulses, # Name of output pulse series
#         STConfigService = tauL4_seededRTConfigService_nodust,
#         MaxNIterations = -1,
#         SeedProcedure = "AllHLCHits",
#     )

#     tray.Add(check_object_exists,object_key=srt_tw_pulses)

#     # The goal here is to reduce the noise events by looking
#     # for events which have tightly time correlated hits in the
#     # DC Fiducial volume. Use the old-school DC fiducial
#     # IC86 definition.

#     DOMList = DOMS.DOMS( "IC86")

#     tw_fid_pulses = tw_pulses + "_DCFid"

#     tray.AddModule( 
#         "I3OMSelection<I3RecoPulseSeries>", "DCCFidPulses",
#         selectInverse  = True,
#         InputResponse  = tw_pulses,
#         OutputResponse = tw_fid_pulses,
#         OmittedKeys    = DOMList.DeepCoreFiducialDOMs
#     )

#     # Runs an X microsecond dynamic time window cleaning over the
#     # StaticTW cleaned recopulses in the traditional DeepCore
#     # fiducial region.

#     # Define the dynamic time windows to test
#     # GRECO settled on on 200 so for now only using this, but may want to re-optimise
#     dtw = 200 #175,200,250,300,400,600,800

#     dtw_tw_fid_pulses = tw_fid_pulses + ("_DTW%i" % dtw)

#     tray.AddModule( "I3TimeWindowCleaning<I3RecoPulse>", "DynamicTW", 
#                     InputResponse  = tw_fid_pulses,
#                     OutputResponse = dtw_tw_fid_pulses, 
#                     TimeWindow     = dtw,
#                     )

#     def MicroCount(frame):
#         MicroValues = dataclasses.I3MapStringInt()
#         if frame.Has(dtw_tw_fid_pulses):
#             reco_pulse_series = frame[dtw_tw_fid_pulses].apply(frame)
#             MicroValues["STW_m%ip%i_DTW%i"%(stw_minus,stw_plus,dtw)] = len(reco_pulse_series.values())
#         frame[L4_MICROCOUNT_KEY] = MicroValues
#         return

#     # Add the MicroCount MapStringInt object to the frame
#     tray.AddModule( MicroCount, "L4_MicroCount")


#     #
#     # Fill ratio
#     #

#     from icecube import fill_ratio

#     # http://software.icecube.wisc.edu/documentation/projects/fill-ratio/index.html

#     # Note that GRECO used a different pulse series here, but here I try using the standard 
#     # cleaned pulse series and I find it works well, so I stick with it for simplicity.

#     fill_ratio_spherical_radius_mean = 1.6 #TODO Was optimised for GRECO but has not been re-optimised for oscNext, it could be if someone has time

#     tray.AddModule(
#         "I3FillRatioModule",
#         "L4_FillRatio",
#         RecoPulseName = cleaned_pulses,
#         ResultName = L4_FILL_RATIO_KEY,
#         SphericalRadiusMean = fill_ratio_spherical_radius_mean,
#         VertexName = fill_ratio_vertex,
#     )

#     return



# @icetray.traysegment
# def GRECO_BDT(tray, name):
#     '''
#     Run the old GRECO BDTs for comparison, if interested
#     '''

#     icetray.load('tau-bdt', False)


#     #
#     # GRECO L4 BDT
#     #

#     # Hack to get the L3 NAbove200 variable
#     def GetL3Vars(frame) :
#         frame["tmp_L3_NAbove200"] = dataclasses.I3Double(frame[L3_KEY+"_Vars"]["NAbove200PE"])
#         frame["tmp_vertex_guess"] = dataclasses.I3Position(frame[L3_KEY+"_Vars"]["VertexGuessX"],frame[L3_KEY+"_Vars"]["VertexGuessY"],frame[L3_KEY+"_Vars"]["VertexGuessZ"])
#     tray.Add(GetL3Vars,"GetL3Vars")

#     # BDT code chokes without this
#     tray.Add(check_object_exists,object_key="L4_ToIParams")

#     def L4BDTChecks(frame):
#         if (frame.Has("L4_MicroCount") and frame["L4_MicroCount"].get("STW7500_DTW200") > 3):
#             return True
#         # end if()
#         return False
#     # end def()

#     # Run the old GRECO L4 BDT
#     tray.AddModule(
#         "TauBDTL4", #tau-bdt
#         L4_GRECO_BDT_KEY,
#         BDTOutput=L4_GRECO_BDT_KEY,
#         iLinefit=L4_LINEFIT_KEY,
#         NAbove200="tmp_L3_NAbove200",
#         ToI_TauL4Params=L4_TOI_KEY+"Params",
#         QR_Box=L4_QRBOX_KEY,
#         VertexGuess=tmp_vertex_guess, #TODO This is at L3 now
# #           If=L4BDTChecks #TODO REeplace this?
#     )


#     #
#     # GRECO L5 BDT
#     #

#     def L5Cleaning(frame):
#         if not frame.Has("DUNK_SRTTWOfflinePulsesDC_Extract"):
#             return False
#         # end if()
        
#         if not frame.Has("SPEFit11"):
#             return False
#         elif frame["SPEFit11"].fit_status == 0:
#             return True
#         else:
#             return False

#     # Run the old GRECO L5 BDT
#     tray.AddModule( "TauBDTL5", "GRECO_L5_BDT",
#                     BDTOutput   = "GRECO_L5_BDT",
#                     VariableMap = "DUNK_SRTTWOfflinePulsesDC_Extract",
#                     VICH        = "L7VetoHitsTotalPE",
#                     Zenith      = "SPEFit11",
# #                        If          = L5Cleaning,
#                     )

#     return



# @icetray.traysegment
# def compute_L4_cut( tray, name, ):

#     from icecube.oscNext.tools.classifier import I3Classifier

#     #
#     # Compute noise cuts
#     #

#     # Noise straight cuts
#     # This is a loose cut
#     # This is not used right now, but keeping for now anyway in case we find issues with the noise classifier in the future

#     def L4_noise_straight_cuts(frame,output_key) :
#         keep = False
#         if  ( frame["SRTTWOfflinePulsesDCHitMultiplicity"].n_hit_doms >= 8 ) and \
#             ( frame["IC2018_LE_L3_Vars"]["STW9000_DTW300Hits"] >= 2 ) and \
#             ( frame["L4_micro_count"]["STW_m3500p4000_DTW200"] >= 2 ) and \
#             ( frame["L4_fill_ratio"].fill_ratio_from_mean >= 0.03 ) and \
#             ( frame["SRTTWOfflinePulsesDCHitStatistics"].z_sigma >= 8. ) and \
#             ( frame["SRTTWOfflinePulsesDCHitStatistics"].z_travel >= -50. ) :
#             keep = True
#         frame[output_key] = icetray.I3Bool(keep)
#     tray.Add(L4_noise_straight_cuts,"L4_noise_straight_cuts",output_key=L4_NOISE_STRAIGHT_CUT_KEY)

#     # Add a noise classifier to the tray
#     noise_model = os.path.expandvars(os.path.join(CLASSIFIER_MODEL_DIR,"L4_noise_model.joblib"))
#     noise_classifier = I3Classifier(model_file=noise_model, class_key="neutrino", output_key=L4_NOISE_MODEL_PREDICTION_KEY)
#     tray.Add(noise_classifier,"oscNext_L4_noise_classifier")


#     #
#     # Compute atmospheric muon cuts
#     #

#     # Add a data-trained muon classifier to the tray
#     muon_model_data = os.path.expandvars(os.path.join(CLASSIFIER_MODEL_DIR,"L4_muon_model_data.joblib"))
#     muon_classifier_data = I3Classifier(model_file=muon_model_data, class_key="neutrino", output_key=L4_MUON_MODEL_PREDICTION_DATA_KEY)
#     tray.Add(muon_classifier_data,"oscNext_L4_muon_classifier_data")
      
#     # Add a MuonGun-trained muon classifier to the tray
#     muon_model_muongun = os.path.expandvars(os.path.join(CLASSIFIER_MODEL_DIR,"L4_muon_model_muongun.joblib"))
#     muon_classifier_muongun = I3Classifier(model_file=muon_model_muongun, class_key="neutrino", output_key=L4_MUON_MODEL_PREDICTION_MUONGUN_KEY)
#     tray.Add(muon_classifier_muongun,"oscNext_L4_muon_classifier_muongun")

#     # Run GRECO BDTs for comparison
#     # tray.Add( GRECO_BDT, "GRECO_BDT" )


#     #
#     # Make the overall cut
#     #

#     # Combine the noise and muon classifiers
#     def overall_cut(frame) :
#         passed_noise_cut = frame["L4_NoiseClassifier_ProbNu"].value >= 0.7
#         passed_muon_cut = frame["L4_MuonClassifier_Data_ProbNu"].value >= 0.65 # Using the data-driven classifier
#         frame[L4_CUT_BOOL_KEY] = icetray.I3Bool( passed_noise_cut & passed_muon_cut )
#     tray.Add(overall_cut,"L4_overall_cut")

#     return



# @icetray.traysegment
# def oscNext_L4( tray, name, 
#                 uncleaned_pulses,
#                 cleaned_pulses,
#                 ):
#     '''
#     This is the main oscNext L4 tray segment
#     '''

#     raise Exception("L4 not yet migrated to GitHub IceTray")

#     #
#     # Apply L3 cut(s)
#     #

#     # Only keep frames passing L3
#     tray.Add(oscNext_cut,"L3_cut",processing_level=3)


#     #
#     # Calculate variables
#     #

#     # Common
#     tray.Add(
#         oscNext_L4_common_variables,
#         "oscNext_L4_common_variables", 
#         cleaned_pulses=cleaned_pulses,
#     )

#     # Noise
#     tray.Add(
#         oscNext_L4_noise_cut_variables, 
#         "oscNext_L4_noise_cut_variables", 
#         fill_ratio_vertex=L4_FIRST_HLC_KEY,
#         uncleaned_pulses=uncleaned_pulses,
#         cleaned_pulses=cleaned_pulses,
#     )

#     # Muons
#     tray.Add(
#         oscNext_L4_atm_muon_classifier_variables, 
#         "oscNext_L4_atm_muon_classifier_variables", 
#         uncleaned_pulses=uncleaned_pulses,
#         cleaned_pulses=cleaned_pulses,
#     )


#     #
#     # L4 cut
#     #

#     # Compute cut
#     tray.Add(
#         compute_L4_cut, 
#         "compute_L4_cut", 
#     )


#     #
#     # Done
#     #

#     # Can dump frame for debugging
#     if False :
#         tray.Add("Dump","Dump")

#     return
