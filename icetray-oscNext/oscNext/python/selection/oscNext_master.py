#!/usr/bin/env python

'''
This is top-level code managing all processing levels of oscNext
A script for running the selection is here: $I3_SRC/oscNext/resources/scripts/run_oscNext.py

Tom Stuttard
'''

#TODO In general, replace the old, slow python implementation of variables with the new C++ versions from Michael's LowEnVariables project, where available : http://code.icecube.wisc.edu/svn/sandbox/LowEnVariables/trunk/private/LowEnVariables/LowEnAlgorithms.cxx

import collections, os, sys, copy, socket

from icecube import icetray
from icecube.oscNext.selection.globals import *
from icecube.oscNext.tools.data_quality import check_object_exists
from icecube.oscNext.tools.processor import i3_processor


#
# Get all processing level traysegments
#

# Choose oscNext variant
OSCNEXT_VARIANT = "GNN" # Choose from: RetroReco, GNN

# Container to store processing level tray segments
#   key = processing level
#   value = traysegmet (python function)
PROCESSING_LEVEL_TRAY_SEGMENTS = collections.OrderedDict()

# L3
# See https://wiki.icecube.wisc.edu/index.php/DeepCore_Level3_2012
# from icecube.level3_filter_lowen.LowEnergyL3TraySegment import DCL3MasterSegment, DeepCoreCuts # Deprecated (oscNext_meta)
# from icecube.oscNext.selection.LowEnergyL3TraySegment import DCL3MasterSegment, DeepCoreCuts      #TODO Needs migration to be finished
# # PROCESSING_LEVEL_TRAY_SEGMENTS[3] = DCL3MasterSegment
# PROCESSING_LEVEL_TRAY_SEGMENTS[3] = DeepCoreCuts    
# L3
from icecube.oscNext.selection.oscNext_L3 import oscNext_L3
PROCESSING_LEVEL_TRAY_SEGMENTS[3] = oscNext_L3

# L4
# from icecube.oscNext.selection.oscNext_L4 import oscNext_L4  #TODO Needs migrating
# PROCESSING_LEVEL_TRAY_SEGMENTS[4] = oscNext_L4

# L5
# from icecube.oscNext.selection.oscNext_L5 import oscNext_L5   #TODO Needs migrating
# PROCESSING_LEVEL_TRAY_SEGMENTS[5] = oscNext_L5

# L6 onwards depends on variant
if OSCNEXT_VARIANT == "RetroReco" :

    # L6
    from icecube.oscNext.selection.oscNext_L6 import oscNext_L6
    PROCESSING_LEVEL_TRAY_SEGMENTS[6] = oscNext_L6

    # L7
    from icecube.oscNext.selection.oscNext_L7 import oscNext_L7
    PROCESSING_LEVEL_TRAY_SEGMENTS[7] = oscNext_L7


elif OSCNEXT_VARIANT == "GNN" :

    # L6
    from icecube.oscNext.selection.oscNext_GNN_L6 import oscNext_GNN_L6
    PROCESSING_LEVEL_TRAY_SEGMENTS[6] = oscNext_GNN_L6

    # L7 
    from icecube.oscNext.selection.oscNext_GNN_L7 import oscNext_GNN_L7
    PROCESSING_LEVEL_TRAY_SEGMENTS[7] = oscNext_GNN_L7


#
# Get all HDF5 keys
#

#TODO This is messy, probably easier just to define an exclude list....

# Get some simulation key names
from icecube.oscNext.frame_objects.simulation import IN_ICE_PRIMARY_KEY, AIR_SHOWER_PRIMARY_KEY, EXTRA_TRUTH_INFO
from icecube.oscNext.frame_objects.neutrinos import STARTING_NEUTRINO_KEY
from icecube.oscNext.frame_objects.weighting import WEIGHT_DICT_KEY

# Start by defining things already in the frames and useful before the oscNext processing stages start
# THis means everything from L3 and below
HDF5_KEYS = [

    # Standard stuff
    "I3EventHeader",
    "I3TriggerHierarchy",
    "FilterMask",

    # Truth
    "I3MCTree",
    "I3MCTree_preMuonProp",
    "I3MCWeightDict",
    IN_ICE_PRIMARY_KEY,
    AIR_SHOWER_PRIMARY_KEY,
    STARTING_NEUTRINO_KEY,
    EXTRA_TRUTH_INFO,
    WEIGHT_DICT_KEY,

    # L2
    "SPEFit2",
    "SPEFit2FitParams",
    "SPEFit2_DC",
    "SPEFit2_DCFitParams",

]

# Try to add GENIE info
# This will only work if GENIE icetray is available
#TODO Removed for now since there is no available converter for the complicated struct
# try :
#     from icecube import genie_icetray
#     from icecube.oscNext.frame_objects.genie import GENIE_RESULT_DICT_KEY
#     HDF5_KEYS.append(GENIE_RESULT_DICT_KEY)
# except :
#     print("WARNING: genie_icetray not available, '%s' variables will be missing from the HDF5 files" % GENIE_RESULT_DICT_KEY)

# Add the various processing levels
from icecube.oscNext.selection.oscNext_L3 import L3_HDF5_KEYS
HDF5_KEYS.extend(L3_HDF5_KEYS)

from icecube.oscNext.selection.oscNext_L4 import L4_HDF5_KEYS
HDF5_KEYS.extend(L4_HDF5_KEYS)

from icecube.oscNext.selection.oscNext_L5 import L5_HDF5_KEYS
HDF5_KEYS.extend(L5_HDF5_KEYS)

if OSCNEXT_VARIANT == "RetroReco" :

    from icecube.oscNext.selection.oscNext_L6 import L6_HDF5_KEYS
    HDF5_KEYS.extend(L6_HDF5_KEYS)

    from icecube.oscNext.selection.oscNext_L6 import L6_HDF5_KEYS
    HDF5_KEYS.extend(L6_HDF5_KEYS)


elif OSCNEXT_VARIANT == "GNN" :

    from icecube.oscNext.selection.oscNext_GNN_L6 import L6_HDF5_KEYS
    HDF5_KEYS.extend(L6_HDF5_KEYS)

    from icecube.oscNext.selection.oscNext_GNN_L7 import L7_HDF5_KEYS
    HDF5_KEYS.extend(L7_HDF5_KEYS)

# Add the standard pulse series
for pulses in [ CLEANED_PULSES, UNCLEANED_PULSES ] :
    HDF5_KEYS.append( pulses )
    HDF5_KEYS.append( pulses + "_TruthFlags" )
    HDF5_KEYS.append( pulses + "HitMultiplicity" )
    HDF5_KEYS.append( pulses + "HitStatistics" )
    HDF5_KEYS.append( pulses + "TimeCharacteristics" )


#
# Run event selection
#

@i3_processor
def oscNext_processing( tray, data_type, processing_levels, uncleaned_pulses, cleaned_pulses, gcd_file, dataset=None, file_num=None, fix_sim_headers=False, write_event_id=False ) :
    '''
    Function to run the oscNext processing levels
    Uses the `i3_processor` decorator for boiler plate processing stuff 
    '''

    #
    # Cleaning
    #

    # Remove frames with no pulses, can't be used for anything
    #TODO Why are there frames with no pulses? Maybe just cleaned ones, if SRT/TW produces no hits maybe nothing written to frame?
    tray.Add( check_object_exists, "check_uncleaned_pulses", object_key=uncleaned_pulses )
    tray.Add( check_object_exists, "check_cleaned_pulses", object_key=cleaned_pulses )


    #
    # Event header handling
    #

    from icecube.oscNext.frame_objects.simulation import FixSimEventHeaders

    # Enforce simulation header formatting (if user requests it)
    # Note that this relies on the fact we are processing one file at a time (e.g. the code is not
    # yet smart enough to handle cases with merged files)
    # Also can directly write new (sequential) event IDs if requested

    # Check inputs
    if write_event_id :
        assert fix_sim_headers, "If enable `write_event_id`, must also enable `fix_sim_headers`"

    # Only do this if user requested, and never on real data
    if fix_sim_headers and (data_type != "data") :

        # Check user provided what we need
        assert isinstance(dataset,int), "Must provide dataset number when `fix_sim_headers` option is specified"
        assert isinstance(file_num,int), "Must provide file number when `fix_sim_headers` option is specified"

        # Add the module
        sim_header_fixer = FixSimEventHeaders( sub_event_stream=SUB_EVENT_STREAM, dataset_id=dataset, file_id=file_num, write_event_id=write_event_id, assert_unique=True, assert_ascending=True )
        tray.Add( sim_header_fixer, "oscNext_sim_header_fixer", streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics] )


    #
    # Run the processing
    #

    # Loop over processing levels
    for level in processing_levels :

        # Add the truth flags for reco pulses at L3 (in future, directly store these during MC production)
        if level == 3 :
            from icecube.oscNext.frame_objects.pulses import record_reco_pulse_truth_flags
            for reco_pulse_map_key in [ uncleaned_pulses, cleaned_pulses ] : 
                tray.Add(
                    record_reco_pulse_truth_flags, 
                    "record_reco_pulse_truth_flags_"+reco_pulse_map_key,
                    reco_pulse_map_key=reco_pulse_map_key,
                )

        # Define any custom args for this processing level traysegment
        custom_args = {}

        # Must provide (un)cleaned pulses series map names for oscNext segments
        custom_args["uncleaned_pulses"] = uncleaned_pulses
        custom_args["cleaned_pulses"] = cleaned_pulses

        # Level 3 needs to have the DeepCore filter year passed
        if level == 3 :
            custom_args["year"] = str(DC_FILTER_YEAR) #TODO For pass 1 2012 (or old CORSIKA), this needs to be 12 instead (but no longer use this anyway)

        # Reconstruction segments can need a few dedicated things
        if level == 6 :
            # if OSCNEXT_VARIANT == "RetroReco" :
            custom_args["gcd_file"] = gcd_file #TODO should instead get this from the frame

        # Add the tray segment
        tray.AddSegment( PROCESSING_LEVEL_TRAY_SEGMENTS[level], "oscNext_L%i"%level, **custom_args )


    #
    # Add some extra useful stuff to frame
    #

    # Store useful information about the simulation
    from icecube.oscNext.frame_objects.simulation import simulation_info
    tray.Add( simulation_info, "oscNext_sim_info", data_type=data_type )

    # Store the weighting info
    from icecube.oscNext.frame_objects.weighting import weighting
    tray.Add( weighting, "oscNext_weighting", data_type=data_type, dataset=dataset )

    # Store useful information about the pulse series
    # Doing this AFTER the selection in case it is already there (this module checks first and skips if already present)
    # Have removed this, L3 already does it
    # from icecube.oscNext.frame_objects.pulses import pulse_info
    # for pulses in [uncleaned_pulses,cleaned_pulses] :
    #     tray.Add( pulse_info, "oscNext_pulse_info_%s"%pulses, pulses=pulses )


def run_oscNext(
    input_file,
    gcd_file,
    processing_levels,
    data_type,
    output_file=None,
    hdf5_file=None,
    tmp_dir=None, # Always provide this unless really cannot (e.g. some awkward clusters)
    gridftp=False, # Use grid FTP to transfer files (otherwise just directly access filesystem)
    dataset=None,
    file_num=None,
    fix_sim_headers=False,
    write_event_id=False,
    metadata=None,
    # keep_failed_frames=False, #TODO
    hash_type="sha256",
    uncleaned_pulses=None,
    cleaned_pulses=None,
) :
    '''
    This is the main, top-level function to run the oscNext processing chain.
    It runs the event selection itself (via `oscNext_processing`) and also takes care of things like metadata, file provenance, etc
    Note that an executable script running thois can be found here: $I3_SRC/oscNext/resources/scripts/run_oscNext.py
    '''

    #TODO document args

    #TODO Move some of this functionality to i3processor

    import sys, os, datetime, json
    import numpy as np


    # 
    # Check inputs
    #

    # Check input and GCD file exist
    #TODO

    # Check data types
    assert data_type in DATA_TYPES, "Unknown data type '%s' (choose from %s)" % (data_type,DATA_TYPES)

    # Check processing levels
    for processing_level in processing_levels : 
        assert processing_level in PROCESSING_LEVEL_TRAY_SEGMENTS.keys(), "Invalid processing level : %i (choose from %s)" % (processing_level,PROCESSING_LEVEL_TRAY_SEGMENTS.keys())
    assert np.all(np.diff(processing_levels) == 1), "Processing levels must be sequential : %s" % processing_levels

    # User can request an output i3 file, HDF5 file, or both, but not neither
    assert (output_file is not None) or (hdf5_file is not None), "Must provide `output_file`, `hdf5_file` or both"

    # Check where we are
    hostname = socket.gethostname()

    # Init a HDF5 key list
    hdf5_keys = copy.copy(HDF5_KEYS)

    # Report
    print("")
    print("Running oscNext processing with settings :")
    print("  I3_BUILD : %s" % (os.environ["I3_BUILD"]) )
    print("  Data type : %s" % (data_type) )
    print("  Processing level(s) : %s" % ( "+".join([ "L%i"%l for l in processing_levels ])) )
    print("  Temporary dir : %s" % (tmp_dir) )
    print("  GridFTP : %s" % (gridftp) )
    print("")


    #
    # Pulse series
    #

    # Set some default pulse series if none provided
    if uncleaned_pulses is None :
        uncleaned_pulses = UNCLEANED_PULSES
    if cleaned_pulses is None :
        cleaned_pulses = CLEANED_PULSES

    # Make sure they are addded to the HDF5 keys, including derived stats, etc
    for pulses in [cleaned_pulses,uncleaned_pulses] :
        hdf5_keys.append( pulses )
        hdf5_keys.append( pulses + "HitMultiplicity" )
        hdf5_keys.append( pulses + "HitStatistics" )
        hdf5_keys.append( pulses + "TimeCharacteristics" )


    #
    # Metadata
    #

    # Create metadata container if none provided by the user
    if metadata is None :
        metadata = collections.OrderedDict()

    # Add some default metadata
    metadata["num_merged_files"] = 1 # Not merging files here...
    metadata["uncleaned_pulses"] = uncleaned_pulses
    metadata["cleaned_pulses"] = cleaned_pulses


    #
    # Run the processing
    #

    # Run the actual oscNext event selection
    oscNext_processing(
        gcd_file=gcd_file,
        i3_files=[input_file],
        output_file=output_file,
        hdf5_file=hdf5_file,
        hdf5_keys=hdf5_keys,
        tmp_dir=tmp_dir,
        gridftp=gridftp,
        hash_type=hash_type,
        metadata=metadata,
        sub_event_stream=SUB_EVENT_STREAM,
        processing_levels=processing_levels,
        uncleaned_pulses=uncleaned_pulses,
        cleaned_pulses=cleaned_pulses,
        data_type=data_type,
        dataset=dataset,
        file_num=file_num,
        fix_sim_headers=fix_sim_headers,
        write_event_id=write_event_id,
    )

    # Done
    return



def run_oscNext_command_line() :
    '''
    Interface to `run_oscNext` using command line args.
    This an be used by scripts.
    '''

    import sys, os, argparse
    import numpy as np
    from icecube.oscNext.tools.metadata import DictAction


    # 
    # Parse inputs
    #

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--datatype", type=str, required=True, choices=DATA_TYPES, help="The data type to be processed (used for weighting)")
    parser.add_argument("-i", "--inputfile", type=str, required=True, help="Input file path")
    parser.add_argument("-g", "--gcdfile", type=str, required=True, help="GCD file path")
    parser.add_argument("-o", "--outputfile", type=str, required=False, default=None, help="Output file path")
    parser.add_argument("--hdf5file",type=str, required=False, default=None, help="HDF5 file path (will contain the same data as the output file, but in HDF5 file format") # Not using "-h", this is "help"
    parser.add_argument("-l", "--levels", type=int, nargs='+', required=True, help="Processing level to run (can specify multiple, must be sequential)")
    parser.add_argument("-d", "--dataset", type=int, required=False, default=None, help="Dataset number (used for weighting, only required with corsika)")
    parser.add_argument("--file-num",type=int, required=False, default=None, help="File number within dataset. Used for creating unique event headers")
    parser.add_argument("-m", "--metadata", action=DictAction, nargs="+", required=False, default=None, help="Optionally provide a dict of metadata. Format : -m key1:val1 key2:val2")
    parser.add_argument("--tmp-dir", type=str, required=False, default=None, help="RECOMMENDED. Output/HDF5 files will be written to a tmp directory during processing, and only moved to their final locations once everything has completed successfully.")
    parser.add_argument("--gridftp", action="store_true", required=False, help="Use `gridftp` for file transfer (useful for clusters without access to the IceCube datastore")
    parser.add_argument("--fix-sim-headers", action="store_true", required=False, help="Fix and makes checks on simulation event headers (ensures uniqueness and easy mapping)")
    parser.add_argument("--write-event-id", action="store_true", required=False, help="Directly write new, incrementing event IDs")
    args = parser.parse_args()


    #
    # Run the processing function
    #

    # Report on which versions are being used

    # Run the main function
    run_oscNext(
        data_type=args.datatype,
        input_file=args.inputfile,
        gcd_file=args.gcdfile,
        output_file=args.outputfile,
        hdf5_file=args.hdf5file,
        processing_levels=args.levels,
        metadata=args.metadata,
        dataset=args.dataset,
        file_num=args.file_num,
        tmp_dir=args.tmp_dir,
        gridftp=args.gridftp,
        # keep_failed_frames=False,
        fix_sim_headers=args.fix_sim_headers,
        write_event_id=args.write_event_id,
    )
