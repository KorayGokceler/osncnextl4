#!/usr/bin/env python

from argparse import ArgumentParser
from os.path import expandvars
import os, sys, random
from globals import ice_model
from icecube.oscNext.tools.file_transfer import gridftp_fileio
from icecube.oscNext.frame_objects.pulses import record_mcpulse_truth_flags
from I3Tray import *

#
# Get args
#

usage = "usage: %prog [options]"
parser = ArgumentParser(usage)
parser.add_argument("-o", "--outfile",default="./test_output.i3",
                    dest="OUTFILE", help="Write output to OUTFILE (.i3{.gz} format)")
parser.add_argument("-i", "--infile",default="./test_input.i3",
                    dest="INFILE", help="Read input from INFILE (.i3{.gz} format)")
parser.add_argument("-r", "--runnumber", type=str, default="1",
                    dest="RUNNUMBER", help="The run number for this simulation, is used as seed for random generator")
parser.add_argument("-s", "--seed", type=str, default=None,
                    dest="SEED", help="The seed for this simulation for random generator, if different than runnumber")
parser.add_argument("-f", "--filenr",type=str,default="1",
                    dest="FILENR", help="File number, stream of I3SPRNGRandomService")
parser.add_argument("-g", "--gcdfile", default=os.getenv('GCDfile'),
            dest="GCDFILE", help="Read in GCD file")
parser.add_argument("-e","--efficiency", type=float,default=1.2,
                    dest="EFFICIENCY",help="DOM Efficiency ... the same as UnshadowedFraction")
parser.add_argument("-n","--noise", default="vuvuzela",
                    dest="NOISE",help="Noise model (vuvuzela/poisson)")
parser.add_argument("-l", "--holeice",  required=True, dest="HOLEICE", 
                    help="Pick the hole ice parameterization, corresponds to a file name path relative to $I3_SRC/ice-models/resources/models/") 
parser.add_argument("-m", "--icemodel", default = ice_model, 
                    dest = "ICEMODEL", type=str,
                    help="Should be the same ice model as used for photon propagation (step2)")
parser.add_argument("--gridftp", required=False, 
                    dest = "GRIDFTP", action="store_true", 
                    help="Indicate that grid FTP is being used for file I/O")
parser.add_argument("--tmpdir",default=None, #TODO better default, e.g. default to using this feature
                    dest="TMPDIR", help="Tmp directory for intermediate file writing")
parser.add_argument("--simple-hqe-scaling", dest = "SIMPLE_HQE_SCALING", action="store_true", 
                    help="Toggle use of old HQE QE curve, e.g. simply scale the NQE QE curves by 1.35 (this is an older, deprecated method, but retained as an option for backwards compatibility) ")
# parser.add_argument("-s","--scalehad", type="float", default=1.,
#                     dest="SCALEHAD",help="Scale light from hadrons") # This is currently not used
parser.add_argument("--vuvuzela-start-time", default = -11.*I3Units.microsecond, 
                    dest = "VUVUZELA_START_TIME", type=float, help="Start of time window used for vuvzela [ns]")
parser.add_argument("--vuvuzela-end-time", default = 11.*I3Units.microsecond, 
                    dest = "VUVUZELA_END_TIME", type=float, help="End of time window used for vuvzela [ns]")
args = parser.parse_args()

args.FILENR=int(args.FILENR)
args.RUNNUMBER=int(args.RUNNUMBER)

if args.SEED==None:
    args.SEED=args.RUNNUMBER
else:
    args.SEED=int(args.SEED)

print("GCD file    : %s" % args.GCDFILE)
print("Input file  : %s" % args.INFILE)
print("Output file : %s" % args.OUTFILE)


#
# IceTray
#

import random

from icecube import icetray, dataclasses, dataio, simclasses #, recclasses
from icecube import phys_services, sim_services, DOMLauncher, DomTools, clsim, trigger_sim #, genie_icetray

from RemoveLatePhotons_V5 import RemoveLatePhotons

@gridftp_fileio
def step_3_icetray(gcd_file, input_file, output_file) :
    '''
    The step3 icetray job
    Using a decorator to add gridftp file I/O support
    '''

    print("Local GCD file path    : %s" % gcd_file)
    print("Local input file path  : %s" % input_file)
    print("Local output file path : %s" % output_file)

    def BasicHitFilter(frame):
        hits = 0
        if frame.Has("MCPESeriesMap"):
           hits = len(frame.Get("MCPESeriesMap"))
        if hits>0:
    #        print("has photons")
            return True
        else:
    #       print("does NOT photons")
            return False

    def BasicDOMFilter(frame):
        if frame.Has("InIceRawData"):
            if len(frame['InIceRawData']) > 0:
                return True
            else:
                return False
        else:
           return False


    tray = I3Tray()

    print('Using RUNNR: ', args.RUNNUMBER)
    print('Using SEED: ', args.SEED)
    print("DOM efficiency: ", args.EFFICIENCY)
    print("Using hole ice: ", args.HOLEICE )
    print("Looking for ice model in ", expandvars("$I3_SRC/ice-models/resources/models/"))
    if args.ICEMODEL=="" or args.ICEMODEL==None:
        print("\033[93mNo ice model provided. The baseline efficiency can be found in cfg.txt")
        print("of the ice model used for photon propagation. \033[0m")
        print("\033[93m\033[1mBe very careful! \033[0m")
        icemodel_path=None
    else:
        # icemodel_path=expandvars("$I3_SRC/ice-models/resources/models/%s"%args.ICEMODEL)
        icemodel_path=expandvars("$I3_SRC/ice-models/resources/models/ICEMODEL/"+args.ICEMODEL) # Update path for modern code
        if os.path.isdir(icemodel_path) :
            print("Folder with ice model found: ", icemodel_path)
        else: 
            print("Error! No ice model with such name found in :" )
            print(expandvars("$I3_SRC/ice-models/resources/models/"))
            exit() 
    print("Ice model path: ", icemodel_path)
    # Random service
    from globals import max_num_files_per_dataset
    tray.AddService("I3SPRNGRandomServiceFactory","sprngrandom")(
        ("Seed",args.SEED),
        ("StreamNum",args.FILENR),
        ("NStreams", max_num_files_per_dataset),
        ("instatefile",""),
        ("outstatefile",""),
    )

    # Now fire up the random number generator with that seed
    randomService = phys_services.I3SPRNGRandomService(
        seed = args.SEED,
        nstreams = max_num_files_per_dataset,
        streamnum = args.FILENR)


    ### START ###

    tray.Add(dataio.I3Reader, FilenameList = [gcd_file, input_file])


    ####
    ## Remove photons from neutron decay and other processes that take too long (unimportant)
    ####

    tray.AddModule(RemoveLatePhotons, "RemovePhotons",
                   InputPhotonSeries = "I3Photons",
                   TimeLimit = 1E5) #nanoseconds

    ####
    ## Make hits from photons (change efficiency here already!)
    ####

    tray.AddModule("I3GeometryDecomposer", "I3ModuleGeoMap")
    # from ReduceHadronicLightyield import HadLightyield

    # print("Scaling hadrons with: ", options.SCALEHAD)
    # tray.AddModule(HadLightyield , "scalecascade", 
    #               Lightyield = options.SCALEHAD)

    gcd_file_path = gcd_file
    gcd_file = dataio.I3File(gcd_file_path)

    tray.AddSegment(clsim.I3CLSimMakeHitsFromPhotons, "makeHitsFromPhotons",
    #                MCTreeName="I3MCTree_clsim",
    #                PhotonSeriesName="UnweightedPhotons2",
                    PhotonSeriesName="I3Photons",
                    MCPESeriesName="MCPESeriesMap",
                    RandomService=randomService,
                    DOMOversizeFactor=1.,
                    DOMEfficiency=args.EFFICIENCY, #`UnshadowedFraction` -> `DOMEfficiency` (see Feb. 21, 2022 release notes, https://github.com/icecube/icetray/blob/main/clsim/RELEASE_NOTES)
                    IceModelLocation = icemodel_path,
    #               UseHoleIceParameterization=holeice
                    HoleIceParameterization=expandvars("$I3_SRC/ice-models/resources/models/ANGSENS/%s"%args.HOLEICE),
                    GCDFile=gcd_file_path,
                    SimpleHQEScaling=args.SIMPLE_HQE_SCALING,
                    )



    #from icecube.BadDomList import bad_dom_list_static
    txtfile = os.path.expandvars('$I3_SRC') + '/BadDomList/resources/scripts/bad_data_producing_doms_list.txt'
    #BadDoms = bad_dom_list_static.IC86_bad_data_producing_dom_list(118175, txtfile)
    tray.AddModule(BasicHitFilter, 'FilterNullMCPE', Streams = [icetray.I3Frame.DAQ, icetray.I3Frame.Physics])
    #print(BadDoms)
    mcpe_to_pmt = "MCPESeriesMap"
    if args.NOISE == 'poisson':
        print("Error! Poisson noise is not supported anymore. Exiting")
        exit 
    elif args.NOISE == 'vuvuzela':
        from icecube import vuvuzela
      # Have removed the vuvuzela traysegment for now, using the module instead below so that the noise parameters are properly taken into account
    #   tray.AddSegment(vuvuzela.AddNoise, 'VuvuzelaNoise',
    #                 InputName = mcpe_to_pmt,
    #                 OutputName = mcpe_to_pmt + "_withNoise",
    # #                ExcludeList = BadDoms,
    #                 StartTime = -11*I3Units.microsecond,
    #                 EndTime   = 11*I3Units.microsecond,
    #                 DisableLowDTCutoff = True
    #                 )

        print("Vuvuzela time window : [%s, %s] ns" % (args.VUVUZELA_START_TIME, args.VUVUZELA_END_TIME))

        mcpeout = mcpe_to_pmt + '_withNoise'
        tray.AddModule("Vuvuzela", "vuvuzela_noise" ,
            InputHitSeriesMapName  = mcpe_to_pmt,
            OutputHitSeriesMapName = mcpeout,
          StartWindow            = args.VUVUZELA_START_TIME,
          EndWindow              = args.VUVUZELA_END_TIME,
          # IceTop                 = False, # arg no longer exists
          # InIce                  = True, # arg no longer exists
          ScaleFactor            = 1.0,
          DeepCoreScaleFactor    = 1,
          DOMsToExclude          = [], # This will be cleaned later by DOM launch cleaner
          RandomService          = "I3RandomService",
          SimulateNewDOMs        = True,
          DisableLowDTCutoff     = True,
          UseIndividual          = True
        )

    elif args.NOISE == 'none':
            print('\n*******ERROR: Noiseless simulation!!********\n')
            exit()
            mcpeout = mcpe_to_pmt

    else:
        print('Pick a valid noise model!')
        exit()


    # PMT simulation
    mcpulse_out = mcpeout + "_weighted"
    tray.AddModule("PMTResponseSimulator","rosencrantz",
        Input=mcpeout,  
        Output=mcpulse_out,
        MergeHits=True,
        )

    # DOM readout simulation
    tray.AddModule("DOMLauncher", "guildenstern",
        Input= mcpulse_out,
        Output="InIceRawData_unclean",
        UseTabulatedPT=True,
        )

    tray.AddModule("I3DOMLaunchCleaning","launchcleaning")(
           ("InIceInput","InIceRawData_unclean"),
           ("InIceOutput","InIceRawData"),
           ("FirstLaunchCleaning",False),
    #       ("CleanedKeys",BadDoms)
           )

    # Pulse truth flags
    tray.AddModule(
        record_mcpulse_truth_flags, 
        "record_mcpulse_truth_flags",
        signal_mcpe_map_key=mcpe_to_pmt, 
        signal_and_noise_mcpe_map_key=mcpeout, 
        mcpulse_map_key=mcpulse_out, 
        Streams=[icetray.I3Frame.DAQ],
    )

    # Dropping frames without InIceRawData
    tray.AddModule(BasicDOMFilter, 'FilterNullInIce', Streams = [icetray.I3Frame.DAQ, icetray.I3Frame.Physics])
    ###### triggering 
    tray.AddModule('Delete', 'delete_triggerHierarchy',
                   Keys = ['I3TriggerHierarchy', 'TimeShift', 'CleanIceTopRawData'])

    #time_shift_args = { #'I3MCTreeNames': [],
    #                    'I3MCPMTResponseMapNames': [],
    #                    'I3MCHitSeriesMapNames' : [] }

    tray.AddSegment(trigger_sim.TriggerSim, 'trig', 
                    gcd_file = gcd_file,
                    time_shift_args = {"SkipKeys" : ["BundleGen"]}, # 
     #               time_shift_args = time_shift_args,
    #added in run_id
                    run_id=1)

    # Not skipping these keys for now (check what gets dropped in the L2)
    skipkeys = ["MCPMTResponseMap",
                "MCTimeIncEventID",
                "I3MCTree_clsim"
                "I3Photons"
                "clsim_stats",
                "InIceRawData_unclean",
                ]

    tray.AddModule("I3Writer","writer",
                   #SkipKeys=skipkeys, All of these get thrown out by the L2 anyways ... keep them? 
                   Filename = output_file,
                   Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics, icetray.I3Frame.TrayInfo, icetray.I3Frame.Simulation],
                  )

    tray.AddModule("TrashCan","adios")

    tray.Execute()
    tray.Finish()


#
# Run
#

step_3_icetray(
    gridftp=args.GRIDFTP,
    gcd_file=args.GCDFILE,
    input_file=args.INFILE,
    output_file=args.OUTFILE,
    tmp_dir=args.TMPDIR,
)
