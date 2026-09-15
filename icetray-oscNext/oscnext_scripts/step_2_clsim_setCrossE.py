#!/usr/bin/env python

from optparse import OptionParser
from os.path import expandvars
import os, sys, random, copy
from globals import ice_model
from icecube.oscNext.tools.file_transfer import gridftp_fileio
from icecube.icetray import logging
from icecube import dataclasses
import numpy as np


# This script will perform a hybridCLSim propagation.
#
# NOTE: There is no bad_dom_cleaning!!!
#       This you still have to do after the propagation!!!


#
# Get args
#

usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("-o", "--outfile",default="./test_output.i3",
                  dest="OUTFILE", help="Write output to OUTFILE (.i3{.gz} format)")
parser.add_option("-i", "--infile",default="./test_input.i3",
                  dest="INFILE", help="Read input from INFILE (.i3{.gz} format)")
parser.add_option("-r", "--runnumber", type="string", default="1",
                  dest="RUNNUMBER", help="The run/dataset number for this simulation, is used as seed for random generator")
parser.add_option("-l", "--filenr",type="string",default="1",
                   dest="FILENR", help="File number, stream of I3SPRNGRandomService")
parser.add_option("-g", "--gcdfile", default=os.getenv('GCDfile'),
      dest="GCDFILE", help="Read in GCD file")
parser.add_option("-e","--efficiency", type="float",default=1.2, # Using efficiency > 1 as default so we can support systematics sets
                  dest="EFFICIENCY",help="DOM Efficiency ... the same as UnshadowedFraction")
parser.add_option("-m","--icemodel", default=ice_model,
                  dest="ICEMODEL",help="Ice model (spice_3.2.1, spice_mie, spice_lea, etc)")
parser.add_option("-a", "--holeice",  default="angsens/as.flasher_p1_0.30_p2_-1", dest="HOLEICE", 
                  help="Pick the hole ice parameterization, corresponds to a file name path relative to $I3_SRC/ice-models/resources/models/") 
parser.add_option("-c","--crossenergy", type="float",default=30.0, # Keeping arg as --crossenergy (not --hadr-crossenergy) for backwards compatibility
                  dest="HADR_CROSSENERGY",help="The cross energy where the hybrid clsim approach will be used for hadrons")
parser.add_option("--em-crossenergy", type="float",default=0.1,
                  dest="EM_CROSSENERGY",help="The cross energy where the hybrid clsim approach will be used for EM")
parser.add_option("--simple-hqe-scaling", dest = "SIMPLE_HQE_SCALING", action="store_true", 
                    help="Toggle use of old HQE QE curve, e.g. simply scale the NQE QE curves by 1.35 (this is an older, deprecated method, but retained as an option for backwards compatibility) ")
parser.add_option("--geant4-muons", dest = "USE_GEANT4_FOR_MUONS", action="store_true", 
                    help="Toggle use of GEANT4 for mupn propagation and light yield")
parser.add_option("-t", action="store_true",  dest="GPU", default=False ,help="Run on GPUs or CPUs")
parser.add_option("--gridftp", dest = "GRIDFTP", action="store_true", 
                    help="Indicate that grid FTP is being used for file I/O")
parser.add_option("--tmpdir",default=None, #TODO better default, e.g. default to using this feature
                    dest="TMPDIR", help="Tmp directory for intermediate file writing")
parser.add_option("--particle-history", dest = "PARTICLE_HISTORY", action="store_true", 
                    help="Store particle history in MCTree during CLSim")
parser.add_option("--disable-wavelength-bias", dest = "DISABLE_WAVELENGTH_BIAS", action="store_true", 
                    help="Disable wavelength bias in photon propagation (inefficient so not for mass production, but can simplify things for testing)")
parser.add_option("--keep-events-with-no-hits", dest = "KEEP_EVENTS_WITH_NO_HITS", action="store_true", 
                    help="Can optionally keep events with no hits (for testing)")

(options,args) = parser.parse_args()
if len(args) != 0:
        crap = "Got undefined options:"
        for a in args:
                crap += a
                crap += " "
        parser.error(crap)


options.FILENR=int(options.FILENR)
options.RUNNUMBER=int(options.RUNNUMBER)
if options.GPU:
        CPU=False
else:
        CPU=True


#
# Tools for purging PROPOSAL particles from the MC tree
#

#TODO move these somewhere common

# Define the particle types generated via muon energy losses in PROPOSAL
MUON_ENERGY_LOSS_PARTICLE_PARTICLE_TYPES = [ 
    dataclasses.I3Particle.EMinus,
    dataclasses.I3Particle.EPlus,
    dataclasses.I3Particle.DeltaE,
    dataclasses.I3Particle.PairProd,
    dataclasses.I3Particle.Brems,
    dataclasses.I3Particle.NuclInt,
]

MUON_PARTICLE_TYPES = [ 
    dataclasses.I3Particle.MuMinus, 
    dataclasses.I3Particle.MuPlus,
]

def get_energy_loss_particles(mc_tree, particle, out=None) :
    '''
    Helper function to grt all energy loss particles added to the MCTree from  given initial particle
    For example, get the particles added to the MCTree by PROPOSAL during muon propagation
    '''

    if out is None :
        out = []

    for child in mc_tree.children(particle) :

        # Check a few of the expected properties of particles added to the MC tree by PROPOSAL
        # assert child.fit_status == dataclasses.I3Particle.NotSet
        assert child.shape == dataclasses.I3Particle.Null

        # Check the particle type is consistent with muon energy losses
        assert child.type in MUON_ENERGY_LOSS_PARTICLE_PARTICLE_TYPES, "%s is not an energy loss particle"%(child.type)

        # Add to output
        out.append(child)

        # Also check the child's children recursively
        get_energy_loss_particles(mc_tree, child, out=out)

    return out


def purge_proposal_outputs(frame, mc_tree_key="I3MCTree", backup_mc_tree_key="I3MCTree_before_PROPOSAL_pruning") :
    '''
    This function removes all PROPOSAL outputs from a file:

      1) Energy loss particles added to the MC tree by PROPOSAL are removed
      2) Muon track lengths are set to NaN
      3) MMCTrackList is deleted 

    This is required before using GEANT4 for the muon propagation in clsim
    '''

    #TODO re-write to avoid looping twice?


    # Checks
    assert mc_tree_key in frame, "Cannot find I3MCTree key in frame : %s" % mc_tree_key
    assert backup_mc_tree_key not in frame, "Backup I3MCTree key already exists in frame : %s" % backup_mc_tree_key

    # Get MC tree
    mc_tree = frame[mc_tree_key]


    #
    # Prune muon energy loss particles
    #

    # Init container of particles for pruning
    particles_to_prune = []

    # Loop over muons
    for p in mc_tree :
        if p.type in MUON_PARTICLE_TYPES :

            # Get list of energy loss particles that need to be pruned
            particles_to_prune.extend( get_energy_loss_particles(mc_tree, p) )

    # Report
    print("Pruning %i muon energy loss particles (of %i total particles)" % (len(particles_to_prune), len(mc_tree)))

    # Make a new pruned MC tree
    pruned_mc_tree = copy.deepcopy(mc_tree)
    for p in particles_to_prune :
        pruned_mc_tree.erase(p.id)


    #
    # Set muon length to NaN (PROPOSAL sets their length)
    #

    # Loop over muons
    for p in pruned_mc_tree :
        if p.type in MUON_PARTICLE_TYPES :

            # Set length to NaN
            # Cannot simply change property or else it won't stick, need to replace particle
            p_clone = p.clone()
            p_clone.length = np.nan
            pruned_mc_tree.replace(p, p_clone)


    #
    # Remove MMC track list
    #

    if frame.Has("MMCTrackList") :
        frame.Delete("MMCTrackList")


    #
    # Update MCTree in frame
    #

    frame[backup_mc_tree_key] = mc_tree
    frame.Delete(mc_tree_key)
    frame[mc_tree_key] = pruned_mc_tree

    return True



#
# IceTray
#

from I3Tray import *
import random
from icecube import icetray, dataclasses, dataio, simclasses
from icecube import phys_services, sim_services
#from icecube import diplopia
from icecube import clsim


@gridftp_fileio
def step_2_icetray(gcd_file, input_file, output_file) :
    '''
    The step2 icetray job
    Using a decorator to add gridftp file I/O support
    '''

    photon_series = "I3Photons"
    # print('CUDA devices: ', options.DEVICE)
    tray = I3Tray()
    print('Using RUNNUMBER: ', options.RUNNUMBER)

    # Now fire up the random number generator with that seed
    from globals import max_num_files_per_dataset
    randomService = phys_services.I3SPRNGRandomService(
        seed = options.RUNNUMBER,
        nstreams = max_num_files_per_dataset,
        streamnum = options.FILENR)

    tray.AddService("I3SPRNGRandomServiceFactory","sprngrandom")(
            ("Seed",options.RUNNUMBER),
            ("StreamNum",options.FILENR),
            ("NStreams", max_num_files_per_dataset),
            ("instatefile",""),
            ("outstatefile",""),
    )

    def BasicHitFilter(frame):
        hits = 0
        if frame.Has(photon_series):
           hits = len(frame.Get(photon_series))
        if hits>0:
           return True
        else:
           return False


    ### START ###

    tray.AddModule('I3Reader', 'reader',
                FilenameList = [gcd_file, input_file]
                )


    #tray.AddModule(ModMCTree, "modmctree", mctree="I3MCTree",
    # addhadrons=True
    # )

    tray.AddModule("I3GeometryDecomposer", "I3ModuleGeoMap")

    # Purge PROPOSAL first if running GEANT4-based muon simulation
    # Must do this before running clsim
    if options.USE_GEANT4_FOR_MUONS :
        print("WARNING: Pruning PROPOSAL energy loss particles from the I3MCTree ahead of GEANT4 muon propagation")
        tray.AddModule(purge_proposal_outputs, "purge_proposal_outputs", Streams = [icetray.I3Frame.DAQ, icetray.I3Frame.Physics])

    # icemodel_path = expandvars("$I3_SRC/ice-models/resources/models/" + options.ICEMODEL)
    icemodel_path = expandvars("$I3_SRC/ice-models/resources/models/ICEMODEL/" + options.ICEMODEL) # Update path for modern code

    print('Ice model ', icemodel_path)
    print("DOM efficiency: ", options.EFFICIENCY)
    print("Setting hadronic cross energy: " , float(options.HADR_CROSSENERGY), "GeV")
    print("Setting EM cross energy: " , float(options.EM_CROSSENERGY), "GeV")
    print("Using CPUs ", CPU)
    print("Using GPUs ", options.GPU)

    gcd_file_path = gcd_file
    gcd_file = dataio.I3File(gcd_file_path)

    print("OpenCL devices: ")
    for d in clsim.I3CLSimOpenCLDevice.GetAllDevices():
        icetray.logging.log_warn(str(d))
    print("---------------------")

    # tray.AddSegment(clsim.I3CLSimMakePhotons, 'goCLSIM',
    #                 UseCPUs=CPU,
    #                 UseGPUs=options.GPU,
    # #   UseOnlyDeviceNumber=[0],
    # #                OpenCLDeviceList=[0],
    #                 MCTreeName="I3MCTree",
    #                 OutputMCTreeName="I3MCTree_clsim",
    #                 FlasherInfoVectName=None,
    #                 MMCTrackListName="MMCTrackList",
    #                 PhotonSeriesName=photon_series,
    #                 ParallelEvents=1000, 
    #                 RandomService=randomService,
    #                 IceModelLocation=icemodel_path,
    #                 #UnWeightedPhotons=True, #turn off optimizations
    #                 UseGeant4=True,
    #                 CrossoverEnergyEM=0.1,
    #                 CrossoverEnergyHadron=float(options.CROSSENERGY),
    #                 StopDetectedPhotons=True,
    # #                UseHoleIceParameterization=False, # Apply it when making hits!
    #                 HoleIceParameterization=expandvars("$I3_SRC/ice-models/resources/models/%s"%options.HOLEICE),
    #                 DoNotParallelize=False,
    #                 DOMOversizeFactor=1., 
    #                 UnshadowedFraction=options.EFFICIENCY, #normal in IC79 and older CLSim versions was 0.9, now it is 1.0
    #                 GCDFile=gcd_file,
    #                 ExtraArgumentsToI3CLSimModule={
    #                     #"UseHardcodedDeepCoreSubdetector":True, #may save some GPU memory
    #                     #"EnableDoubleBuffering":True,
    #                     "DoublePrecision":False, #will impact performance if true
    #                     "StatisticsName":"clsim_stats",
    #                     "IgnoreDOMIDs":[],
    #                     }
    #                 )


    tray.AddSegment(clsim.I3CLSimMakePhotons, 'goCLSIM',
                    UseCPUs=CPU,
                    UseGPUs=options.GPU,
    #   UseOnlyDeviceNumber=[0],
    #                OpenCLDeviceList=[0],
                    MCTreeName="I3MCTree",
                    OutputMCTreeName="I3MCTree_clsim",
                    FlasherInfoVectName=None,
                    # MMCTrackListName="MMCTrackList", # arg not present in current CLSim #TODO what has this been replaced with? does the nutau problem relate to this?
                    PhotonSeriesName=photon_series,
                    MCPESeriesName=None, # Don't want to make MCPEs (step2 handles this), just photons
                    RandomService=randomService,
                    IceModelLocation=icemodel_path,
                    UnWeightedPhotons=options.DISABLE_WAVELENGTH_BIAS,
                    UseGeant4=True,
                    UseGeant4ForMuons=options.USE_GEANT4_FOR_MUONS,
                    UseI3PropagatorService=False, # Must disable this when using `UseGeant4=True`
                    CrossoverEnergyEM=float(options.EM_CROSSENERGY),
                    CrossoverEnergyHadron=float(options.HADR_CROSSENERGY),
                    StopDetectedPhotons=True,
                    HoleIceParameterization=expandvars("$I3_SRC/ice-models/resources/models/ANGSENS/%s"%options.HOLEICE),
                    DoNotParallelize=False,
                    DOMOversizeFactor=1., # We do not use this for low energies, as is not accurate
                    DOMEfficiency=options.EFFICIENCY, #`UnshadowedFraction` -> `DOMEfficiency` (see Feb. 21, 2022 release notes, https://github.com/icecube/icetray/blob/main/clsim/RELEASE_NOTES)
                    SimpleHQEScaling=options.SIMPLE_HQE_SCALING, # Optionally, use the old HQE QE curve (backwards compatibility only)
                    GCDFile=gcd_file_path,
                    ParticleHistory=options.PARTICLE_HISTORY, # Store the secondaries produces by GEANT4, in the tree specified by `OutputMCTreeName` (note that some are coalesced by the ParticleHistoryGranularity param)
                    # ExtraArgumentsToI3CLSimModule={ # replaced with `ExtraArgumentsToI3PhotonPropagationClientModule`
                    ExtraArgumentsToI3PhotonPropagationClientModule={
                        #"UseHardcodedDeepCoreSubdetector":True, #may save some GPU memory
                        #"EnableDoubleBuffering":True,
                        # "DoublePrecision":False, #will impact performance if true # Doesn't work now
                        "StatisticsName":"clsim_stats",
                        # "ParallelEvents":1000, # This was an arg from `I3CLSimModule`, but does not seem to exist for the newer client module
                        # "IgnoreDOMIDs":[],
                        }
                    )


    # Tested that all frames go through CLSIM. Removing the ones without any hits to save space.
    if not options.KEEP_EVENTS_WITH_NO_HITS :
        tray.AddModule(BasicHitFilter, 'FilterNullPhotons', Streams = [icetray.I3Frame.DAQ, icetray.I3Frame.Physics])

    SkipKeys = ["I3MCTree_bak"]

    tray.AddModule("I3Writer","writer",
                   SkipKeys=SkipKeys,
                   Filename = output_file,
                   Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics, icetray.I3Frame.TrayInfo, icetray.I3Frame.Simulation],
                  )

    tray.AddModule("TrashCan","adios")

    tray.Execute()
    tray.Finish()


#
# Run
#

step_2_icetray(
    gridftp=options.GRIDFTP,
    gcd_file=options.GCDFILE,
    input_file=options.INFILE,
    output_file=options.OUTFILE,
    tmp_dir=options.TMPDIR,
)
