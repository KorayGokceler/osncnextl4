#!/usr/bin/env python

'''
Inject bare leptons into the detector (and propagate them if muons)

Tom Stuttard
'''

import os, argparse

import numpy as np

from I3Tray import I3Units, I3Tray
from icecube import icetray, dataclasses, dataio, phys_services
from icecube.sim_services.propagation import get_propagators

from step_1_toy import create_primary, basic_mc_boilerplate, STREAMS

#
# Functions
#

def bare_lepton_generator(
    frame, 
    random_service,
    lepton_type,
    energy_GeV=None,
    x_m=None,
    y_m=None,
    z_m=None,
    coszen=None,
    azimuth_deg=None,
) :
    '''
    Simple bare lepton generator
    '''

    # Get particle type
    if lepton_type.lower() in [ "e", "electron", "eminus" ] :
        particle_type = "EMinus"
    elif lepton_type.lower() in [ "mu", "muon", "muminus" ] :
        particle_type = "MuMinus"
    elif lepton_type.lower() in [ "etau", "tau", "tauminus" ] :
        particle_type = "TauMinus"
    else :
        raise Exception("Unknown lepton_type : %s" % lepton_type)

    # Default to a uniformly sample random energy
    if energy_GeV is None :
        energy_GeV = random_service.uniform(1., 100.)

    # Default direction (straight up)
    if coszen is None :
        coszen = -1.
    if azimuth_deg is None :
        azimuth_deg = 0.

    # Default initial positiom, nice and central and towards the bottom since travelling up
    if x_m is None :
        x_m = 0.
    if y_m is None :
        y_m = 0.
    if z_m is None :
        z_m = -450.

    # Create the muon
    create_primary(
        frame=frame,
        particle_type=particle_type,
        energy_GeV=energy_GeV,
        pos_x_m=x_m,
        pos_y_m=y_m,
        pos_z_m=z_m,
        coszen=coszen,
        azimuth_deg=azimuth_deg,
        time_ns=0.,
        shape=dataclasses.I3Particle.StartingTrack, # Manually specifying this, as if don't it will be assigned the shape `Primary`, and clsim ignores these particles
    )


#
# Main
#

if __name__ == "__main__" :


    #
    # Get inputs
    #

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-events",type=int, required=True)
    parser.add_argument("-o", "--output-file",type=str, required=True)
    parser.add_argument("-f", "--file-num",type=int, required=True)
    parser.add_argument("-d", "--dataset-num",type=int, required=True)
    parser.add_argument("-l", "--lepton-type",type=str, choices=["e", "mu", "tau"], required=True)
    parser.add_argument("-e", "--energy", type=float, default=None, required=False, help="Can optionally provide a single energy value (for scans) [GeV]")
    parser.add_argument("--proposal", action="store_true")
    parser.add_argument("--seed",type=int,required=False,default=None)
    parser.add_argument("--streamnum",type=int,required=False,default=None)
    args = parser.parse_args()


    #
    # Set up generator
    #

    tray = I3Tray()

    # Create events
    tray.AddModule("I3InfiniteSource", "infinite_source")

    # Set up basic stuff (RNG, event header)
    random_service = basic_mc_boilerplate(
        tray=tray,
        dataset_num=args.dataset_num,
        file_num=args.file_num,
        seed=args.seed,
        streamnum=args.streamnum,
    )

    # Create lepton
    tray.Add(
        bare_lepton_generator, 
        "lepton_generator", 
        lepton_type=args.lepton_type,
        energy_GeV=args.energy,
        random_service=random_service, 
        streams=[icetray.I3Frame.DAQ],
    )


    #
    # Propogate the muons in the detector
    #

    # Check for user arg
    if args.proposal :

        # Only for muons
        if args.lepton_type == "mu" :

            # 
            # Use the propagator definition from the GENIE step1 script
            #

            raise Exception("This needs updating to support modern IceTray (see genie-reader step1 script for an example to copy)")

            from icecube import PROPOSAL, sim_services
            propagators = sim_services.I3ParticleTypePropagatorServiceMap()
            
            mediadef=os.path.expandvars('$I3_BUILD/PROPOSAL/resources/mediadef')
            assert os.path.exists(mediadef)
            
            muMinusPropagator = PROPOSAL.I3PropagatorServicePROPOSAL(
                    mediadef=mediadef,
                    cylinderRadius=1200,
                    cylinderHeight=1700,
                    type=dataclasses.I3Particle.ParticleType.MuMinus)
            muPlusPropagator = PROPOSAL.I3PropagatorServicePROPOSAL(
                    mediadef=mediadef,
                    cylinderRadius=1200,
                    cylinderHeight=1700,
                    type=dataclasses.I3Particle.ParticleType.MuPlus)
                
            propagators[dataclasses.I3Particle.ParticleType.MuMinus] = muMinusPropagator
            propagators[dataclasses.I3Particle.ParticleType.MuPlus] = muPlusPropagator
            tray.AddModule('I3PropagatorModule', 'muon_propagator',
                    PropagatorServices=propagators,
                    RandomService=random_service,
                    InputMCTreeName="I3MCTree",
                    OutputMCTreeName="I3MCTree")    

            # Report propagators
            print("Propagators :")
            for particle, propagator in propagators.items() :
                print("  %s : %s" % (particle, propagator))


    #
    # Tear down
    #

    tray.AddModule('I3Writer', 'writer', Streams=STREAMS, filename=args.output_file)
    tray.Execute(args.num_events)
    tray.Finish()
