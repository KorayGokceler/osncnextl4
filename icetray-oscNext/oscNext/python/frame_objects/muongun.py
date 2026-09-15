'''
Tools for using MuonGun simulation

Tom Stuttard
'''

import numpy as np

from icecube import dataclasses, icetray, phys_services
import icecube.oscNext.frame_objects.simulation # Cannot use 'from'' due to circular dependence
from icecube.oscNext.frame_objects.weighting import WEIGHT_DICT_KEY, WEIGHT_KEY, create_weight_dict
from icecube.oscNext.tools.misc import UpdateFrameObject
from icecube.oscNext.frame_objects.geom import calc_particle_stopping_point, calc_rho_36, get_deepcore_containment, get_icecube_containment

# Weight dict keys
MUONGUN_RAW_WEIGHT_KEY = "raw_weight"
MUONGUN_KDE_PROB_KEY = "prob_passing_KDE"
MUONGUN_NUM_EVENTS_KEY = "num_events"


class SelectCylinderEvents(icetray.I3ConditionalModule): 
    '''
    This module selects events that pass through the cylinder
    '''
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)  
        self.AddParameter("TargetCylinder", "Target cylinder", None)
        self.AddOutBox('OutBox')
    
    def Configure(self):
        self.target_cylinder = self.GetParameter("TargetCylinder")
        self.n_events_all = 0 
        self.n_events_accepted = 0
        
    def DAQ(self, frame):  
        if not frame.Has("I3MCTree"): 
            # just skipp the frames without MCTree, this should not happen, but who knows
            print("Warning! No MCTree in the frame... skipping the frame" )            
            return True 
        trueMuon = dataclasses.get_most_energetic_muon(frame['I3MCTree'])
        intersection_pairs = self.target_cylinder.intersection(trueMuon.pos, trueMuon.dir)
        self.n_events_all +=1
        if np.isnan(intersection_pairs.first):
            return False
        else: 
            self.n_events_accepted+=1
            self.PushFrame(frame)
            return True

    def Finish(self):
        print("All events : " , self.n_events_all)
        print("Accepted : " , self.n_events_accepted)


def muongun_weighter(frame,overwrite=False) :
    '''
    Calculate weight for a muongun event
    '''

    #TODO some degeneracy with the following code, resolve this:
    #  https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/hignight/LE_simulation_scripts/step_1_muongun.py#L256
    #  https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/kleonard/kde_filter/kde_filter.py#L91


    #
    # Check what is already in there
    #

    # Create the weight dict if required (isn't necessarily there already for muongun)
    create_weight_dict(frame)

    # Get the weight dict
    with UpdateFrameObject(frame, WEIGHT_DICT_KEY) as weight_dict :

        # If the weight is already calculated, nothing to do (unless overwriting)
        if overwrite == False :
            if WEIGHT_KEY in weight_dict :
                return


        #
        # Calc weight
        #

        # Start from the raw weight
        weight = weight_dict[MUONGUN_RAW_WEIGHT_KEY]

        # Normalise by num events
        weight /= weight_dict[MUONGUN_NUM_EVENTS_KEY]

        # Correct for KDE-prescale, if used
        if MUONGUN_KDE_PROB_KEY :
            weight /= weight_dict[MUONGUN_KDE_PROB_KEY]

        # Write the weight to the weight dict
        weight_dict[WEIGHT_KEY] = weight



def muon_extra_truth_info(frame, output_key=None, overwrite=False) :
    '''
    Add a bunch of extra truth info about the muon
    '''

    # Defaults
    if output_key is None :
        output_key = icecube.oscNext.frame_objects.simulation.EXTRA_TRUTH_INFO

    # Check if already computed
    if (not frame.Has(output_key)) or overwrite :

        # Get the primay muon
        muon, _ = icecube.oscNext.frame_objects.simulation.get_atmospheric_muon_primaries(frame)

        # Generation point rho
        gen_point_rho36 = calc_rho_36(x=muon.pos.x, y=muon.pos.y)

        # Determine the stopping point
        stopping_point = calc_particle_stopping_point(muon)
        stopping_point_rho36 = calc_rho_36(x=stopping_point.x, y=stopping_point.y)

        # Stopped muon flags
        ic_stopping_containment = get_icecube_containment(x=stopping_point.x, y=stopping_point.y, z=stopping_point.z) 
        dc_stopping_containment = get_deepcore_containment(x=stopping_point.x, y=stopping_point.y, z=stopping_point.z) 

        #TODO flag if it crossed DeepCore/Upgrade fiducial (and store a DCA to center of fiducial volume)

        # Delete old objects if overwriting
        if overwrite and frame.Has(output_key) :
            frame.Delete(output_key)

        #TODO store information on stochastics...

        # Store in frame
        results = dataclasses.I3MapStringDouble()
        results["generation_point_rho36"] = gen_point_rho36
        results["stopping_point_x"] = stopping_point.x
        results["stopping_point_y"] = stopping_point.y
        results["stopping_point_z"] = stopping_point.z
        results["stopping_point_rho36"] = stopping_point_rho36
        results["dc_stopping_containment"] = int(dc_stopping_containment)
        results["ic_stopping_containment"] = int(ic_stopping_containment)
        frame[output_key] = results

