'''
Tools for storing weights in the frame

Tom Stuttard, Michael Larson
'''

from icecube import icetray, dataclasses, phys_services
from icecube.icetray import I3Bool
from icecube.oscNext.tools.misc import UpdateFrameObject

import sys, numbers
import numpy as np

#
# Globals
#

# Define some keys used for frame objects
# Having them here as globals make it easy for users to grab them and add to e.g. a list of output keys for their HDF5 file generation process
WEIGHT_DICT_KEY = "I3MCWeightDict" 
WEIGHT_KEY = "weight" 
NUM_MERGED_FILES_KEY = "num_merged_files"
UNMERGED_WEIGHT_KEY = "unmerged_weight" 
MERGED_WEIGHT_KEY = "merged_weight"
LIVETIME_KEY = "livetime_s"
FINAL_WEIGHT_KEY = "final_weight"


#
# Functions
#

def create_weight_dict(frame) :
    '''
    Create the I3MCWeightDict in the frame, if one doesn't already exist
    '''
    if WEIGHT_DICT_KEY not in frame :
        frame[WEIGHT_DICT_KEY] = dataclasses.I3MapStringDouble()


@icetray.traysegment
def weighting( tray, name, data_type, dataset=None, cr_flux_model=None, overwrite=False, num_files=None ) :
    '''
    Add oscNext weights for all data types (including detector data p0otentially)
    Either calculates them, or makes sure they're in a unified format
    '''

    # Nothing to do for detector data
    if data_type == "data" :
        return


    #
    # Calculate weight
    #

    # Weighting depends on data type
    if data_type in [ "genie", "nugen" ] :

        #
        # GENIE weights
        #

        from icecube.oscNext.frame_objects.neutrinos import NeutrinoWeighter, add_single_powerlaw_flux_weight

        is_nugen = data_type == "nugen"

        # Create the neutrino weighter and add it
        # It will apply a flux (Honda 2015 by default) and oscillations using global fit values to get example weights
        # A weight with no oscillations applied (but flux still included) will also be weritten
        neutrino_weighter = NeutrinoWeighter(
            is_nugen=is_nugen, 
            prob3=False,  # prob3 not available in modern IceTray
        )

        tray.Add( 
            neutrino_weighter, 
            "neutrino_weighter", 
            overwrite=overwrite,
        )

        # Also add a simple single power law (with no oscillations) flux case
        # Useful for unbiased samples for training machine learning algorithms
        tray.Add(
            add_single_powerlaw_flux_weight,
            "add_single_powerlaw_flux_weight",
            norm=2.e-2, # This is a number that roughly matches the overall rate for down-going numu at oscNext L5 in the Honda 2006 flux (assuming index=-3) 
            spectral_index=-3., # Something roughly atmospheric-like
            overwrite=overwrite,
        )

        # Also a flat energy weighting, also for machine learning training purposes
        tray.Add(
            add_single_powerlaw_flux_weight,
            "flat_energy_spectrum_weight",
            norm=1., #TODO Does this value matter? 
            spectral_index=0, # Something roughly atmospheric-like
            overwrite=overwrite,
        )



    elif data_type == "corsika" :

        #
        # CORSIKA
        #

        assert dataset is not None, "Must provide a dataset in order to weight CORSIKA data"

        # Add the CORSIKA weighting segment
        from icecube.oscNext.frame_objects.corsika import corsika_weighter
        tray.Add( 
            corsika_weighter, 
            'corsika_weighter', 
            dataset=dataset, 
            flux_model=cr_flux_model, 
            overwrite=overwrite,
            use_simprod_db=False, # This isn't available for newer sets that used iceprod2
         )


    elif data_type == "muongun" :

        #
        # MuonGun
        #

        #TODO Note that there is now a way to harvest the generator from the "S" frame
        #TODO See http://software.icecube.wisc.edu/documentation/projects/MuonGun/weighting.html

        # Add the MuonGun weighter
        # Note that this is assumes oscNext MuonGun step1 script was used
        from icecube.oscNext.frame_objects.muongun import muongun_weighter
        tray.Add(
            muongun_weighter, 
            'muongun_weighter', 
            overwrite=overwrite,
        )


    elif data_type == "noise" :

        #
        # Noise
        #

        # Not implementing overwrite here, as doesn't mean anything in this case...

        # Add the noise weighting segment
        #TODO Make the noise sim write to the I3MCWeightDict by default
        from icecube.oscNext.frame_objects.noise import noise_weighter
        tray.Add(
            noise_weighter, 
            'noise_weighter', 
            noise_weight_key="noise_weight",
        )


    #
    # Error handling
    #

    else :
        raise Exception("Unsupported `data_type` found : %s" % data_type)


    #
    # Compute "final" weight
    #

    # The standard "weight" variable only considers the file in question
    # A fully normalised version also considers the total number of files used
    # It is only valid when that same number of files are used
    # Note that this always updates/overwrites the final weight to reflect the num files being used right now

    # Only compute if user provided num files
    if num_files is not None :

        # Not relevent for detector data
        if data_type != "data" :

            def compute_final_weight( frame, num_files ) :

                #TODO store num_files ?

                # For the dtector data case, weight dict may not already exist. Create it if not.
                if WEIGHT_DICT_KEY not in frame :
                    frame[WEIGHT_DICT_KEY] = dataclasses.I3MapStringDouble()

                # Open the weight dict for editing
                with UpdateFrameObject(frame,WEIGHT_DICT_KEY) as weight_dict :

                    assert num_files > 0

                    # Normalise the "per file" weight by the number of files
                    final_weight = weight_dict[WEIGHT_KEY] / float(num_files) 

                    # Always update/overwrite the final weight to reflect the num files being used right now
                    weight_dict[FINAL_WEIGHT_KEY] = final_weight

                    # Do the same for the single power law weight case if present
                    from icecube.oscNext.frame_objects.neutrinos import NU_SINGLE_POWERLAW_WEIGHT_KEY, NU_SINGLE_POWERLAW_STEM
                    if NU_SINGLE_POWERLAW_WEIGHT_KEY in weight_dict :
                        weight_dict[NU_SINGLE_POWERLAW_STEM+"_final_weight"] = weight_dict[NU_SINGLE_POWERLAW_WEIGHT_KEY] / float(num_files)

            tray.Add( compute_final_weight, "compute_final_weight", num_files=num_files )



# def update_weights_in_merged_files(frame) :
#     '''
#     Update the weights in a file that is the result of merging some number of files
#     '''

#     #TODO should check when merging that files are compatible (e.g. same energy range, ...)

#     # Check weight dict is present
#     if WEIGHT_DICT_KEY not in frame :
#         return
#     # assert WEIGHT_DICT_KEY in frame, "Could not find weight dict '%s' in the frame" % WEIGHT_DICT_KEY

#     # Grab the weight dict
#     with UpdateFrameObject(frame,WEIGHT_DICT_KEY) as weight_dict :

#         # Check weights have been calculated
#         assert WEIGHT_KEY in weight_dict, "Could not find weight '%s' in the weight dict" % WEIGHT_DICT_KEY

#         # # Check merging hasn't already been performed
#         # assert NUM_MERGED_FILES_KEY in weight_dict, "File merging already performed"

#         # Check if this file is a merged file
#         if NUM_MERGED_FILES_KEY in weight_dict :

#             num_merged_files = weight_dict[NUM_MERGED_FILES_KEY]

#             # Check num merged files makes sense
#             assert isinstance(num_merged_files,numbers.Number)
#             assert num_merged_files >= 1

#             # Store the unmerged weight (if hasn't already been done)
#             if UNMERGED_WEIGHT_KEY not in weight_dict :
#                 weight_dict[UNMERGED_WEIGHT_KEY] = weight_dict[WEIGHT_KEY]

#             # Calculate merged weight
#             weight_dict[MERGED_WEIGHT_KEY] = weight_dict[UNMERGED_WEIGHT_KEY] / float(num_merged_files)

#             # Make the merged weight the "official" weight
#             weight_dict[WEIGHT_KEY] = weight_dict[MERGED_WEIGHT_KEY]
