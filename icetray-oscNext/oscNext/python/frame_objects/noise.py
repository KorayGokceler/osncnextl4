'''
Tools for using noise simulation

Tom Stuttard
'''

from icecube import icetray
from icecube.oscNext.frame_objects.weighting import WEIGHT_DICT_KEY, WEIGHT_KEY, create_weight_dict
from icecube.oscNext.tools.misc import UpdateFrameObject


def noise_weighter( frame, noise_weight_key="noise_weight" ) :
    '''
    Module for weighting noise events
    Basically just grabs the weight defined at generation time

    Noise weight calculation defined on slide 10 here: https://drive.google.com/file/d/1lLMjyw0XC4XjtvjegpfpRdxEgRYAJUqM/view
    '''


    #
    # Populate a weight dict using the input weight info
    #

    # Only if doesn't exist
    if WEIGHT_DICT_KEY not in frame :

        # Create the weight dict
        create_weight_dict(frame)

        # Open the weight dict for editing
        with UpdateFrameObject(frame,WEIGHT_DICT_KEY) as weight_dict :

            # Copy noise weight into it
            # Copy across the weight info from the existing weight structure into the weight dict
            assert noise_weight_key in frame, "Cannot find weight '%s' in frame" % weight_key
            for k,v in frame[noise_weight_key].items() :
                weight_dict[k] = v

            # Ensure consistent naming
            weight_dict[WEIGHT_KEY] = weight_dict.pop("weight")
            

    #
    # Checks
    #

    # Check found the weight
    assert WEIGHT_KEY in frame[WEIGHT_DICT_KEY]
