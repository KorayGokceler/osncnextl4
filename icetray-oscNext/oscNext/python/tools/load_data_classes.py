'''
When using IceCube software, need to import the python modules containing the 
class that you want to read from a frame. However, depending on what code you 
have checked out you may only have a subset available.

This code is designed to load a bunch of useful modules containing dataclasses, but
not completely fail if they are not available for you (just warn).

Tom Stuttard
'''

import importlib, collections

# Module to support python 2 basestring class in python 3
from past.builtins import basestring


# Define a default list of frame objects
default_data_class_modules = [
    "icecube.dataclasses",
    "icecube.simclasses",
    "icecube.recclasses",
    "icecube.tensor_of_inertia",
    "icecube.millipede",
]


def load_data_classes(modules=None) :
    '''
    Call this function to load IceCube data class modules.
    Can either provide a list of modules, or use a default list.
    '''

    # Use default modules if no list provided
    if modules is None :
        modules = default_data_class_modules

    # Check list
    assert isinstance(modules,collections.abc.Sequence), "`modules` must be a list or similar"
    for m in modules : 
        assert isinstance(m,str), "Each module in `modules` must be a string" #basetring in python 2

    # Load the modules
    for module in modules :
        try : 
            importlib.import_module(module)
        except :
            print("WARNING : Failed importing '%s', some frame objects may not be loaded" % module)

