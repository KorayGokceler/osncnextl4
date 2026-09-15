'''
Tools for working with metadata.
Here metadata is assumed to a be a singel-layed dict with simple data types

Tom Stuttard
'''

#
# argparse
#

import argparse

# This defines tools requires to specify arbitrary metadata as a command line arg

class DictAction(argparse.Action) :
    '''
    argparse action that can be used to parse dicts as arguments

    Usage:
      Define arg like this: 
        parser.add_argument("-d","--my_dict",action=DictAction,nargs="+")
      At command line specify arg like this:
        -d key1:val1 key2:val2
    
    '''

    def __call__(self, parser, namespace, values, option_string=None) :
        output_dict = {}
        for element in values :
            key_val = element.split(":")
            assert len(key_val) == 2, "Invalid dict arg formatting (must be key:val) : %s" % element
            key = key_val[0]
            try :
                val = eval(key_val[1])
            except :
                val = key_val[1]
            output_dict[key] = val
        setattr(namespace, self.dest, output_dict)


#
# HDF5
#

import tables, collections

# This defines tools to read/write metadata from/to a HDF5 file

#TODO There is duplication here with `fridge/utils/hdf5_tools.py` which would be nice to resolve

def write_metadata_to_hdf(hdf_file,metadata) :
    '''
    Write a dictionary of metadata to the HDf5 file
    Store them add attributes of the `root` group
    '''            

    assert isinstance(metadata,collections.abc.Mapping), "`metadata` must be dict-like"
    #TODO enforce single layered, or flatten
    assert len(metadata), "`metadata` is empty"

    for key,val in metadata.items() :
        assert not hasattr(hdf_file.root._v_attrs,key), "HDF5 file already containes metadata with key `%s`" % key
        setattr(hdf_file.root._v_attrs,key,val) 


def read_metadata_from_hdf(hdf_file) :
    '''
    Read the HDF file attributes (used to store metadata)
    '''

    metadata = collections.OrderedDict()

    for attr_name in hdf_file.root._v_attrs._v_attrnamesuser : #TODO Include sys attribytes? (_v_attrnames)
        metadata[attr_name] = getattr(hdf_file.root._v_attrs,attr_name)

    return metadata
