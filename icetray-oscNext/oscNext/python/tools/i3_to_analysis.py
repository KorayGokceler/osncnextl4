'''
Produce "per event" HDF5 files from i3 files.

The data is broken up into subcategories (for example nue, numu, ...) and each variable is filled once per event.
These are useful for analysis work, much like ROOT tuples. Can use efficient masking for cuts, etc...

Includes standard information such as weighting (handling normalisation due to num files processed, etc)
Includes function for handling common event typrs, e.g. GENIE, MuonGun, etc

This is also the driver under the bonnet of the i3_to_pisa.py converter

Tom Stuttard
'''

#TODO Consider using utils.hdf5_tools.table_to_arrays to make this more general


import glob, sys, os, tables, collections, copy, random, string

import numpy as np

from icecube import icetray, dataio, dataclasses, phys_services, sim_services
from icecube.icetray import I3Frame, I3Units
from I3Tray import I3Tray

from icecube.oscNext.tools.load_data_classes import load_data_classes
load_data_classes()


#
# HDF5 helper functions
#

#TODO There is duplication here with `fridge/utils/hdf5_tools.py` which would be nice to resolve

HDF_ARRAY_TYPES = ["Array","CArray","EArray"]

def get_subnode_names(node) :
    '''
    Return the names for the subnodes of the passed node, exlcluding I3 trabl writer indexing stuff
    '''
    return [ node._v_name for node in node._f_walknodes() if node._v_name != "__I3Index__" ]


def create_or_append_to_array(hdf5_file,group,array_name,array_data,description=None) :
    '''
    Add data to an array in pytables
    If the array does not already exist, create it, otherwise append
    '''

    import warnings

    # Suppress NaturalNameWarning (e.g. when there are "." characters in the name)
    original_warnings = list(warnings.filters)
    warnings.simplefilter('ignore', tables.NaturalNameWarning)

    # Change null description to string
    if description is None : 
        description = ""

    # Check if the array already exists
    appending = True
    if array_name not in group :

        # No existing array of this name
        appending = False

        # Create an empty array
        # Using extendable array so can append if desired
        atom = tables.Atom.from_dtype(array_data.dtype)
        shape = tuple([0]*array_data.ndim)
        hdf5_file.create_earray( group, array_name, atom, shape, description )

    # Get the array
    array = getattr(group,array_name)

    # Check the array is expandanle
    assert hasattr(array,"append"), "Cannot append to array, array is not extenable (use tables.EArray)"

    # Add the data to the array
    array.append(array_data)

    # Unsuppress warning
    warnings.filters = original_warnings

    return appending


#
# i3 file helper functions
#

def get_frame_variable(frame,frame_path) :
    '''
    Return the frame variable specified by `frame_path`
    Layers of hierachy are specified by "." delimiter
    Can handle attributes or map entries, e.g. :
        "I3MCWeightDict.OneWeight" -> frame["I3MCWeightDict"]["OneWeight"]
        "MCPrimary.dir.zenith" -> frame["MCPrimary"].dir.zenith
    '''

    #TODO Move this somewhere more generic, this is a useful function

    value = None

    # Split the variable using the "." delimiter to get the frame object key and frame object variable key
    tokens = frame_path.split(".")
    frame_key = tokens[0]
    frame_subkeys = []
    for token in tokens[1:] :
        frame_subkeys.append(token)

    frame_key = str(frame_key) # py2/3 compatiblity issues

    # Get the variable from the frame
    if frame_key in frame :

        # If not frame object value, this is POD
        if len(frame_subkeys) == 0 :
            value = frame[frame_key].value

        # Otherwise get the variable from the frame object
        # The variable could be a key or an attribute
        else :
            value = frame[frame_key]
            for frame_subkey in frame_subkeys :

                frame_subkey = str(frame_subkey) # py2/3 compatiblity issues

                # Get the value if it is an attribute
                if hasattr(value,frame_subkey) :
                    value = getattr(value,frame_subkey)

                # Otherwise get it as a key
                # If that doesn't exist either, this path doesn't exist in this frame
                else :
                    try :
                        value = value[frame_subkey]
                    except : 
                        # attempt to use the subkey as an array index
                        try:
                            value = value[int(frame_subkey)].real
                        except:
                            value = None
                            pass

        # Check found something
        if value is None :
            raise Exception("Cannot find variable %s in frame, although parent exists." % (frame_path))

        # Check type
        import numbers
        assert isinstance(value,(numbers.Number,bool)), "Expected classifier input variable %s to be numeric or boolean, found %s" % (frame_path,type(value))

    return value




#
# Main conversion class
#

class I3toAnalysisConverter(object) :

    def __init__(self,output_file,metadata,support_bool=True,debug_flag=False) :

        #
        # Create the output file
        #

        self.output_file_path = output_file

        # Create the file, only if it does not exist yet
        if os.path.exists(self.output_file_path):
            raise IOError(
                'The file "%s" to be created already exists and would be'
                ' overwritten. If this is really what you want, please'
                ' move/rename/delete it manually.' % output_file
            )

        self.output_file = tables.open_file(
            filename=self.output_file_path,
            mode="w",
            title="HDF5 Events File"
        )

        # setup a debig flag we can call accross the class
        self.debug = debug_flag

        #TODO Need to remove the file if job exits prematurely...


        #
        # AOB
        #

        # Store metadata
        self.metadata = metadata

        # `h5py` does not support bools (`pytables` does)
        # Set `support_bool` to false if you will be reading with h5py
        self.support_bool = support_bool 

        # Flag indicating if have closed the file
        self.closed = False


    def __enter__(self) :
        return self


    def __exit__(self, exception_type, exception_value, traceback) :

        # If here because of an exception, rethrow it
        if exception_type is not None :
            return False

        # Otherwise close down in the standard way
        self.close()
        return False


    def close(self) :
        '''
        Run all post-processig, clean-up, etc, and close the output file
        '''

        # Only close once
        if not self.closed :

            # Check data
            self._check_outputs()

            # Store the metadata
            if self.metadata is not None :
                for key, val in self.metadata.items() :
                    self.add_metadata_entry(key, val) 

            # Close the file
            self.output_file.close()

            #TODO Free memory?
            #TODO Is this an efficient way to handle memory in HDF5 ?

            self.closed = True


    def add_metadata_entry(self,key,value):
        '''
        Add extra entry into the metadata that will be saved in the output
        file
        '''
        setattr(self.output_file.root._v_attrs, key, value) 


    def _check_outputs(self) :
        '''
        Perform checks of the final data after all conversion has been performed
        '''

        # Check wrote something
        total_num_events = 0
        for at in HDF_ARRAY_TYPES :
            for obj in self.output_file.root._f_walknodes(at) :
                total_num_events += obj.nrows
        #assert total_num_events > 0, "No output data written for given inputs"


    def convert(self,

        # Core arguments for defining what to convert
        gcd_file,
        i3_files,
        variable_map,

        stream=None,
        sub_event_stream=None,
        add_p_frames=False,

        # Useful extras
        classification=None, # (Optional) Provide an integer representing this data class (useful when producing data for training a classifier) 
        group_name=None, # (Optional) a key (or path) to for a sub-group to store the data in
        add_sim_info=False,

        # Weighting
        #TODO THis needs more thought
        weight=False,
        data_type=None,
        dataset=None,
        livetime=1.0,

        # Use can provide their own tray segments to run
        # These cannot required any non-default parameters (wrap in another segment if required)
        user_segments=None,

        # Option to produce one hdf5 file per i3 file
        # in a given folder.
        #
        # If None, conversion will produce a single hdf5 file
        per_file_conversion = None

    ) :

        '''
        A generic conversion function
        TODO Document properly
        '''

        #
        # Check inputs
        #

        # Check the input files
        num_files = len(i3_files)
        assert num_files > 0, "No input files provided for %s" % group_name

        # Check variable map
        duplicates = list(set( [v for v in variable_map if variable_map.count(v) > 1] ))
        assert len(duplicates) == 0, "Duplicate variable keys found : %s" % duplicates

        # Handle sub-event stream when creating P frames
        if add_p_frames :
            assert sub_event_stream is None, "Cannot specify an alternative sub event stream when using `add_p_frames`"
            sub_event_stream = "nullsplit"

        # Set some defaults
        if stream is None :
            stream = I3Frame.Physics

        if sub_event_stream is None :
            from icecube.oscNext.selection.globals import SUB_EVENT_STREAM
            sub_event_stream = SUB_EVENT_STREAM

        # Check weighting args
        if weight :
            assert data_type is not None, "Must provide `data_type` when weighting"

        # Fix some py2/3 compatibility issues
        if gcd_file is not None :
            gcd_file = str(gcd_file)
        i3_files = [ str(f) for f in i3_files ] 
        variable_map = [ str(v) for v in variable_map ]



        #
        # Specialised imports
        #

        #TODO This is a bit hacky, do better...

        # # Import GENIE if required
        # if len([ v for v in variable_map if "I3GENIEResultDict" in v ]) > 0 :
        #     from icecube import genie_icetray


        #
        # Check if a file-per-file conversion is requested
        #

        if not per_file_conversion is None:


            #
            # Per file conversion
            #

            assert isinstance(per_file_conversion, str),'ERROR: per_file_conversion must be a string'


            #
            # Create the directory if it doesn't exists
            #
            if not os.path.isdir(per_file_conversion):
                print('Creating folder ',per_file_conversion,'...')
                os.makedirs(per_file_conversion)


            #
            # Loop Over Each file, producing one hdf5 output per i3 file
            # The main output file only stores metadata.
            output_files = []
            for i3_f in i3_files:

                i3_out = per_file_conversion+i3_f.split('/')[-1][:-7]+'.hdf5' # Take the root of the input file name, minus the .i3.zst extension
                print(i3_out)


                # Open an hdf5 file
                output_file_handle = tables.open_file(filename=i3_out,
                                                      mode="w",
                                                      title="HDF5 Events File")


                # Choose group to write to
                # Can write to a "group" within the HDF5 (basically a directory)
                # User can specify one, or else will just use the root

                if group_name is None :

                    # If not output key specified, use the root
                    group = output_file_handle.root

                else :

                    # If an output key exists, create a group for this group_name in the 
                    # output file, or grab the existing node if it already exists
                    if hasattr(output_file_handle.root,group_name) :
                        group = getattr(output_file_handle.root,group_name)
                    else :
                        group = output_file_handle.create_group(output_file_handle.root, group_name, group_name)


                self.run_a_conversion(gcd_file=gcd_file,
                                      i3_files = [i3_f],
                                      output_file_handle=output_file_handle,
                                      variable_map = variable_map,

                                      stream=stream,
                                      sub_event_stream = sub_event_stream,
                                      add_p_frames=add_p_frames,
                                      
                                      classification = classification,
                                      group=group,
                                      add_sim_info=add_sim_info,
                                      weight = weight,
                                      data_type = data_type,
                                      dataset = dataset,
                                      livetime = livetime,
                                      user_segments = user_segments)


                # close the main metadata outputfile
                output_file_handle.close()
                
            return output_files


        # Other option: save everything on the main output file
        else:

            #
            # Single file conversion
            #

            # Choose group to write to (single hdf5 file version)
            if group_name is None :
                group = self.output_file.root
            else :
                if hasattr(self.output_file.root,group_name) :
                    group = getattr(self.output_file.root,group_name)
                else :
                    group = self.output_file.create_group(self.output_file.root, group_name, group_name)

            # Run the conversion functions
            return self.run_a_conversion(
                gcd_file=gcd_file,
                i3_files = i3_files,
                output_file_handle = self.output_file,
                variable_map = variable_map,

                stream=stream,
                sub_event_stream = sub_event_stream,
                add_p_frames=add_p_frames,

                classification = classification,
                group=group,
                add_sim_info=add_sim_info,
                weight = weight,
                data_type = data_type,
                dataset = dataset,
                livetime = livetime,
                user_segments = user_segments,
            )



    def run_a_conversion(
        self,
        gcd_file,
        i3_files,
        output_file_handle,

        variable_map,
        stream = None,
        sub_event_stream=None,
        add_p_frames=False, 

        classification = None,
        group = None,

        weight = False,
        data_type = None,
        dataset = None,
        livetime = 1.0,

        add_sim_info=False,
        user_segments = None,
    ):

        # Run an icetray job to read the input files and parse the variables
        #TODO Use the actual variable map, where the output name can be differnt (helpful for PISA)

        # Number of files within ONE ice tray
        num_files = len(i3_files)

        # Create the tray
        tray = I3Tray()

        # convert all files of the list into strings
        L = [str(l) for l in [gcd_file]+i3_files]

        # Read input files
        file_path_list = [gcd_file] + i3_files
        tray.AddModule("I3Reader", "read", FilenameList=file_path_list )
        
        # If requested, create P frames from Q frames
        # Assign a sub-event stream now, and later we make sure to use it when running the tray
        if add_p_frames :
            tray.AddModule('I3NullSplitter',"NullSplitter", SubEventStreamName=sub_event_stream)

        # Choose frames of interest
        tray.AddModule( lambda frame: frame.Stop==stream, "ChooseEventStream")
        tray.AddModule( lambda frame: frame['I3EventHeader'].sub_event_stream==sub_event_stream, "ChooseSubEventStream")

        # Add simulation info to the frame
        if add_sim_info :
            from icecube.oscNext.frame_objects.simulation import simulation_info
            tray.Add( simulation_info, "oscNext_sim_info", data_type=data_type )
        
        # Store weighting info
        # Add the final weight key to the variable map (if not already)
        #TODO Make this more general
        if weight :
            from icecube.oscNext.frame_objects.weighting import weighting, WEIGHT_DICT_KEY, FINAL_WEIGHT_KEY
            tray.Add( weighting, "oscNext_weighting", data_type=data_type, dataset=dataset, num_files=num_files )
            final_weight_key = WEIGHT_DICT_KEY + "." + FINAL_WEIGHT_KEY
            if final_weight_key not in variable_map :
                variable_map.append(final_weight_key)
        
        # Add any user-defined tray modules/segments
        if user_segments is not None :
            for i_seg,segment in enumerate(user_segments) :
                tray.Add( segment, "user_segment_%03i"%i_seg )

        

        #
        # class to Extract variables from a frame
        #
        class ExtractVariables(object) :

            def __init__(self, variable_keys, support_bool=True) :
                duplicates = list(set( [v for v in variable_keys if variable_keys.count(v) > 1] ))
                assert len(duplicates) == 0, "Duplicate variable keys found : %s" % duplicates
                self.frame_counter = 0
                self.variable_data = collections.OrderedDict( (v,[]) for v in variable_keys )
                self.support_bool = support_bool

            def __call__(self,frame) :

                # Loop over variables
                for var_path in self.variable_data.keys() :

                    # Grab the value from the frame
                    var_value = get_frame_variable(frame,var_path)

                    #TODO Should get_frame_variable have an option to check the return value is the deepest level of the structure?

                    # If couldn't find the variable, write NaN
                    if var_value is None :
                        var_value = np.NaN

                    # Convert bool->int if requested
                    if (not self.support_bool) and isinstance(var_value, bool) :
                        var_value = int(var_value)

                    # Write to output container
                    self.variable_data[var_path].append(var_value)

                # Count frames extracted
                self.frame_counter += 1


        # Add the module for grabbing variables
        extract_variables = ExtractVariables(variable_keys=variable_map, support_bool=self.support_bool) #TODO use real map
        tray.Add(extract_variables,"extract_variables")

        # Run the tray
        if self.debug:
            print('DEBUG MODE: ONLY RUNNING ON 1000 FRAMES')
            tray.Execute(1000)
        else:
            tray.Execute()
        tray.Finish()

        # Record the number of events in this call of the converter
        num_events = extract_variables.frame_counter

        #
        # Write the arrays
        #

        # Write the arrays to the output HDF5 file
        for k,v in extract_variables.variable_data.items() :
            appending = create_or_append_to_array( output_file_handle, group, k, np.asarray(v) )
            if not appending :
                print("Created array : %s" % k)


        #
        # Add useful extra information
        #

        if num_events > 0 :

            # Add the class code if user provided one
            if classification is not None :
                create_or_append_to_array( output_file_handle, group, "classification", np.full((num_events,),classification,dtype=int) )

        #
        # Add data_livetime to the root of the table
        #
        if livetime is None:
            livetime = 1.0
        create_or_append_to_array(hdf5_file=output_file_handle,
        	                       group = group,
                                   array_name = "data_livetime",
                                   array_data = livetime*np.ones(num_events,dtype=float),
                                   description=None)


        #
        # Done
        #

        # Return the data group in case caller wants to post-process
        return group




#
# i3 -> PISA variable mapping
#

# This is deprecated

# class I3FrameObjToVariableMap():
#     '''
#     This class is used to map between I3 frame objects to some output variables
#     '''

#     class MapElement():
#         """Sub-class for holding the mapping for a single variable"""

#         def __init__(self, i3_frame_obj, i3_frame_obj_variable=None, output_variable=None):
#             """
#             Params:
#                 i3_frame_obj : str
#                     Key of frame object in i3 file
#                 i3_frame_obj_variable : str
#                     Key or attribute name of the object within the frame object
#                     Only required if the frame is not a POD type, in which case `i3_frame_obj_variable` can be set to `None`
#                 output_variable : str
#                     Name of the output variable
#                     If None, will use the `i3_frame_obj` or `i3_frame_obj.i3_frame_obj_variable`
#             """
#             self.i3_frame_obj = i3_frame_obj
#             self.i3_frame_obj_variable = i3_frame_obj_variable
#             self.output_variable = output_variable

#         def __str__(self) :
#             msg = '%s' % self.i3_frame_obj
#             if self.i3_frame_obj_variable is not None :
#                 msg += '.%s' % self.i3_frame_obj_variable
#             msg += ' -> %s\n' % self.output_variable
#             return msg


#     def __init__(self) :
#         self.variables = []

#     def add(self, i3_frame_obj, i3_frame_obj_variable=None, output_variable=None):
#         """Add a single mapping."""

#         # Create output_variable if none provided
#         if output_variable is None :
#             output_variable = i3_frame_obj
#             if i3_frame_obj_variable is not None :
#                 output_variable += "." + i3_frame_obj_variable

#         # Check hasn't already been added
#         if output_variable in self.output_variables :
#             raise ValueError(
#                 'Output variable "%s" added multiple times to map!'
#                  % output_variable
#             )

#         # Add to the map
#         if i3_frame_obj_variable is not None:
#             self.variables.append(
#                 self.MapElement(
#                     i3_frame_obj=i3_frame_obj,
#                     i3_frame_obj_variable=i3_frame_obj_variable,
#                     output_variable=output_variable
#                 )
#             )

#     def add_mappings(self, mappings):
#         """Add multiple nested mappings."""
#         for i3_frame_obj, var_list in sorted(mappings.items()):
#             for (var, out_var) in ( var_list.items() if isinstance(var_list,collections.Mapping) else var_list ):
#                 self.add(
#                     i3_frame_obj=i3_frame_obj,
#                     i3_frame_obj_variable=var,
#                     output_variable=out_var
#                 )

#     @property
#     def i3_frame_objs(self) :
#         '''
#         Return all frame object keys, without duplication and preserving order
#         '''
#         output_list = []
#         for v in self.variables :
#             if v.i3_frame_obj not in output_list :
#                 output_list.append(v.i3_frame_obj)
#         return output_list

#     @property
#     def output_variables(self) :
#         '''
#         Return all output variable keys
#         '''
#         return [ v.output_variable for v in self.variables ]

#     def __getitem__(self,i3_frame_obj) :
#         return [ v for v in self.variables if v.i3_frame_obj == i3_frame_obj ]

#     def __str__(self) :
#         msg = 'I3 variable map:\n'
#         for v in self.variables :
#             msg += '  %s' % v
#         return msg
