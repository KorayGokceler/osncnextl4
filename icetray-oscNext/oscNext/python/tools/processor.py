'''
Generic i3 file processing tool. Takes care of lots of the boiler plate stuff.

Used by the main oscNext processing, but also by potting scripts, i3->PISA converter, etc.

Tom Stuttard
'''

import os, inspect, shutil, resource, datetime, socket, collections, json

from icecube import icetray
from icecube.oscNext.selection.globals import SUB_EVENT_STREAM
from icecube.oscNext.tools.file_transfer import InputFile, OutputFile


#
# Globals
#

# Streams to keep when writing i3 files
OUTPUT_STREAMS = [ icetray.I3Frame.TrayInfo, icetray.I3Frame.Simulation, icetray.I3Frame.DAQ, icetray.I3Frame.Physics ]


#
# Processor
#

@icetray.traysegment
def data_quality( tray, name, sub_event_stream=None ) :
    '''
    Combine a bunch of data quality checks into a single segment
    ''' 

    from icecube.oscNext.tools.data_quality import check_object_exists, RemoveAdjacentDuplicateEventIDs

    # Default sub-event stream
    if sub_event_stream is None :
        sub_event_stream = SUB_EVENT_STREAM

    # Put in a few filters to remove frames that will breaks jobs downstream
    # Had to do this to handle the first bunch of pass2 L3 GENIE files
    #TODO Store number vetoed
    #TODO Drop or just ignore?
    tray.AddModule( lambda frame: frame['I3EventHeader'].sub_event_stream==sub_event_stream, "DropUnusedSubEventStreams")

    # Handle issue with missing events in HDF5 files
    remove_adjacent_duplicate_event_ids = RemoveAdjacentDuplicateEventIDs()
    tray.AddModule(remove_adjacent_duplicate_event_ids,"remove_adjacent_duplicate_event_ids")


def i3_processor(func) :
    '''
    A decorator for taking care of oscNext i3 files processing boilerplate

    Decorated function must have the following args (but does not need to use them):
        tray : IceTray instance
    Any other arg can be passed down via kwargs
    '''

    def i3_processor_core(
        gcd_file,
        i3_files,
        output_file=None,
        hdf5_file=None,
        hdf5_keys=None,
        num_files=None,
        num_events=None, # Use with a lot of caution, will screw the weighting so only really for teesting
        tmp_dir=None,
        gridftp=False,
        hash_type="sha256",
        metadata=None,
        print_tray_usage=True,
        sub_event_stream=None,
        **func_kw # Use this to let user define extra arguments for `func`
    ) :

        import os
        import numpy as np

        from I3Tray import I3Tray
        from icecube import icetray, dataio, dataclasses, recclasses, simclasses, phys_services, sim_services
        from icecube.icetray import I3Frame, I3Units
        from icecube.tableio import I3TableWriter, I3TableService
        from icecube.hdfwriter import I3HDFTableService


        #
        # Check inputs
        #

        # Record total number of files
        total_num_files = len(i3_files) # This uses the total number of files we actually have

        # Truncate to max requested num files for the calculation
        i3_files = i3_files[:num_files]

        # Record number of files used
        num_files_used = len(i3_files)

        # Check found something
        assert len(i3_files) > 0, "No input files found"

        # Default sub-event stream
        if sub_event_stream is None :
            sub_event_stream = SUB_EVENT_STREAM

        # Create the directory for the output files if required
        # Don't do this in grid FTP mode as the paths don't exist
        if not gridftp :
            if output_file is not None :
                output_dir = os.path.dirname(output_file) 
                if ( len(output_dir) > 0 ) and ( not os.path.exists(output_dir) ) :
                    os.makedirs(output_dir)

            if hdf5_file is not None :
                hdf5_dir = os.path.dirname(hdf5_file) 
                if ( len(hdf5_dir) > 0 ) and ( not os.path.exists(hdf5_dir) ) :
                    os.makedirs(hdf5_dir)



        #
        # Handle file I/O
        #

        #TODO Merge with `file_transfer.gridftp_fileio`?

        # Here there is a bunch of stuff for handling:
        #   1) Writing to an intermediate tmp dir (helps avoid partial files from faild jobs)
        #   2) Remote file transfer via gridftp

        # Choose file protocol
        # This is either directly using locally accessible files, or using GridFTP for transfer
        file_protocol = "gridftp" if gridftp else "local"

        # Format and create tmp dir, if provided
        if tmp_dir is not None :
            tmp_dir = os.path.expanduser( os.path.expandvars(tmp_dir) ) # Expand shell variables
            if ( len(tmp_dir) > 0 ) and ( not os.path.exists(tmp_dir) ) :
                os.makedirs(tmp_dir)
                print("Created tmp dir : %s" % tmp_dir)

        # Use file wrapper for all input files
        # Automatically added required prefix for GridFTP transfer if relevent (unless file is available on CVMFS)
        # If using grid FTP, use a tmp dir (rather than the current working dir) for input files to avoid the input 
        # files being copied back to the submitter at the end of the job (wasting precious disk space). Otherwise
        # do not bother with a tmp dir as can directly use the input files from their existing location. 
        input_file_tmp_dir = tmp_dir if gridftp else None
        gcd_file = InputFile(file_path=gcd_file, protocol=("local" if gcd_file.startswith("/cvmfs") else file_protocol), tmp_dir=input_file_tmp_dir, add_prefix=True ) # GCD may be stored on CVMFS
        i3_files = [ InputFile(file_path=f, protocol=file_protocol, tmp_dir=input_file_tmp_dir, add_prefix=True) for f in i3_files ] 
        
        # Similar story for output files
        # Always use a tmp dir here, as writing to a tmp dir and then moving the file to the final destination 
        # once the job ends successfully removes the issue of partial but readable i3 files being left by failed
        # jobs. So this is useful even when using local files
        if output_file is not None :
            output_file = OutputFile(file_path=output_file, protocol=file_protocol, tmp_dir=tmp_dir, add_prefix=True)
        if hdf5_file is not None :
            hdf5_file = OutputFile(file_path=hdf5_file, protocol=file_protocol, tmp_dir=tmp_dir, add_prefix=True)

        # Get all input files (each transfer them across network if applicable)
        gcd_file.get()
        for f in i3_files :
            f.get()


        #
        # Init HDF5 file writing
        #

        if hdf5_file is not None :
 
            # Handle issues with HDF5 file locking on some file systems
            os.environ["HDF5_USE_FILE_LOCKING"] = "'False'"

            # Init HDF5 keys list
            if hdf5_keys is None :
                hdf5_keys = []

            # Check keys
            for k in hdf5_keys :
                assert isinstance(k,str), "`hdf5_keys` must be strings, found %s %s" % (k,type(k))  

            # Remove duplicates from the HDF5 keys
            hdf5_keys = list(set(hdf5_keys)) # Note that this does NOT preserve order, but doesn't matter



        #
        # Prepare for the processing chain
        #

        # List of i3 files to read
        if i3_files is None :
            all_i3_files = None
        else :
            assert gcd_file is not None, "Must provide GCD file"
            all_i3_files = [gcd_file] + i3_files

        # Grab the local paths to the files (for example after file transfer)
        # Handle py2 vs 3 strings here (to avoid boost python issues)
        all_i3_files = [ str(f.tmp_path) for f in all_i3_files ]

        # Create the tray
        tray = I3Tray()


        #
        # Input streams
        #

        # Setup stagers
        # Required if want to use gridftp for file transfer,
        # and doesn't adversely affect other operation modes
        tray.context['I3FileStager'] = dataio.get_stagers() #TODO Remove now manually handling grid FTP?

        # Read input file(s)
        tray.AddModule("I3Reader", "i3_file_reader", FilenameList=all_i3_files)
        
        # Data quality checks
        tray.Add( data_quality, "data_quality" )


        #
        # Call the user function/module/segment
        #

        if func is not None :

            # Some of the args passed to this function (`i3_processor_core`)
            # may also be desired by the user. Check the function args to see 
            # and provide the required args if necessary. Checking first avoid 
            # passing args the user function wasn't expecting and breaking things. 
            user_func_arg_names = inspect.getargspec(func)[0]
            if "gcd_file" in user_func_arg_names :
                func_kw["gcd_file"] = gcd_file.tmp_path
    
            # Call the func
            func( tray=tray, **func_kw )


        #
        # Output streams
        #

        # Write to i3 file
        if output_file is not None :
            tray.AddModule( 
                'I3Writer', 
                'i3_file_writer',
                Filename=output_file.tmp_path,
                Streams=OUTPUT_STREAMS,
                DropOrphanStreams=[icetray.I3Frame.DAQ]
            )

        # Write to HDF5 file too
        if hdf5_file is not None :

            try :
                from icecube import genie_icetray # Need this if going to store weight dict (but might not exist in more modern IceTray   TODO: how to handle this)
            except :
                pass

            # Add the module
            tray.AddModule(
                I3TableWriter, 'hdf5_file_writer',
               TableService    = I3HDFTableService(hdf5_file.tmp_path),
               SubEventStreams = [sub_event_stream],
               #BookEverything  = True, #TODO Add flag to do this
               keys=hdf5_keys,
            )


        #
        # Run the tray
        #

        # Actually run the tray
        processing_start_time = datetime.datetime.now() 
        if num_events is None :
            tray.Execute()
        else :
            tray.Execute(num_events)
        processing_end_time = datetime.datetime.now() 

        # Print usage info
        # Sorted by usertime
        if print_tray_usage :
            print(">>>>>>>>>>>> Tray usage >>>>>>>>>>>>")
            tray_usage = tray.Usage()
            tray_usage_keys = [ x.key() for x in tray_usage ]
            tray_usage_usertime = [ x.data().usertime for x in tray_usage ]
            tray_usage_systime = [ x.data().systime for x in tray_usage ]
            tray_usage_ncall = [ x.data().ncall for x in tray_usage ]
            sorted_indices = np.argsort(tray_usage_usertime)
            for i in sorted_indices :
                print("  %s : usertime = %s : systime = %s : ncall = %i" % (tray_usage_keys[i],tray_usage_usertime[i],tray_usage_systime[i],tray_usage_ncall[i]) )
            print("<<<<<<<<<<<< Tray usage <<<<<<<<<<<<")

        # Done
        tray.Finish()


        #
        # Statistics
        #

        # Processing time
        processing_time_taken = processing_end_time - processing_start_time 
        print("Processing time taken : %s" % processing_time_taken)

        # Memory usage
        #TODO Units may change depending on OS (this should be fine for linux and OSX)
        max_mem_usage_MB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1e-3
        max_mem_usage_children_MB = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1e-3
        print("Max memory usage : %0.3g MB (%0.3g MB for children)" % (max_mem_usage_MB, max_mem_usage_children_MB) )


        #
        # Data provenance (compute hash for in/output files)
        #

        data_provenance = None

        if output_file is not None :

            # Can choose not to hash
            if hash_type is not None :

                from icecube.oscNext.tools.hash_tools import md5_hash, sha256_hash

                data_provenance = collections.OrderedDict()

                hashing_start_time = datetime.datetime.now()

                # Check hash type
                if hash_type == "md5" :
                    hash_func = md5_hash
                elif hash_type == "sha256" :
                    hash_func = sha256_hash
                else :
                    raise Exception("Unrecognised hash type : %s" % hash_type)

                # Hash the various input/output files and store in the provenance file
                data_provenance["gcd_file"] = collections.OrderedDict([
                    ("file_path", gcd_file.file_path ),
                    (hash_type, hash_func(gcd_file.tmp_path) ),
                ])

                data_provenance["input_files"] = []
                for f in i3_files :
                    data_provenance["input_files"].append(
                        collections.OrderedDict([
                            ("file_path", f.file_path ),
                            (hash_type, hash_func(f.tmp_path) ),
                        ])
                    )

                if output_file is not None :
                    data_provenance["output_file"] = collections.OrderedDict([
                        ("file_path", output_file.file_path ),
                        (hash_type, hash_func(output_file.tmp_path) ),
                    ])

                if hdf5_file is not None :
                    data_provenance["hdf5_file"] = collections.OrderedDict([
                        ("file_path", hdf5_file.file_path ),
                        (hash_type, hash_func(hdf5_file.tmp_path) ),
                    ])

                # Time reporting (hashing can take time)
                hashing_end_time = datetime.datetime.now() 
                hashing_time_taken = hashing_end_time - hashing_start_time 
                print("Hashing time taken : %s" % hashing_time_taken)


        #
        # Write metadata
        #

        if output_file is not None :

            # Create metadata container if none provided by the user
            if metadata is None :
                metadata = collections.OrderedDict()

            # Write some processing metadata
            metadata["processing_time_s"] = processing_time_taken.total_seconds()
            metadata["processing_end_time"] = str(processing_end_time) # Avoid storing datetime instance directly
            metadata["hostname"] = socket.gethostname()
            metadata["tmp_dir"] = tmp_dir
            metadata["max_mem_usage_MB"] = max_mem_usage_MB
            metadata["I3_SRC"] = os.path.realpath( os.path.abspath( os.path.expandvars("$I3_SRC") ) )
            metadata["I3_BUILD"] = os.path.realpath( os.path.abspath( os.path.expandvars("$I3_BUILD") ) )

            # Add provenance data
            metadata["provenance"] = data_provenance

            # Write a dedicated JSON file containing metadata
            # Only during the i3 file writing, not if just adding a HDF5 file
            # Already added the prefix to the output file which I'm using to get the file path, so no need to add prefix again
            if output_file is not None :
                metadata_file = OutputFile(file_path=(output_file.file_path.split(".i3")[0] + ".json"), protocol=("gridftp" if gridftp else "local"), tmp_dir=tmp_dir, add_prefix=False)
                with open(metadata_file.tmp_path, 'w') as f :
                    json.dump(metadata, f, indent=4)
                metadata_file.send()

            # Re-open the HDF5 file and add the metadata to that too for ease of access
            # Remove the provenance information, not required and is not formatted in a manner that the HDF5 metadata supports
            if hdf5_file is not None :
                import tables
                from icecube.oscNext.tools.metadata import write_metadata_to_hdf
                hdf = tables.open_file(filename=hdf5_file.tmp_path,mode="r+")
                metadata.pop("provenance")
                write_metadata_to_hdf(hdf,metadata)
                hdf.close()


        #
        # Move output files to final destination
        #

        # Send the output files
        if output_file is not None :
            output_file.send()

        if hdf5_file is not None :
            hdf5_file.send()

        # If created the tmp dir, now remove it
        #TODO



    # Return the wrapped function (allows use as decorator)
    return i3_processor_core
