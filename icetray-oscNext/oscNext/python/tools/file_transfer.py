'''
Tools for file transfer across a network.

Also can be used with local files to provide an 
intermediate tmp dir functionality.

Tom Stuttard
'''

import os, subprocess, shutil

#
# File handlers
#

class FileBase(object) :

    def __init__(self, file_path, protocol, tmp_dir=None, add_prefix=False) :

        self.file_path = file_path
        self.protocol = protocol
        self.tmp_dir = tmp_dir

        # Checks
        self.supported_protocols = ["gridftp", "rsync", "local"]
        assert self.protocol in self.supported_protocols

        # Add required protocol prefix if requesed
        if add_prefix :
            if self.protocol == "gridftp" :
                self.file_path = ICECUBE_DATASTORE_GRIDFTP_PATH + self.file_path

        # Define local path
        if self.protocol == "local" :
            # If using a local file, directly use the file unless a tmp dir is provided, in which case use it
            self.tmp_path = self.file_path if self.tmp_dir is None else os.path.join( self.tmp_dir, os.path.basename(self.file_path) )
        else :
            # Otherwise, use a tmp dir (make a default if required)
            if self.tmp_dir is None :
                self.tmp_dir = ""
            self.tmp_path = os.path.join( self.tmp_dir, os.path.basename(self.file_path) )
        self.tmp_path = os.path.abspath(self.tmp_path)


class InputFile(FileBase) :
    '''
    Handler for an input file (copied TO the local area)
    '''

    def __init__(self, *args, **kwargs) :
        FileBase.__init__(self, *args, **kwargs)

    def get(self) :

        if self.protocol == "gridftp" :
            gridftp_copy(src=self.file_path, dst=self.tmp_path)
        elif self.protocol == "rsync" :
            rsync_copy(src=self.file_path, dst=self.tmp_path)
        elif self.protocol == "local" :
            if self.tmp_dir is not None :
                shutil.copyfile(src=self.file_path, dst=self.tmp_path)


class OutputFile(FileBase) :
    '''
    Handler for an output file (copied FROM the local area)
    '''

    def __init__(self, *args, **kwargs) :
        FileBase.__init__(self, *args, **kwargs)

    def send(self) :

        if self.protocol == "gridftp" :
            gridftp_copy(src=self.tmp_path, dst=self.file_path)
        elif self.protocol == "rsync" :
            rsync_copy(src=self.tmp_path, dst=self.file_path)
        elif self.protocol == "local" :
            if self.tmp_dir is not None :
                shutil.move(src=self.tmp_path, dst=self.file_path)



#
# Grid FTP
#

GRIDFTP_PREFIX = 'gsiftp://'
LOCAL_FILE_PREFIX = 'file://'

ICECUBE_DATASTORE_GRIDFTP_PATH = GRIDFTP_PREFIX+'gridftp.icecube.wisc.edu/'


def tmp_path_format(path) :
    '''
    Add required prefix for local files
    '''
    if not path.startswith(LOCAL_FILE_PREFIX) :
        path = LOCAL_FILE_PREFIX + os.path.abspath(path)
    return path


def gridftp_copy(src, dst, globus_url_copy='globus-url-copy', options=['-nodcau', '-rst', '-cd']) :
    '''
    Copy a file using grid FTP.
    Remember that the relevent certificates must be used.
    '''

    # Format local file paths
    if not src.startswith(GRIDFTP_PREFIX) :
        src = tmp_path_format(src)
    if not dst.startswith(GRIDFTP_PREFIX) :
        dst = tmp_path_format(dst)

    # Perform transfer
    command = [globus_url_copy] + options + [src, dst]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    # Error handling
    assert proc.returncode == 0, "%s failed with status %d: %s" % (globus_url_copy, proc.returncode, stderr.strip())


def test_gridftp() :

    tmp_file = "gridftp_test.txt"
    with open(tmp_file, "w") as f :
        f.write("hello\n")

    gridftp_copy(src=tmp_file, dst=ICECUBE_DATASTORE_GRIDFTP_PATH+"/data/user/stuttard/tmp/"+tmp_file)



def gridftp_fileio(func, *args, **kwargs) :
    '''
    A decorator to add gridftp fileio boiler plate to an IceTray job
    Also handles non-gridftp applications by using `gridftp=False`
    
    Also handles intermediate tmp dir for output files, to ensure partial files don't get transferred
    '''

    def inner(output_file, input_file=None, gcd_file=None, gridftp=False, tmp_dir=None, *args, **kwargs) :

        #
        # Init file transfer and tmp dir
        #

        # Enforce use of tmp dir when using grid FTP
        if gridftp :
            assert tmp_dir is not None, "Must provide a tmp dir when running using grid FTP (this prevents partially complete files from failed jobs being transferred back)"

        # Format and create tmp dir, if provided
        if tmp_dir is not None :
            tmp_dir = os.path.expanduser( os.path.expandvars(tmp_dir) ) # Expand shell variables
            if ( len(tmp_dir) > 0 ) and ( not os.path.exists(tmp_dir) ) :
                os.makedirs(tmp_dir)
                print("Created tmp dir : %s" % tmp_dir)

        # Choose protocol
        file_protocol = "gridftp" if gridftp else "local"

        # Need a tmp dir for input files only when using gridftp
        input_file_tmp_dir = tmp_dir if gridftp else None

        # Init GCD file
        # GCD may be stored on CVMFS, so no need to copy in that case
        if gcd_file is not None :
            gcd_file = InputFile(file_path=gcd_file, protocol=("local" if gcd_file.startswith("/cvmfs") else file_protocol), tmp_dir=input_file_tmp_dir, add_prefix=True )
            gcd_file.get()

        # Init input file
        if input_file is not None :
            input_file =  InputFile(file_path=input_file, protocol=file_protocol, tmp_dir=input_file_tmp_dir, add_prefix=True)
            input_file.get()

        # Init output file
        output_file = OutputFile(file_path=output_file, protocol=file_protocol, tmp_dir=tmp_dir, add_prefix=True)


        #
        # Run processing function
        #

        # Add file paths (to local locations) to kwargs
        if gcd_file is not None :
            kwargs["gcd_file"] = str(gcd_file.tmp_path)
        if input_file is not None :
            kwargs["input_file"] = str(input_file.tmp_path)
        kwargs["output_file"] = str(output_file.tmp_path)
        
        # Call the main function
        func(*args, **kwargs)


        #
        # Send data back
        #

        output_file.send()


    return inner




#
# rsync
# 

def rsync_copy(src, dst, recursive=False, ssh=True) :
    '''
    Transfer files/directories via rsync
    '''

    # Build the commnd
    command = "rsync"

    if ssh :
        command += " -e ssh"

    if recursive :
        command += -r

    command += " %s %s" % (src, dst) 

    print(command)

    # Do the transfer
    status = subprocess.call( command, shell=True )

    # Check it worked
    assert status == 0, "rsync failed : Error code = %i" % status


def test_rsync() :

    tmp_file = "rsync_test.txt"
    with open(tmp_file, "w") as f :
        f.write("hello\n")

    rsync_copy(src=tmp_file, dst="submit-1:/scratch/stuttard", recursive=False, ssh=True)


#
# Test
#

def test() :

    test_rsync()

    test_gridftp()


if __name__ == "__main__" :

    test()
