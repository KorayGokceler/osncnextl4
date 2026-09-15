'''
Hashing tools

Tom Stuttard
'''

#TODO This is a copy of fridge/utils/hash_tools.py, avoid the reproduction by having a common source

import os, hashlib

def md5_hash(file_path) :
    '''
    Compute the MD5 hash for a file
    Can then compare file checksums to see if they are identical
    Taken from: https://www.pythoncentral.io/hashing-files-with-python/
    '''
    assert os.path.isfile(file_path), "File does not exist, cannot compute MD5 hash : %s" % file_path
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()


def sha256_hash(file_path) :
    '''
    Compute the SHA256 hash for a file
    Can then compare file checksums to see if they are identical
    Taken from https://gist.github.com/Zireael-N/ed36997fd1a967d78cb2
    '''
    assert os.path.isfile(file_path), "File does not exist, cannot compute SHA256 hash : %s" % file_path
    with open(file_path, 'rb') as afile:
        contents = afile.read()
        sha256 = hashlib.sha256(contents).hexdigest()
    return sha256


#
# Test
#

if __name__ == "__main__" :

    # Get the hash of this file
    this_file = os.path.abspath(__file__)
    print("File         : %s" % os.path.basename(this_file))
    print("MD5 hash     : %s"%(md5_hash(this_file)))
    print("SHA256 hash  : %s"%(sha256_hash(this_file)))

