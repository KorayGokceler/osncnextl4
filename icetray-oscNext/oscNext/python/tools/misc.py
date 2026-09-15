'''
Miscellaneous tools for oscNext

Tom Stuttard
'''

import collections

class UpdateFrameObject(object) :
    '''
    Update a frame object, using the `with` syntax
    '''

    def __init__(self, frame, obj_key) :

        self.frame = frame
        self.obj_key = obj_key

        # Check the object exists
        assert self.obj_key in self.frame, "Could not find '%s' in frame" % (self.obj_key)

        # Grab the frame object
        self.obj = self.frame[self.obj_key]


    def write(self) :

        # Delete the existing object
        self.frame.Delete(self.obj_key)

        # Write the new one
        self.frame[self.obj_key] = self.obj


    def __enter__(self)  :
        return self.obj


    def __exit__(self, type, value, traceback):
        self.write()


def get_all_registered_modules() :
    '''
    Get a dict of all registered modules (separated by project)
    '''

    from icecube import icetray

    results = collections.OrderedDict()

    for p in icetray.projects() :
        print("%s" % p)
        results[p] = icetray.modules(p)

    return results

