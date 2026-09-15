'''
Data quality tools
Used for killing frame,s files, etc with problems

Tom Stuttard
'''

def check_object_exists(frame,object_key) : 
    '''
    Module that filters out frames that do not contain `object_key`
    '''
    return object_key in frame


class RemoveAdjacentDuplicateEventIDs(object):
    '''
    Use this module to handle issues with adjacent events with identical event IDs.

    These can cause errors when one of both of the events are ignored by a 
    conditional module, where any variables written on that module will be missing 
    from any output hDF5 file.

    This is a bug and should be fixed.
    '''

    def __init__(self) :
        self.previous_event_id = None
        self.num_frames_removed = 0

    def __call__(self,frame) :
        #TODO specify frame type?
        this_event_id = frame["I3EventHeader"].event_id
        if self.previous_event_id is not None :
            if this_event_id == self.previous_event_id :
                #TODO Also kill the other one?
                self.num_frames_removed = self.num_frames_removed + 1
                return False
        self.previous_event_id = this_event_id
        return True


# #TODO segemnt
# def clean_detector_problems() :
#     '''
#     Tray segemnt for clearing problems identified with the detector, such as:

#       (1) Flaring DOMs
#             - https://docushare.icecube.wisc.edu/dsweb/Get/Document-80514/Sparky_DOMs_JB_0623.pdf
# - https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/filterscripts/trunk/python/flaringDOMFilter.py

#       (2) Standard candle emission
#             - https://drive.google.com/file/d/0B5ScBLaqj7TWLXlkNWFCMUhBTzg/view
              # - https://drive.google.com/file/d/0B5ScBLaqj7TWLXlkNWFCMUhBTzg/view

        # (3) Coxae DOM problems
             # - https://drive.google.com/file/d/0B5ScBLaqj7TWNHh1WEJhbzJJeEE/view

#     TODO What else? CallibrationErrata?
#     '''

#   #TODO
