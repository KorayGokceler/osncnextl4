'''
Tools for use with triggers

Tom Stuttard
'''

def trigger_fired(frame,trig_id) :
    '''
    Return true if the specified trigger fired
    '''

    #TODO Allow use to specify multiple triggers (OR them)

    # Get the trigger hierarchy
    if "I3TriggerHierarchy" not in frame :
        return 
    trig_hierarchy = frame['I3TriggerHierarchy']

    # Check if trigger fired
    trigger_fired = False
    for trig in trig_hierarchy:
       if trig.key.config_id == trig_id: 
        trigger_fired = True

    return trigger_fired


def add_trigger_to_frame(frame,trig_id,cut=False) :
    '''
    Function to check if the given trigger has fired
    and write the result to the frame

    Can also cut frames that fail this trigger is desired
    '''

    from icecube import dataclasses, icetray

    # Check if the trigger fired
    fired = trigger_fired(frame,trig_id=trig_id)

    # Store
    output_key = "trigger_%i" % trig_id
    frame[output_key] = icetray.I3Bool(fired)

    # Handle cut if user requested it
    if cut :
        return fired
    else :
        return

