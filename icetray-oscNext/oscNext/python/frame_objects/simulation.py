'''
Tools for storing simulation information to the frame

Tom Stuttard, Michael Larson
'''

import sys, numbers

from icecube import icetray, dataclasses, phys_services
from icecube.oscNext.tools.misc import UpdateFrameObject

import numpy as np

#
# Globals
#

# Define some keys used for frame objects
# Having them here as globals make it easy for users to grab them and add to e.g. a list of output keys for their HDF5 file generation process
IN_ICE_PRIMARY_KEY = "MCInIcePrimary"
AIR_SHOWER_PRIMARY_KEY = "MCAirShowerPrimary"
EXTRA_TRUTH_INFO = "MCExtraTruthInfo"


# Particle types - useful for energy loss / secondary particle calculations
MUON_PARTICLE_TYPES = [ 
    dataclasses.I3Particle.MuMinus, 
    dataclasses.I3Particle.MuPlus,
]

TAU_PARTICLE_TYPES = [ 
    dataclasses.I3Particle.TauMinus, 
    dataclasses.I3Particle.TauPlus,
]

EM_PARTICLE_TYPES = [ # See https://github.com/icecube/icetray/blob/icecube_upgrade/clsim/private/clsim/I3CLSimLightSourceToStepConverterPPC.cxx ('isElectron' definition)
    dataclasses.I3Particle.EMinus, 
    dataclasses.I3Particle.EPlus,
    dataclasses.I3Particle.Brems,
    dataclasses.I3Particle.DeltaE,
    dataclasses.I3Particle.PairProd,
    dataclasses.I3Particle.Gamma,
    dataclasses.I3Particle.Pi0, # Pi0 decays to 2 gammas and produce EM showers
]

HADRONIC_PARTICLE_TYPES = [ # See https://github.com/icecube/icetray/blob/icecube_upgrade/clsim/private/clsim/I3CLSimLightSourceToStepConverterPPC.cxx ('isHadron' definition)
    dataclasses.I3Particle.Hadrons,
    dataclasses.I3Particle.Neutron,
    dataclasses.I3Particle.PiPlus,
    dataclasses.I3Particle.PiMinus,
    dataclasses.I3Particle.K0_Long,
    dataclasses.I3Particle.KPlus,
    dataclasses.I3Particle.KMinus,
    dataclasses.I3Particle.PPlus,
    dataclasses.I3Particle.PMinus,
    dataclasses.I3Particle.K0_Short,
    dataclasses.I3Particle.Eta,
    dataclasses.I3Particle.Lambda,
    dataclasses.I3Particle.SigmaPlus,
    dataclasses.I3Particle.Sigma0,
    dataclasses.I3Particle.SigmaMinus,
    dataclasses.I3Particle.Xi0,
    dataclasses.I3Particle.XiMinus,
    dataclasses.I3Particle.OmegaMinus,
    dataclasses.I3Particle.NeutronBar,
    dataclasses.I3Particle.LambdaBar,
    dataclasses.I3Particle.SigmaMinusBar,
    dataclasses.I3Particle.Sigma0Bar,
    dataclasses.I3Particle.SigmaPlusBar,
    dataclasses.I3Particle.Xi0Bar,
    dataclasses.I3Particle.XiPlusBar,
    dataclasses.I3Particle.OmegaPlusBar,
    dataclasses.I3Particle.DPlus,
    dataclasses.I3Particle.DMinus,
    dataclasses.I3Particle.D0,
    dataclasses.I3Particle.D0Bar,
    dataclasses.I3Particle.DsPlus,
    dataclasses.I3Particle.DsMinusBar,
    dataclasses.I3Particle.LambdacPlus,
    dataclasses.I3Particle.WPlus,
    dataclasses.I3Particle.WMinus,
    dataclasses.I3Particle.Z0,
    dataclasses.I3Particle.NuclInt,
]


#
# Functions
#

def get_mc_tree(frame) :
    '''
    Some simprod-produced files are mising I3MCTree, but have a "_preMuonProp" version that can be used, at least w.r.t. the primary particle
    '''
    mc_tree = None

    for key in ['I3MCTree','I3MCTree_preMuonProp'] : 

        if key in frame :
            mc_tree = frame[key]
            break

    assert mc_tree is not None, "Could not find I3MCTree"

    return mc_tree


def get_primary_neutrino(frame) :
    '''
    Get the primary neutrino from the frame
    '''
    mc_tree = get_mc_tree(frame)
    primary = mc_tree[0]
    assert np.abs(primary.pdg_encoding) in [12, 14, 16], "Problem getting primary neutrino"
    return primary


def get_atmospheric_muon_primaries(frame) :
    # For atmopsheric muons, have a primary from the air shower and then the primary particle in the ice which is the muon

    #TODO Need to handle multiple primaries here (e.g. CORSIKA coincidences)
    #TODO maybe use get_weighted_primary

    mc_tree = get_mc_tree(frame)

    muon_primary = dataclasses.get_most_energetic_muon(mc_tree)
    air_shower_primary = dataclasses.get_most_energetic_primary(mc_tree)

    return muon_primary, air_shower_primary


def add_primaries(frame,data_type) :
    '''
    Add the primary particles to the frame
    '''

    # if "I3MCTree" not in frame : #TODO replace (preMuonProp issue????
    #     return

    # Get the primary particle(s)
    in_ice_primary = None
    if data_type in ["genie", "nugen"] :
        in_ice_primary = get_primary_neutrino(frame)
    elif data_type in ["muongun","corsika"] :
        in_ice_primary,air_shower_primary = get_atmospheric_muon_primaries(frame)
        if AIR_SHOWER_PRIMARY_KEY not in frame : 
            if air_shower_primary is not None :
                frame[AIR_SHOWER_PRIMARY_KEY] = air_shower_primary
    if IN_ICE_PRIMARY_KEY not in frame : 
        if in_ice_primary is not None :
            frame[IN_ICE_PRIMARY_KEY] = in_ice_primary

    #TODO track,cascade,secondaries,...


class muon_background_classifier(icetray.I3ConditionalModule): 
    '''
    This module classifies the background type for muongun
    '''

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)    
        self.str36_x = 46.29
        self.str36_y = -34.88
        self.AddParameter("InnerCylinder", "Inner cylinder", 
                          phys_services.Cylinder(1000000, 200.0, dataclasses.I3Position(self.str36_x,self.str36_y,-330)))
        # making "infinite cylinder"
    def Configure(self):
        self.target_cylinder = self.GetParameter("InnerCylinder")


    def DAQ(self, frame):  
        if not frame.Has("I3MCTree"): 
            print("Error!!! Frama has no mctree... are you sure you know what you are doing? ")
            self.PushFrame(frame)    
            return True
        trueMu = dataclasses.get_most_energetic_muon(frame['I3MCTree'])
        frame['trueMuon'] = trueMu
        if trueMu==None:
            print("Error!!! Muon not found... is it a muon simulation? ")
            self.PushFrame(frame)    
            return True            
        dir2 = np.array([trueMu.dir.x, trueMu.dir.y ])/np.sqrt(trueMu.dir.x**2 + trueMu.dir.y**2)
        proj = (dir2[0]*(self.str36_x - trueMu.pos.x) +
                dir2[1]*(self.str36_y - trueMu.pos.y))
        dist = np.sqrt((self.str36_x - trueMu.pos.x)**2 +
                       (self.str36_y - trueMu.pos.y)**2)   
        rho_closest = np.sqrt(dist**2 - proj**2)
        frame['trueMuon_closestRho'] = dataclasses.I3Double(rho_closest)
        intersections = self.target_cylinder.intersection(trueMu.pos, trueMu.dir)
        entryPos = trueMu.pos + intersections.first *trueMu.dir
        exitPos  = trueMu.pos + intersections.second*trueMu.dir 
        frame['trueMuon_entryPos']        =  entryPos
        frame['trueMuon_exitPos']         =  exitPos        
        if rho_closest > 200.0:
            muon_class = 0 # Not defined class, muon missed inner cylinder completely, likely noise
        elif trueMu.pos.z > 799.9:
            muon_class = 1
        elif (trueMu.pos.z < 799.9) and (trueMu.pos.z > 200.0):
            muon_class = 2
        elif ((trueMu.pos.z < 200.0) and
             (frame['trueMuon_entryPos'].z > -220.0)):
            muon_class = 3
        elif ((trueMu.pos.z < 200.0) and
             (frame['trueMuon_entryPos'].z < -220.0) and
             (frame['trueMuon_entryPos'].z > -430.0) ):
            muon_class = 4
        elif  ((trueMu.pos.z < 200.0) and
              (frame['trueMuon_entryPos'].z < -430.0) ):
            muon_class = 5  
        else: 
            muon_class = -1 # Undefined muon class, why?
        frame['trueMuon_backgroundClass'] = icetray.I3Int( muon_class)
        self.PushFrame(frame)    
        return True
    def Physics(self, frame):  
        self.PushFrame(frame)    
        return True


@icetray.traysegment
def simulation_info( tray, name, data_type ) :
    '''
    Aggregate a bunch of useful stuff that one might want to compute for simulations
    ''' 

    #TODO Need to deal with uncontrolled segfault that is thrown if the fluxes in NuFlux have not been unzipped

    from icecube.oscNext.frame_objects.simulation import add_primaries

    # Avoid issues with case
    data_type = data_type.lower()

    # Only for simulation
    if data_type == "data" :
        return


    #
    # Truth information
    #

    # Add MC primary
    # Not relevent for pure noise triggers simulation
    if data_type != "noise" :
        tray.AddModule( add_primaries, "add_primaries", data_type=data_type )

    # Remove events with unphsyical genie events
    if data_type == "genie" : 
        from icecube.oscNext.frame_objects.genie import filter_bad_genie_events
        tray.AddModule( filter_bad_genie_events, "filter_bad_genie_events" ) # Needs to be run AFTER 'add_primaries', but BEFORE 'neutrino_extra_truth_info'
    
    # Some useful extra truth info for various particle types
    # Overwriting any previous frame object with this name, since this code has been updated
    if data_type == "genie" : 
        from icecube.oscNext.frame_objects.neutrinos import neutrino_extra_truth_info
        tray.AddModule( neutrino_extra_truth_info, "neutrino_extra_truth_info", overwrite=True )
    elif data_type == "muongun" : 
        from icecube.oscNext.frame_objects.muongun import muon_extra_truth_info
        tray.AddModule( muon_extra_truth_info, "muon_extra_truth_info", overwrite=True )



class FixSimEventHeaders(object) :
    '''
    Update the event ID in each frame to give a unique, traceable ID to every event
    Only do this for simulation, real data had event IDs specified by the DAQ

    Designed to process a single file at a time, will fail if run in a tray with 
    multiple files (or a single file made from merged files) as inout

    Scheme is:
      run_id -> dataset_id (will write this)
      sub_run_id -> file_id (will write this)
      event_id -> incrementing, unique within file (either write or just enforce this)
      sub_event_id -> not touching this, the frame splitting decides this

    '''

    def __init__(self, sub_event_stream, dataset_id, file_id, write_event_id=False, assert_unique=False, assert_ascending=False ) :

        self.sub_event_stream = sub_event_stream
        self.dataset_id = dataset_id
        self.file_id = file_id
        self.write_event_id = write_event_id
        self.assert_unique = assert_unique
        self.assert_ascending = assert_ascending

        assert isinstance(self.sub_event_stream, str)
        assert isinstance(self.dataset_id, int), "`dataset_id` must be an integer"
        assert isinstance(self.file_id, int), "`file_id` must be an integer"

        self.orig_event_record = self.EventRecord()
        self.new_event_record = self.EventRecord()
        self.event_counter = 0


    class EventRecord() :
        '''
        A sub-class for handling event ID checks
        '''

        def __init__(self) :
            self.reset()


        def reset(self) :
            self.daq_event_list = []
            self.physics_event_list = []


        def get_event_key(self,frame) :
            return ( frame["I3EventHeader"].event_id, frame["I3EventHeader"].sub_event_id )


        def get_event_list(self, frame) :
            '''
            Get the correct list depending on whetehr we have a DAQ of Physics frame
            '''
            if frame.Stop == icetray.I3Frame.DAQ :
                event_list = self.daq_event_list 
            elif frame.Stop == icetray.I3Frame.Physics :
                event_list = self.physics_event_list 
            else :
                raise Exception("Unsupported frame stop : %s" % frame.Stop)
            return event_list


        def append(self, frame, assert_unique=False, assert_ascending=False) :

            # Form the event key
            event_key = self.get_event_key(frame)

            # Get the correct list
            event_list = self.get_event_list(frame)

            # Enforce conditions
            if len(event_list) > 0 :
                if assert_unique :
                    assert event_key not in event_list, "Non-unique event found : %s" % (event_key,)
                if assert_ascending :
                    assert ( ( event_key[0] > event_list[-1][0] ) or ( ( event_key[0] == event_list[-1][0] ) and ( event_key[1] > event_list[-1][1] ) ) ), "Non-ascending event found : %s" % (event_key,)

            # Check if this is a new event
            # Only use event num, not sub-event num
            #TODO This fails if by chance there are two identical event IDs from non-identical events together. Try handling with check on primary too
            if len(event_list) == 0 :
                new_event = True
            else :
                prev_event_key = event_list[-1]
                new_event = prev_event_key[0] != event_key[0]

            # Store
            event_list.append(event_key)

            return new_event



    def __call__(self, frame) :

        # Only want Q/P frames
        if ( frame.Stop not in [ icetray.I3Frame.DAQ, icetray.I3Frame.Physics ] ) :
            return 

        # Check found header
        assert "I3EventHeader" in frame, "Could not find event header"

        # Open the header for editing
        with UpdateFrameObject(frame, "I3EventHeader") as header :

            # Only act on the requested sub-event stream (for P frames, no splitting in Q frames)
            use_frame = ( frame.Stop == icetray.I3Frame.DAQ ) or ( ( frame.Stop == icetray.I3Frame.Physics ) and ( header.sub_event_stream == self.sub_event_stream ) )
            if use_frame == False :
                return

            # Add the original event/sub-event ID that current exists in the file to the record
            is_new_event = self.orig_event_record.append( frame, assert_unique=(False if self.write_event_id else self.assert_unique), assert_ascending=(False if self.write_event_id else self.assert_ascending) )

            # Update counter
            if is_new_event :
                self.event_counter += 1

            # Update the run/subrun IDs
            header.run_id = self.dataset_id
            header.sub_run_id = self.file_id

            # Update the event ID
            if self.write_event_id :
                header.event_id = self.event_counter 

        # Add the new/updated event/sub-event ID that current exists in the file to the record
        self.new_event_record.append( frame, assert_unique=self.assert_unique, assert_ascending=self.assert_ascending )


#
# Test
#

def test_FixSimEventHeaders() :

    import numpy as np

    from I3Tray import I3Tray
    from icecube import icetray, sim_services, dataclasses

    icetray.logging.set_level(icetray.logging.I3LogLevel.LOG_INFO)


    sub_event_stream = "MySubEventStream"

    random_state = np.random.RandomState(12345)



    #
    # Check dataset/file ID write works
    #

    tray = I3Tray()

    tray.Add("BottomlessSource")

    def gen_header(frame) :
        frame["I3EventHeader"] = dataclasses.I3EventHeader()
        frame["I3EventHeader"].run_id = np.random.randint(1,100)
        frame["I3EventHeader"].sub_run_id = np.random.randint(1,100)
        frame["I3EventHeader"].sub_event_stream = sub_event_stream
    tray.Add(gen_header,"gen_header")

    dataset_id = 123
    file_id = 456
    fixer = FixSimEventHeaders( sub_event_stream=sub_event_stream, dataset_id=dataset_id, file_id=file_id )
    tray.AddModule(fixer)

    def checker_header(frame) :
        assert frame["I3EventHeader"].run_id == dataset_id, "`run_id` not equal to dataset ID"
        assert frame["I3EventHeader"].sub_run_id == file_id, "`sub)run_id` not equal to file ID"
    tray.Add(checker_header,"checker_header")

    try :
        tray.Execute(10)
        tray.Finish()
        print(">>> Test PASSED")
    except Exception as e :
        print(">>> Test FAILED : %s" % str(e))



    #
    # Check event ID checks
    #

    # Case 1, everything is grand, should survie asserts

    tray = I3Tray()

    tray.Add("BottomlessSource")

    class GenHeader(object) :
        def __init__(self) :
            self.event_id = 1
            self.sub_event_id = 0
        def __call__(self,frame) :
            frame["I3EventHeader"] = dataclasses.I3EventHeader()
            frame["I3EventHeader"].event_id = self.event_id
            frame["I3EventHeader"].sub_event_id = self.sub_event_id
            frame["I3EventHeader"].sub_event_stream = sub_event_stream
            self.event_id += 1
            self.sub_event_id = 1 if self.sub_event_id == 0 else 0

    gen_header = GenHeader()
    tray.Add(gen_header,"gen_header")

    fixer = FixSimEventHeaders( sub_event_stream=sub_event_stream, dataset_id=123, file_id=456, assert_unique=True, assert_ascending=True )
    tray.AddModule(fixer)

    try :
        tray.Execute(10)
        tray.Finish()
        print(">>> Test PASSED")
    except Exception as e :
        print(">>> Test FAILED : %s" % str(e))



    # Case 2, non-unique, should fail assert

    tray = I3Tray()

    tray.Add("BottomlessSource")

    event_keys = [ [1,0], [2,0], [2,1], [2,1]  ]

    class GenHeader(object) :
        def __init__(self) :
            self.counter = 0
        def __call__(self,frame) :
            frame["I3EventHeader"] = dataclasses.I3EventHeader()
            frame["I3EventHeader"].event_id = event_keys[self.counter][0]
            frame["I3EventHeader"].sub_event_id = event_keys[self.counter][1]
            frame["I3EventHeader"].sub_event_stream = sub_event_stream
            self.counter += 1

    gen_header = GenHeader()
    tray.Add(gen_header,"gen_header")

    fixer = FixSimEventHeaders( sub_event_stream=sub_event_stream, dataset_id=123, file_id=456, assert_unique=True, assert_ascending=True )
    tray.AddModule(fixer)

    try :
        tray.Execute(len(event_keys))
        tray.Finish()
        print(">>> Test FAILED : Didn't catch thr deliberately introduced non-unique event")
    except Exception as e :
        if str(e) == "Non-unique event found : (2, 1)" :
            print(">>> Test PASSED")
        else :
            print(">>> Test FAILED : %s" % str(e))



    # Case 3, non-incrementing, should fail assert

    tray = I3Tray()

    tray.Add("BottomlessSource")

    event_keys = [ [1,0], [2,0], [2,1], [4,0], [3,1]  ]

    class GenHeader(object) :
        def __init__(self) :
            self.counter = 0
        def __call__(self,frame) :
            frame["I3EventHeader"] = dataclasses.I3EventHeader()
            frame["I3EventHeader"].event_id = event_keys[self.counter][0]
            frame["I3EventHeader"].sub_event_id = event_keys[self.counter][1]
            frame["I3EventHeader"].sub_event_stream = sub_event_stream
            self.counter += 1

    gen_header = GenHeader()
    tray.Add(gen_header,"gen_header")

    fixer = FixSimEventHeaders( sub_event_stream=sub_event_stream, dataset_id=123, file_id=456, assert_unique=True, assert_ascending=True )
    tray.AddModule(fixer)

    try :
        tray.Execute(len(event_keys))
        tray.Finish()
        print(">>> Test FAILED : Didn't catch thr deliberately introduced non-ascending event")
    except Exception as e :
        if str(e) == "Non-ascending event found : (3, 1)" :
            print(">>> Test PASSED")
        else :
            print(">>> Test FAILED : %s" % str(e))


    #
    # Write event IDs
    #

    tray = I3Tray()

    tray.Add("BottomlessSource")

    class GenHeader(object) :
        def __init__(self) :
            self.event_id = random_state.randint(1, 100000) 
            self.sub_event_id = 1
        def __call__(self,frame) :
            new_sub_event = random_state.choice([False, True])
            if new_sub_event :
                self.sub_event_id += 1
            else :
                self.sub_event_id = 1
                self.event_id = random_state.randint(1, 100000)
            frame["I3EventHeader"] = dataclasses.I3EventHeader()
            frame["I3EventHeader"].event_id = self.event_id 
            frame["I3EventHeader"].sub_event_id = self.sub_event_id
            frame["I3EventHeader"].sub_event_stream = sub_event_stream

    gen_header = GenHeader()
    tray.Add(gen_header,"gen_header")

    counter = 0
    def check_event_id(frame) :
        global counter
        assert counter == frame["I3EventHeader"].event_id
        counter += 1
        tray.Add(check_event_id, "check_event_id")

    fixer = FixSimEventHeaders( sub_event_stream=sub_event_stream, dataset_id=123, file_id=456, write_event_id=True, assert_unique=True, assert_ascending=True )
    tray.AddModule(fixer)

    try :
        tray.Execute(20)
        tray.Finish()
        print(">>> Test PASSED")
    except Exception as e :
        print(">>> Test FAILED : %s" % str(e))




if __name__ == "__main__" :

    # Run tests
    test_FixSimEventHeaders()

