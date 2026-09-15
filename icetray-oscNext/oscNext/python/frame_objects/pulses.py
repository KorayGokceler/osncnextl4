'''
Tools for storing pulse information to the frame

Tom Stuttard, Andrii Terliuk
'''

import collections

from scipy.constants import speed_of_light

from icecube import icetray, dataclasses, simclasses
from icecube.oscNext.frame_objects.geom import calc_rho_36
from icecube.icetray import OMKey, I3Units
from icecube.dataclasses import I3RecoPulse, I3RecoPulseSeriesMap
from icecube.oscNext.frame_objects.geom import is_dom_in_deepcore_fiducial, get_dom_geometry


#
# Globals
#

PHOTON_KEY = "I3Photons"

SIGNAL_MCPE_KEY = "MCPESeriesMap"
SIGNAL_AND_NOISE_MCPE_KEY = "MCPESeriesMap_withNoise"

MCPULSE_KEY = "MCPESeriesMap_withNoise_weighted"

DOM_LAUNCH_KEY = "InIceRawData"  #TODO what about "CleanedInIceRawData" (filter_globals.CleanedInIceRawData) ?

CALIBRATED_WAVEFORM_KEY = "CalibratedWaveform"
CALIBRATED_WAVEFORM_RANGE_KEY = CALIBRATED_WAVEFORM_KEY + "Range"
CALIBRATION_ERRATA_KEY = "CalibrationErrata"

SATURATION_WINDOWS_KEY = "SaturationWindows"

ATWD_BUFFER_FULL_TIME = 427.
FADC_BUFFER_FULL_TIME = 6.4e3
SLC_ABORT_TIME = 2.45e3

ATWD_ARTIFICIAL_DEADTIME = 45.

# Define the time window within a frame where there is actually signal for DeepCore events
SIGNAL_TIME_WINDOW = ( 9.6*I3Units.microsecond, 12.5*I3Units.microsecond )


#
# Pulse-related variables
#

@icetray.traysegment
def pulse_info( tray, name, pulses, If=lambda f: True ) :
    '''
    Add information relating to the pulses
    '''

    from icecube.common_variables import hit_multiplicity, hit_statistics, time_characteristics

    # Multiplicity
    multiplicity_key = pulses + "HitMultiplicity"
    tray.AddModule( hit_multiplicity.I3HitMultiplicityCalculator, 
                    multiplicity_key, 
                    PulseSeriesMapName=pulses, 
                    OutputI3HitMultiplicityValuesName=multiplicity_key, 
                    If=(lambda frame : If(frame) and multiplicity_key not in frame) )

    # Statistics
    stats_key = pulses + "HitStatistics"
    tray.AddModule( hit_statistics.I3HitStatisticsCalculator, 
                    stats_key, 
                    PulseSeriesMapName=pulses, 
                    OutputI3HitStatisticsValuesName=stats_key, 
                    If=(lambda frame : If(frame) and stats_key not in frame) )

    # Time characteristics
    time_key = pulses + "TimeCharacteristics"
    tray.AddModule( time_characteristics.I3TimeCharacteristicsCalculator, 
                    time_key, 
                    PulseSeriesMapName=pulses, 
                    OutputI3TimeCharacteristicsValuesName=time_key, 
                    If=(lambda frame : If(frame) and time_key not in frame) )

    # Our own custom pulse info module
    extra_pulse_info_key = pulses + "_ExtraPulseInfo"
    tray.AddModule( compute_extra_pulse_map_info, 
                    extra_pulse_info_key, 
                    pulse_map_name=pulses, 
                    output_key=extra_pulse_info_key, 
                    If=(lambda frame : If(frame) and extra_pulse_info_key not in frame) )


import numpy as np 
from icecube import dataclasses
def find_brightest_string(frame, pulsesname, 
           position_key = None, 
           rho_key = None ):
    '''
    Module to find the string with most light and store it's position to the frame
    '''
    pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulsesname)   
    if position_key == None:
        position_key = pulsesname+'BrStrPos'
    if rho_key == None:
        rho_key = pulsesname+'BrStrRho'
    str_nch = {}
    for i in xrange(1,87):
        str_nch[i] = 0
    for omkey,ompulses in pulses:
        if len(ompulses): str_nch[omkey.string] += 1
    str_nch_items = np.array(str_nch.items())
    maxstrings = str_nch_items[np.argwhere(str_nch_items[:,1]==str_nch_items[:,1].max() ),
                              0].flatten()
    pos_x = 0.0; pos_y = 0.0
    for istr in maxstrings:    
        pos_x += frame['I3Geometry'].omgeo.get(icetray.OMKey(istr, 1)).position.x
        pos_y += frame['I3Geometry'].omgeo.get(icetray.OMKey(istr, 1)).position.y
    pos_x = 1.*pos_x/float(len(maxstrings) ) 
    pos_y = 1.*pos_y/float(len(maxstrings) ) 
    pos_z = []   
    for omkey,ompulses in pulses:
        if omkey.string in maxstrings:
            pos_z.append(frame['I3Geometry'].omgeo.get(omkey).position.z)   
    pos_z=np.mean(pos_z)   
    max_rho = np.sqrt( (pos_x - 46.29)**2 + (pos_y - (-34.88) )**2 ) 
    frame[position_key] = dataclasses.I3Position(pos_x,pos_y,pos_z)    
    frame[rho_key]      = dataclasses.I3Double(max_rho)   
    return True


def parse_pulse_flags(pulse, fadc_min_width_ns=None, return_str=False) :
    '''
    Function to read the bits of the pulse flags and unpack them into meaningful info

    Using pulse width rather than flags to separate FADC vs ATWD pulses, due to a known issue.
    https://github.com/icecube/icetray/issues/2721

    Note that the issue states to use 8ns, but I have found that actually 6ns is correct (TODO link slides)
    '''
    
    # Unpack pulse flags
    hlc = ( pulse.flags >> 0 ) & 0x1

    # Use pulse width to check whether a pulse is (a) FADC-only, or includes ATWD (and probably also FADC)
    if fadc_min_width_ns is None :
        fadc_min_width_ns = 6.
    atwd = pulse.width < (fadc_min_width_ns*icetray.I3Units.ns)

    # Init list of things to return
    return_list = [hlc, atwd]

    # Optionally, return a string label of the channel
    if return_str :
        if hlc :
            if atwd :
                channel_str = "hlc_atwd"
            else :
                channel_str = "hlc_fadc"
        else :
            if atwd :
                raise Exception("Found SLC ATWD pulse, this should not exist?!?!")
            else :
                channel_str = "slc_fadc"
        return_list.append(channel_str)

    return tuple(return_list)



# def parse_pulse_flags_old(pulse) :
#     '''
#     Function to read the bits of the pulse flags and unpack them into meaningful info

#     This is deprecated due to a known bug in the pulse flags, see for more details in 'parse_pulse_flags' docs
#     '''

#     #TODO For SLC, Etienne used flags = 4 (FADC == True, ATWD == LC == False). Why is it not just bit 0?

#     # Unpack pulse flags
#     hlc = ( pulse.flags >> 0 ) & 0x1  # bit 0
#     awtd = ( pulse.flags >> 1 ) & 0x1 # bit 1
#     fadc = ( pulse.flags >> 2 ) & 0x1 # bit 2

#     return hlc, awtd, fadc


def is_om_hqe(frame, om_key) :
    '''
    Function to determine if an OM is High Quantum Efficiency (HQE) or not
    '''

    #TODO Make Upgrade compatible
    #TODO Don't think thus works with newer software that uses the dedicated HQE QE curve (rather than scaling the standard QE curve)

    # Get relative DOM efficiency
    dom_cal = calibration_data = frame["I3Calibration"].dom_cal
    assert om_key in dom_cal, "Could not find OM in DOM cal : %s" % om_key
    relative_dom_eff = dom_cal[om_key].relative_dom_eff

    # Regular DOMs have RDE = 1, HQE DOMS have RDE = 1.35
    hqe = relative_dom_eff > 1.1 # Something between 1 and 1.35

    return hqe


def compute_extra_pulse_map_info(frame, pulse_map_name, output_key=None, reco_particle=None) :
    '''
    Compute some extra information about a pulse map, beyond what is available in the common modules
    '''

    #
    # Prepare
    #

    #TODO charge asymmetry
    #TODO time to 90% charge in event

    # Defaults
    if output_key is None :
        output_key = pulse_map_name + "_ExtraPulseMapInfo"

    # Get the pulses
    pulse_series_map = frame[pulse_map_name]
    if hasattr(pulse_series_map, "apply") :
        pulse_series_map = pulse_series_map.apply(frame)

    # Get OM geometry
    om_geom_map = frame['I3Geometry'].omgeo


    #
    # Init variables
    #

    # Counters (split by channel)
    total_charge, num_hit_doms = collections.OrderedDict(), collections.OrderedDict()
    for qk in [None, "nqe", "hqe"] :
        for ck in [None, "hlc_atwd", "hlc_fadc", "slc_fadc"] :
            total_charge[(qk, ck)] = 0.
            num_hit_doms[(qk, ck)] = 0

    # Bright DOM
    brightest_dom_charge = 0.

    # Earliest, latest, etc
    earliest_pulse, latest_pulse = None, None
    earliest_pulse_om_key, latest_pulse_om_key, brightest_om_key = None, None, None
    earliest_pulse_om_geom, latest_pulse_om_geom, brightest_om_geom = None, None, None


    #
    # Extract pulse data
    #

    # Loop over OMs
    for om_key, pulses in pulse_series_map.items() :

        # Only use standard IceCube DOMs (not IceTop, special devices, etc)
        om_geom = om_geom_map[om_key]
        if om_geom.omtype != om_geom.IceCube : #TODO update for Upgrade support
            continue

        # Check for any pulses for this OM
        if len(pulses) > 0 :

            # Init DOM counters
            dom_charge = { k:0. for k in total_charge.keys() }

            # Check if DOM is HQE
            hqe = is_om_hqe(frame=frame, om_key=om_key)
            qe_key = "hqe" if hqe else "nqe"

            # Loop over pulses and grab data
            for pulse in pulses :

                # Parse flags
                hlc, atwd, channel_key = parse_pulse_flags(pulse, return_str=True)


                #
                # Update charge sums (per DOM)
                #

                # Grab pulse charge
                pulse_charge = pulse.charge

                # Check pulse charge
                assert not (pulse_charge < 0.), "Found negative pulse charge"
                assert pulse_charge != 0., "Pulse charge is exactly zero"

                # Update counters, in the various categories
                for qk in [None, qe_key] :
                    for ck in [None, channel_key] :                            
                        dom_charge[(qk, ck)] += pulse_charge


                #
                # Find earliest/latest pulses
                #

                if (earliest_pulse is None) or (pulse.time < earliest_pulse.time) :
                    earliest_pulse, earliest_pulse_om_key, earliest_pulse_om_geom = pulse, om_key, om_geom

                if (latest_pulse is None) or (pulse.time > latest_pulse.time) :
                    latest_pulse, latest_pulse_om_key, latest_pulse_om_geom = pulse, om_key, om_geom



            #
            # Increment counters (per event)
            #

            for k, q in dom_charge.items() :

                total_charge[k] += q

                if q > 0. :
                    num_hit_doms[k] += 1.


            #
            # Find brightest OM
            #

            if dom_charge[(None, None)] > brightest_dom_charge : # Not separating by QE/channel

                brightest_dom_charge, brightest_om_key, brightest_om_geom = dom_charge[(None, None)], om_key, om_geom


    #
    # Bail out if the pulse map was empty
    #

    # Check if pulse map is empty
    if num_hit_doms[(None, None)] == 0 :

        # Write minimal output here (cannot compute other variables when there are no pulses)
        frame[output_key] = dataclasses.I3MapStringDouble({ "num_hit_doms" : 0 })

        # Done
        return


    #
    # Post-process earliest/latest/bright pulse info
    #

    # Get earliest/latest pulse details
    assert earliest_pulse is not None
    assert latest_pulse is not None

    earliest_pulse_time = earliest_pulse.time
    earliest_pulse_charge = earliest_pulse.charge
    earliest_pulse_z = earliest_pulse_om_geom.position.z
    earliest_pulse_rho = calc_rho_36(x=earliest_pulse_om_geom.position.x, y=earliest_pulse_om_geom.position.y)
    earliest_pulse_om = earliest_pulse_om_key.om
    earliest_pulse_string = earliest_pulse_om_key.string

    latest_pulse_time = latest_pulse.time
    latest_pulse_charge = latest_pulse.charge
    latest_pulse_z = latest_pulse_om_geom.position.z
    latest_pulse_rho = calc_rho_36(x=latest_pulse_om_geom.position.x, y=latest_pulse_om_geom.position.y)
    latest_pulse_om = latest_pulse_om_key.om
    latest_pulse_string = latest_pulse_om_key.string

    # Get brightest DOM details
    brightest_dom_z = brightest_om_geom.position.z
    brightest_dom_rho = calc_rho_36(x=brightest_om_geom.position.x, y=brightest_om_geom.position.y)
    brightest_dom_om = brightest_om_key.om
    brightest_dom_string = brightest_om_key.string

    # Also get fraction of the event charge held by the brightest DOM (sometimes called charge asymmetry)
    brightest_dom_charge_fraction = brightest_dom_charge / total_charge[(None, None)]


    #
    # Post-process average charge per DOM
    #

    average_charge_per_dom = collections.OrderedDict()
    for k in total_charge.keys() : 
        average_charge_per_dom[k] = ( total_charge[k] / float(num_hit_doms[k]) ) if num_hit_doms[k] > 0 else np.NaN


    #
    # Derived variables based on input reco particle
    #

    #TODO merge thgis stuff into 'get_pulse_distance_metrics' instead

    # If user provided a reco particle, compute charge/hits vs length and energy
    if reco_particle is not None :
        assert frame.Has(reco_particle), "Could not find track particle in frame : %s" % reco_particle
        p = frame[reco_particle]
        assert np.isfinite(p.length)
        charge_per_m = total_charge[(None, None)] / float(p.length) if p.length > 0. else np.NaN
        num_hit_doms_per_m = num_hit_doms[(None, None)] / float(p.length) if p.length > 0. else np.NaN
        charge_per_GeV = total_charge[(None, None)] / float(p.energy) if p.energy > 0. else np.NaN
        num_hit_doms_per_GeV = num_hit_doms[(None, None)] / float(p.energy) if p.energy > 0. else np.NaN


    #
    # Write to frame
    #

    output_map = dataclasses.I3MapStringDouble()

    # Brightest DOM
    output_map["brightest_dom_charge"] = brightest_dom_charge
    output_map["brightest_dom_charge_fraction"] = brightest_dom_charge_fraction
    output_map["brightest_dom_z"] = brightest_dom_z
    output_map["brightest_dom_rho36"] = brightest_dom_rho
    output_map["brightest_dom_om"] = brightest_dom_om
    output_map["brightest_dom_string"] = brightest_dom_string

    # Earliest pulse
    output_map["earliest_pulse_time"] = earliest_pulse_time    
    output_map["earliest_pulse_charge"] = earliest_pulse_charge
    output_map["earliest_pulse_z"] = earliest_pulse_z
    output_map["earliest_pulse_rho36"] = earliest_pulse_rho
    output_map["earliest_pulse_om"] = earliest_pulse_om    
    output_map["earliest_pulse_string"] = earliest_pulse_string

    # Latest pulse
    output_map["latest_pulse_time"] = latest_pulse_time    
    output_map["latest_pulse_charge"] = latest_pulse_charge
    output_map["latest_pulse_z"] = latest_pulse_z
    output_map["latest_pulse_rho36"] = latest_pulse_rho
    output_map["latest_pulse_om"] = latest_pulse_om    
    output_map["latest_pulse_string"] = latest_pulse_string

    # Loop over QE and readout channel cases, to get counters
    for qk, ck in total_charge.keys() :

        suffix = ""
        if qk is not None :
            suffix += "_" + qk
        if ck is not None :
            suffix += "_" + ck

        output_map["total_charge"+suffix] = total_charge[(qk, ck)]
        output_map["num_hit_doms"+suffix] = num_hit_doms[(qk, ck)]
        output_map["average_charge_per_dom"+suffix] = average_charge_per_dom[(qk, ck)]

    # Reco particle-based variables
    if reco_particle is not None :
        output_map["charge_per_m"] = charge_per_m
        output_map["num_hit_doms_per_m"] = num_hit_doms_per_m
        output_map["charge_per_GeV"] = charge_per_GeV
        output_map["num_hit_doms_per_GeV"] = num_hit_doms_per_GeV


    # Write to frame
    frame[output_key] = output_map


def find_leading_pulse(pulses, F=None, time_window=None) :
    '''
    Find the leading edge reco pulse, handling potential issues with spurious early small pulses from bad pulse splitting.

    Note that pulse splitting also causes late small pulses, but these don't worry us for leading edge determination.

    Basic idea is to check if the first pulse is shortly followed by a much larger pulse, and if so to take that much larger puls as the tru leading edge.
    '''

    assert len(pulses) > 0 

    # Defaults
    if F is None :
        F = 5. # See Alex Trettin's MSc thesis (section 5.2)
    assert isinstance(F, float)

    if time_window is None :
        time_window = 20. * I3Units.nanosecond # Normally the spurious pulse splitting in +/- 10 ns
    assert isinstance(time_window, float)

    # Easy if only one pulse
    if len(pulses) == 1 :
        return pulses[0]

    #
    # Check to see if the leading pulse is followed shortly after by a much larger pulse
    #

    # Ensure pulses are ascending
    assert all(np.diff([ p.time for p in pulses ]) > 0.)

    # Start from first pulse
    leading_pulse = pulses[0]
    leading_hlc, leading_atwd = parse_pulse_flags(leading_pulse)

    # Get next pulse
    second_pulse = pulses[1]
    second_hlc, second_atwd = parse_pulse_flags(second_pulse)

    # Only pulses on same channel
    if (leading_hlc==second_hlc) and (leading_atwd==second_atwd) :

        # Check if second is within the time window
        if (second_pulse.time - leading_pulse.time) < time_window :

            # Check if second pulse has larger charge
            if second_pulse.charge > (leading_pulse.charge * F) : # Apply charge criteria

                # If made it here, use second pulse as leading pulse
                leading_pulse = second_pulse

    return leading_pulse


#
# Pulse truth flags
#

# Define all pulss sources and give them an integer value (basically an enum)
PULSE_SOURCES = collections.OrderedDict()
PULSE_SOURCES["unknown"] = -1
PULSE_SOURCES["signal"] = 0
PULSE_SOURCES["noise"] = 1
PULSE_SOURCES["afterpulse"] = 2
PULSE_SOURCES["early_afterpulse"] = 3
PULSE_SOURCES["prepulse"] = 4
PULSE_SOURCES["elastic_latepulse"] = 5
PULSE_SOURCES["inelastic_latepulse"] = 6
PULSE_SOURCES["other"] = 7

# Define time window for associating MCPE and MCPulse
# Basically driven by PMT jitter (which is asymmetric)
PE_PULSE_ASSOCIATION_TIME_WINDOW = [-8., +30.]

# Defining time windows for associating the reco and MC pulses
# Depends on which digitizer was used, since each has different sampling rate and thus time resolution
RECO_MC_PULSE_ASSOCIATION_TIME_WINDOW_ATWD = (-10., +10)
RECO_MC_PULSE_ASSOCIATION_TIME_WINDOW_FADC = (-20., +30) # Asymmetric #TODO why?


def record_mcpulse_truth_flags(
    frame, 
    signal_mcpe_map_key=None, 
    signal_and_noise_mcpe_map_key=None, 
    mcpulse_map_key=None, 
    output_key=None,
) :
    '''
    Record the source of MC pulses, e.g. does it result from signal, noise, late/afterpulses, etc?
    
    If the pulse came from an MCPE, find that MCPE, and decide if it was signal or noise. If get match or both, call it signal...
    If not from an MCPE, store the pulse's own source flag
    '''

    #TODO would be better to directly integrate this into PMTResponseSimulator and avoid matching PE with pulses by hand

    from icecube.dataclasses import I3MapKeyVectorInt


    #
    # Prepare
    #

    # Default input map keys
    if signal_mcpe_map_key is None :
        signal_mcpe_map_key = SIGNAL_MCPE_KEY
    if signal_and_noise_mcpe_map_key is None :
        signal_and_noise_mcpe_map_key = SIGNAL_AND_NOISE_MCPE_KEY
    if mcpulse_map_key is None :
        mcpulse_map_key = MCPULSE_KEY

    # Default output key
    if output_key is None :
         output_key = mcpulse_map_key + "_TruthFlags"

    # Check if output already exists
    assert not frame.Has(output_key)

    # Init output
    output_map = I3MapKeyVectorInt()


    #
    # Load data
    #

    # Get MC pulses
    # This is the output from `PMTResponseSimulator`. Despite the stupid key, this is really an `I3MCPulseSeriesMap`
    mcpulse_map = frame[str(mcpulse_map_key)]

    # Get the MCPEs
    signal_mcpe_map = frame[str(signal_mcpe_map_key)] # This is the output from `I3CLSimMakeHitsFromPhotons`
    signal_and_noise_mcpe_map = frame[str(signal_and_noise_mcpe_map_key)] # This is the output from `Vuvuzela`


    #
    # Loop over OMs/pulses
    #

    # Loop over OMs
    for om_key, mcpulse_series in mcpulse_map.items() :

        # Get MCPEs for this OM
        signal_mcpe_series = signal_mcpe_map[om_key] if om_key in signal_mcpe_map else None # Might not be any signal PEs for this DOM
        signal_and_noise_mcpe_series = signal_and_noise_mcpe_map[om_key] # Must be something otherwise could not have MCPulse
        assert len(signal_and_noise_mcpe_series) > 0, "No MCPE found" # Must be something otherwise could not have MCPulse

        # Loop over MCPulses (on this OM)
        pulse_source_values = []
        for mcpulse in mcpulse_series :

            # Check if pulse came from a PE
            # If so, will find the actual PE (either in the signal-only or signal+noise map) to learn its source
            if mcpulse.source == mcpulse.PE :

                '''
                Now want to know if the pulse came from a signal or noise MCPE
                This is not recorded, so we have to match PEs to MCPulses by their time (for a given DOM). PMT time jitter complicates this.
                
                It is possible there are matches between the MCPulse and multiple MCPE, e.g.
                  - Could randomly get two close together signal or noise PEs (especially in high E events)
                  - Could randomly get a noise and signal PE close together
                  - There seem to be bursts of noise pulses where you get a handful of noise PEs within a very short time window
                
                So with this in mind, here we are simply checking for any time match between a signal PE and the pulse, and if so calling it signal
                '''

                is_signal = False

                # Get time window around this pulse in which we will consider any PE found to be associated
                association_time_window_this_pulse = ( mcpulse.time + PE_PULSE_ASSOCIATION_TIME_WINDOW[0], mcpulse.time + PE_PULSE_ASSOCIATION_TIME_WINDOW[1] ) 

                # Check if any signal PE matches
                if signal_mcpe_series is not None :
                    for mcpe in signal_mcpe_series :
                        if mcpe.time >= association_time_window_this_pulse[0] :
                            if mcpe.time >= association_time_window_this_pulse[1] :
                                break # This stops looping once have past the time window to avoid wasted loops (since MCPEs are time sorted)
                            else :
                                is_signal = True
                                # print("Match : PE t = %i, pulse t = %i (diff = %i)" % (mcpe.time, mcpulse.time, mcpulse.time-mcpe.time))
                                break  

                # If not signal, is noise. Verify there is actually a noise PE matching
                #TODO just for testing, as is slow and notm required once tiem windoe is validated

                # Update pulse source
                if is_signal :
                    pulse_source = "signal"
                else :
                    pulse_source = "noise"
                    # print("No match")

            else :

                # Use the source flag from the pulse
                if mcpulse.source == mcpulse.AFTER_PULSE :
                    pulse_source = "afterpulse"

                elif mcpulse.source == mcpulse.EARLY_AFTER_PULSE :
                    pulse_source = "early_afterpulse"

                elif mcpulse.source == mcpulse.PRE_PULSE :
                    pulse_source = "prepulse"

                elif mcpulse.source == mcpulse.ELASTIC_LATE_PULSE :
                    pulse_source = "elastic_latepulse"

                elif mcpulse.source == mcpulse.INELASTIC_LATE_PULSE :
                    pulse_source = "inelastic_latepulse"

                else :
                    raise Exception("Unknown MC pulse source")
                    # pulse_source = "other"

            # Store results for thia pulse
            pulse_source_values.append( PULSE_SOURCES[pulse_source] )

        # Store results for this DOM
        output_map[om_key] = pulse_source_values

    # Write to frame
    frame[output_key] = output_map

    return True


def record_reco_pulse_truth_flags(
    frame, 
    reco_pulse_map_key,
    mcpulse_map_key=None,
    mcpulse_truth_map_key=None,
    output_key=None,
) :
    '''
    Record the source of reco pulses, e.g. does it result from signal, noise, late/afterpulses, etc?
    
    Uses underlying truth flags in MC pulses, assumes 'record_mcpulse_truth_flags' has already been run
    '''

    from icecube.dataclasses import I3MapKeyVectorInt


    #
    # Prepare
    #

    # Default input map keys
    if mcpulse_map_key is None :
        mcpulse_map_key = MCPULSE_KEY
    if mcpulse_truth_map_key is None :
        mcpulse_truth_map_key = mcpulse_map_key + "_TruthFlags"

    # Bail if cannot find the required inputs   #TODO would rather throw error, but since currently handling old MC still without this information, need to be flexible for now
    if not (frame.Has(mcpulse_map_key) and frame.Has(mcpulse_truth_map_key)) :
        return True

    # Default output key
    if output_key is None :
         output_key = reco_pulse_map_key + "_TruthFlags"

    # Check if output already exists
    assert not frame.Has(output_key)

    # Init output
    output_map = I3MapKeyVectorInt()
    

    #
    # Load data
    #

    # Get MC pulses and the corresponding truth flags
    mcpulse_map = frame[str(mcpulse_map_key)]
    assert frame.Has(str(mcpulse_truth_map_key)), "Could not find MCPulse truth flags, perhaps you did not run 'record_mcpulse_truth_flags'?"
    mcpulse_truth_map = frame[str(mcpulse_truth_map_key)]

    # Get the reco pulses
    reco_pulse_map = frame[str(reco_pulse_map_key)]
    if hasattr(reco_pulse_map, "apply") :
        reco_pulse_map = reco_pulse_map.apply(frame)

    # Check types
    assert isinstance(mcpulse_map, simclasses.I3MCPulseSeriesMap) 
    assert isinstance(mcpulse_truth_map, dataclasses.I3MapKeyVectorInt) 
    assert isinstance(reco_pulse_map, dataclasses.I3RecoPulseSeriesMap) 


    #
    # Loop over OMs/pulses
    #

    # Loop over OMs
    for om_key, reco_pulse_series in reco_pulse_map.items() :

        # Get MC pulses for this OM
        mcpulse_series = mcpulse_map[om_key]
        mcpulse_truth_series = mcpulse_truth_map[om_key]
        assert len(mcpulse_series) == len(mcpulse_truth_series)

        # Loop over reco pulses (on this OM)
        pulse_source_values = []
        for reco_pulse in reco_pulse_series :

            '''
            Can be ambiguity matching reco and MC pulses, since not necessarily 1:1 + there is waveform reconstruction resolution.

            Defining association as follows:
                1) If there are any signal MC pulses associated, label the reco pulse as signal
                2) Otherwise, use the closest assoicated MC pulse in time for the truth label
                3) If no associations, mark as "unknown" - TODO why does this happen, and how often?
            '''

            is_signal = False

            # Get time window around this pulse in which we will consider any PE found to be associated
            # This depends on digitizer
            _, atwd = parse_pulse_flags(reco_pulse)
            tw = RECO_MC_PULSE_ASSOCIATION_TIME_WINDOW_ATWD if atwd else RECO_MC_PULSE_ASSOCIATION_TIME_WINDOW_FADC
            association_time_window_this_pulse = ( reco_pulse.time + tw[0], reco_pulse.time + tw[1] ) 

            # Perform the association
            closest_associated_mcpulse = None
            for mcpulse, mcpulse_truth in zip(mcpulse_series, mcpulse_truth_series) :
                if mcpulse.time >= association_time_window_this_pulse[0] :
                    if mcpulse.time >= association_time_window_this_pulse[1] :
                        break # This stops looping once have past the time window to avoid wasted loops (since the pulses are time sorted)
                    else :
                        if mcpulse_truth == PULSE_SOURCES["signal"] :
                            is_signal = True
                            break
                        else :
                            if closest_associated_mcpulse is None :
                                closest_associated_mcpulse = (mcpulse, mcpulse_truth)
                            else :
                                if np.abs(reco_pulse.time - mcpulse.time) < np.abs(reco_pulse.time - closest_associated_mcpulse[0].time) :
                                    closest_associated_mcpulse = (mcpulse, mcpulse_truth)

            # Assign pulse source
            if is_signal :
                pulse_source = PULSE_SOURCES["signal"]
            else :
                if closest_associated_mcpulse is None :
                    pulse_source = PULSE_SOURCES["unknown"]
                else :
                    pulse_source = closest_associated_mcpulse[1]

            # Store results for this pulse
            pulse_source_values.append( pulse_source )

        # Store results for this DOM
        output_map[om_key] = pulse_source_values

    # Write to frame
    frame[output_key] = output_map

    return True


def get_signal_only_pulse_map(frame, pulse_map_key, truth_flags_map_key=None, output_key=None, backwards_compatiblity=False, pulse_types_to_remove=None) :
    '''
    Take a pulse series and apply the truth flags to remove noise pulses (decays, late pulse, afterpulses, etc), leaving a pulse series map that is signal-only

    Can optionally choose a specific type of pulse to remove (pulse_types_to_remove), rather than all non-signal pulses

    This only works MC of course, and only for datasets that have the pulse truth flags included
    '''

    # Check if user defined a specific type of pulse to remove
    if pulse_types_to_remove is not None :
        assert not backwards_compatiblity, "Incompatible args"
        assert isinstance(pulse_types_to_remove, list)
        pulse_types_to_remove_values = []
        for pt in pulse_types_to_remove :
            assert isinstance(pt, str)
            assert pt in PULSE_SOURCES, "User specified an unknown type '%s' for 'pulse_types_to_remove', choose from %s" % (pt, list(PULSE_SOURCES.keys()))
            pulse_types_to_remove_values.append(PULSE_SOURCES[pt]) # Get ints rather than strings for comaprison later

    # Defaults
    if output_key is None :
        if pulse_types_to_remove is None :
            output_key = pulse_map_key + "_SignalOnly"
        else :
            output_key = pulse_map_key + "_No%s"%("".join([ pt.replace("_", " ").title().replace("_", "") for pt in pulse_types_to_remove ]))

    if truth_flags_map_key is None :
        truth_flags_map_key = pulse_map_key + "_TruthFlags"

    # Get the pulses
    input_pulse_map = frame[str(pulse_map_key)]
    if hasattr(input_pulse_map, "apply") :
        input_pulse_map = input_pulse_map.apply(frame)

    # Get the truth flags
    truth_flags_map = frame[str(truth_flags_map_key)]

    # Init output pulse map
    assert output_key not in frame, "'%s' already exists in frame" % output_key
    output_pulse_map = dataclasses.I3RecoPulseSeriesMap()

    # Loop OMs
    for om_key, input_pulses in input_pulse_map.items() :

        # Init container
        output_pulses = []

        # Get flags
        assert om_key in truth_flags_map
        truth_flags = truth_flags_map[om_key]
        assert len(truth_flags) == len(input_pulses)

        # Loop over pulses
        for input_pulse, truth_flag in zip(input_pulses, truth_flags) :

            # Only store signal pulses
            # Handle old version of code (used to be 0/1 for noise/signal, but now more categories)
            if backwards_compatiblity :
                if truth_flag > 0 :
                    output_pulses.append(input_pulse)
            else :
                if pulse_types_to_remove is None :
                    if truth_flag == PULSE_SOURCES["signal"] :
                        output_pulses.append(input_pulse)
                else :
                    if truth_flag not in pulse_types_to_remove_values :
                        output_pulses.append(input_pulse)

        # Add to output container
        output_pulse_map[om_key] = dataclasses.I3RecoPulseSeries(output_pulses)

    # Add to frame
    frame[str(output_key)] = output_pulse_map



#
# Pulse cleaning
#

def apply_deadtime_to_pulses(input_pulses, truth_flags=None) :
    '''
    Apply deadtime to a pulse series/map to remove issues with pulses being split in wavedeform (very different for data and MC)

    The deadtime depends on the readout channel

    Can optionally provide truth flags too for processing along with the pulses themselves
    '''

    raise Exception("Not sure deadtime is a good idea anymore, since there can be spurious pre-pulses as well as suprious post-pulses. Instead should implement pulse merging.")

    #TODO better to merge to preserve charge?

    # Handle map vs series
    if isinstance(input_pulses, dataclasses.I3RecoPulseSeriesMap) :

        #
        # Input is pulse map
        #

        input_pulse_map = input_pulses
        truth_flags_map = truth_flags

        output_pulse_map = collections.OrderedDict() #TODO output I3RecoPulseSeriesMap
        output_truth_flags_map = collections.OrderedDict() #TODO output I3RecoPulseSeriesMap

        # Loop over DOMs in map
        for om_key, input_pulses in input_pulse_map.items() :

            # Get truth flags
            if truth_flags_map is not None :
                truth_flags = truth_flags_map[om_key]

            # Call function recursively to process pulses
            ret = apply_deadtime_to_pulses(input_pulses=input_pulses, truth_flags=truth_flags)
            if truth_flags_map is None :
                output_pulse_map[om_key] = ret
            else :
                output_pulse_map[om_key], output_truth_flags_map[om_key] = ret

        # Return
        if truth_flags is None :
            return output_pulse_map
        else :
            return output_pulse_map, output_truth_flags_map


    elif isinstance(input_pulses, dataclasses.I3RecoPulseSeries) :


        #
        # Input is pulse series
        #

        output_pulses = []
        output_truth_flags = None if truth_flags is None else []

        # Loop input pulses
        dead_until = None
        for i_pulse, pulse in enumerate(input_pulses) :

            # Check if there is a deadtime active
            # If there is, check if this pulses falls within it, and skip it if so
            if dead_until is not None :
                if pulse.time <= dead_until :
                    continue

            # Keep pulse
            output_pulses.append(pulse)
            if truth_flags is not None :
                output_truth_flags.append( truth_flags[i_pulse] )

            # Start a new deadtime
            # This depends on the readout channel
            hlc, atwd = parse_pulse_flags(pulse, return_str=False)
            if hlc :
                dead_until = pulse.time + ATWD_ARTIFICIAL_DEADTIME # Also applying this for HLC+FADC (not sure what is best in that case)
            else :
                dead_until = pulse.time + SLC_ABORT_TIME # SLC is dead for further pulses until the abort


        # Return
        if truth_flags is None :
            return output_pulses
        else :
            return output_pulses, output_truth_flags



@icetray.traysegment
def prereco_pulse_cleaning(
    tray, 
    name, 
    output_key, 
    input_pulse_map_key=None,
    # Steer what to keep/reject (some arguments may be degenerate/incompatible (e.g. apply_deadtime/first_pulse_only))
    signal_time_window_only=False, # Only take pulses within the signal time window
    fiducial_doms_only=False, # Only take pulses from fiducial DOMs
    dom_stats=False, # Only return "per DOM" rather than "per pulse" information. In this case, the time is the leading edge and the charge is the charge sum (or 1 if "no_charge" is set)
    # apply_deadtime=False,
    # SRT cleaning settings
    apply_srt=False,  # Apply SRT algorithm to remove acausal decay noise pulses. Can see settings below
    srt_self_coincidence=False,
    srt_dust_layer_correction=False,
    srt_max_n_iterations=None,
    srt_ic_r=None,
    srt_ic_t=None,
    srt_dc_r=None,
    srt_dc_t=None,
    srt_hlccore_seeding=False,
    # Steer pulse properties
    no_charge=False, # Drop charge informatio, replace 0/1 for hit/no-it
    round_time=False, # Round time to nearest ns (useful because SuperDST does this, but re-extracted waveforms do not, which casues data-MC mismatch)
) :
    '''
    Pulse cleaning module for preparing pulses for reconstruction inputs.

    Includes various options designed to address the following known issues:
      - Decay noise
      - Late/afterpulsing
      - Pulse splitting
      - Charge mis-calibration
      - Round vs fractional seconds in pulse time (SuperDST vs re-extracted waveforms)
    '''

    #TOO option for merging split pulses rather than "dom_stats"

    #
    # Init
    #

    # Check for incompatible args
    if dom_stats :
        # assert not apply_deadtime
        assert not srt_self_coincidence

    # Defaults
    if input_pulse_map_key is None :
        input_pulse_map_key = UNCLEANED_PULSES

    if srt_max_n_iterations is None :
        srt_max_n_iterations = -1

    if srt_ic_r is None :
        srt_ic_r = 150.*I3Units.m
    if srt_ic_t is None :
        srt_ic_t = 1000.*I3Units.ns

    if srt_dc_r is None :
        srt_dc_r = 75.*I3Units.m
    if srt_dc_t is None :
        srt_dc_t = 500.*I3Units.ns

    # Define an intermediate frame object
    intermediate_output_key = output_key + "_preprocessing"

    # Check output doesn't already exist
    def must_not_exist(frame, key) : #TODO probably a standard module to do this in icetray
        assert key not in frame, "'%s' already exists in frame"%key
    tray.AddModule(must_not_exist, "must_not_exist_%s"%output_key, key=output_key)
    tray.AddModule(must_not_exist, "must_not_exist_%s"%intermediate_output_key, key=intermediate_output_key)


    #
    # Pre-process pulse map to address a range of known isses
    #

    # Define a function to perform the first step in the claning (before running standard modules later)

    def _preprocess_pulse_map(frame, output_key, input_pulse_map_key=None) :

        #
        # Prepare
        #

        # Load input pulses
        input_pulse_map = frame[input_pulse_map_key]
        if hasattr(input_pulse_map, "apply") :
            input_pulse_map = input_pulse_map.apply(frame)

        # Init output
        assert not frame.Has(output_key)
        output_map = I3RecoPulseSeriesMap()

        # Grab geometry
        omgeo = frame['I3Geometry'].omgeo


        #
        # Loop over inputs
        #

        # Loop over DOMs
        for om_key, input_pulses in input_pulse_map.items() :

            # Only use standard IceCube DOMs (not IceTop, special devices, etc)
            if omgeo[om_key].omtype != omgeo[om_key].IceCube : #TODO update for Upgrade support
                continue

            # Only accept fiducial DOMs, if requested
            if fiducial_doms_only :
                if not is_dom_in_deepcore_fiducial(frame=frame, om_key=om_key, use_dom_list=True) :
                    continue

            #
            # Get pulses to consider
            #

            output_pulses = []

            # Loop over pulses
            for i_pulse, input_pulse in enumerate(input_pulses) :

                # Only accept pulses within a tight time window around the signal region
                if signal_time_window_only :
                    if (input_pulse.time < SIGNAL_TIME_WINDOW[0]) or (input_pulse.time > SIGNAL_TIME_WINDOW[1]) :
                        continue

                #  If made it here, consider this pulse
                output_pulses.append(input_pulse)

            # Bail if nothing remains
            if len(output_pulses) == 0 :
                continue


            #
            # Apply deadtime
            #

            #TODO this needs rethinking, a proper pulse merging should really be used

            # if apply_deadtime :
            #     output_pulses = apply_deadtime_to_pulses(output_pulses)


            #
            # DOM stats mode
            #

            if dom_stats :

                assert no_charge, "Handling of charge sum for split pulses not yet implemted"

                # Find the leading edge pulse
                leading_pulse = find_leading_pulse(output_pulses, F=5., time_window=20.*I3Units.nanosecond)

                # Use this as the pulse
                output_pulses = [leading_pulse]


            else :
                raise Exception("Need to implement pulse merging to handle pulse splitting before use anything other than DOM stats mode")



            #
            # Update pulse time, charge, etc
            #

            updated_output_pulses = []

            for pulse in output_pulses :

                # Round the pulse time to the nearest second, if requested
                # This addresses differences between SuperDST and re-extracted pulses
                if round_time :
                    pulse_time = round(pulse.time)
                else :
                    pulse_time = pulse.time

                # Ignore charge and instead count every pulse as a single photon
                # Mitigates potential charge modelling issues, and really most the charge variability relates to PMTs, not physics. We really only care about photons.
                if no_charge :
                    pulse_charge = 1.  #TODO consider instead using mean of SPE distribution?
                else :
                    pulse_charge = pulse.charge

                # Create the output pulse
                updated_pulse = I3RecoPulse()
                updated_pulse.flags = pulse.flags  #TODO fix FADC/ATWD flags?
                updated_pulse.time = pulse_time
                updated_pulse.charge = pulse_charge
                updated_pulse.width = pulse.width

                # Store to pulse series
                updated_output_pulses.append(updated_pulse)

            # Store to pulse map
            output_map[om_key] = updated_output_pulses

        # Store to frame
        frame[output_key] = output_map


    # Now add the cleaning step1 module to the tray, so its output is ready for the SRT module next
    tray.AddModule(
        _preprocess_pulse_map,
        output_key+"_prereco_pulse_cleaning_preprocessing",
        input_pulse_map_key=input_pulse_map_key,
        output_key=intermediate_output_key,
    )


    #
    # SRT cleaning
    #

    # Now user has generatd a new pulse map, run SRT cleaning on it if requested

    if apply_srt :

        from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService

        '''
        Run SRT cleaning to remove pulses due to decay noise that are not causally connected to the rest of the pulses

        Many settings, here are some examples from standard code for reference:

          - L2 standard (non-DC-specific) SRT (https://github.com/icecube/icetray/blob/23301bfc5a317eb1e2cc38e4332d361f5eb231e7/filterscripts/python/offlineL2/level2_all_filters.py#L59-L85):
            - srt_self_coincidence = True
            - srt_dust_layer_correction = False
            - srt_max_n_iterations = 3
            - srt_hlccore_seeding = True
            - srt_ic_r = 150.*I3Units.m
            - srt_ic_t = 1000.*I3Units.ns
            - (note: should actually set treat_string_36_as_deepcore=False, but don't think it actually matters if dc_dc_RTRadius/Time are not set)

          - DeepCore filter SRT (https://github.com/icecube/icetray/blob/066a3761ad66cd3d6cfb1f6caff3c80c41c99081/filterscripts/python/deepcorefilter.py#L34):
            - srt_self_coincidence = False
            - srt_dust_layer_correction = True
            - srt_max_n_iterations = -1 (e.g. infinite)
            - srt_hlccore_seeding = False
            - srt_ic_r = 150.*I3Units.m
            - srt_ic_t = 1000.*I3Units.ns
            - srt_dc_r = 75.*I3Units.m
            - srt_dc_t = 500.*I3Units.ns
        '''

        # Start defining SRT config object
        srt_config = dict(
            allowSelfCoincidence         = srt_self_coincidence,
            ic_ic_RTRadius               = srt_ic_r,
            ic_ic_RTTime                 = srt_ic_t,
            treat_string_36_as_deepcore  = True,
        )

        # Add DeepCore-specific args, if user specified them
        if srt_dc_r is not None :
            srt_config["dc_dc_RTRadius"] = srt_dc_r
        if srt_dc_t is not None :
            srt_config["dc_dc_RTTime"] = srt_dc_t

        # Add args related to the dust layer correction, if user requested it
        if srt_dust_layer_correction :
            srt_config["useDustlayerCorrection"] = True
            srt_config["dustlayerUpperZBoundary"] = 0*I3Units.m
            srt_config["dustlayerLowerZBoundary"] = -150*I3Units.m

        # Define SRT module args
        srt_kwargs = dict(
            MaxNIterations =  srt_max_n_iterations,
        )
        if srt_hlccore_seeding :
            srt_kwargs["SeedProcedure"] = "HLCCoreHits"
            srt_kwargs["NHitsThreshold"] = 2 # Value taken from L2 script
        else :
            srt_kwargs["SeedProcedure"] = "AllHLCHits" # 'NHitsThreshold' arg not relevent here

        # Write a pulse map, not a mask (since changing charge, time, etc)
        srt_module_name = "I3SeededRTCleaning_RecoPulse_Module"

        # Add the module
        tray.AddModule(
            srt_module_name, 
            output_key+'_SRT',
            InputHitSeriesMapName  = intermediate_output_key,
            OutputHitSeriesMapName = output_key,
            STConfigService        = I3DOMLinkSeededRTConfigurationService(**srt_config),
            Streams                = [icetray.I3Frame.Physics],
            **srt_kwargs
        )

        # Remove the intermediate pulse map
        tray.AddModule("Delete", intermediate_output_key+"_delete", Keys=[intermediate_output_key])

    else :

        # No SRT to do, just rename the intermediate object
        tray.AddModule("Rename", output_key+"_rename", Keys=[intermediate_output_key, output_key])


#
# Space-time relation between pulses
#

def calc_space_time_relation_between_pulses(frame, pulse_map, allow_self_coincidence=False) :
    '''
    Calcuate the distance/times between pulses

    These can be used to make causality cuts to remove e.g. noise pulses, as the seeded-radius-time algorithm does.
    '''

    #
    # Check inputs
    #

    # If pulse map name was given rather than map itself, load the map
    if isinstance(pulse_map, str) :
        pulse_map = frame[pulse_map]
        if hasattr(pulse_map, "apply") :
            pulse_map = pulse_map.apply(frame)

    # Check input type, should be either pulses or PEs
    assert isinstance(pulse_map, (dataclasses.I3RecoPulseSeriesMap, simclasses.I3MCPulseSeriesMap, simclasses.I3MCPESeriesMap) )

    #
    # Loop over pulses (or PEs)
    #

    # Init outputs
    min_dr_values, min_dt_values, min_v_values = collections.OrderedDict(), collections.OrderedDict(), collections.OrderedDict()
    min_dr_hlc_values, min_dt_hlc_values, min_v_hlc_values = collections.OrderedDict(), collections.OrderedDict(), collections.OrderedDict()

    # Loop over DOMs
    for om_key, pulses in pulse_map.items() :

        # Get DOM geometry
        om_geom = get_dom_geometry(frame=frame, om_key=om_key)

        # Init containers
        for x in [ min_dr_values, min_dt_values, min_v_values, min_dr_hlc_values, min_dt_hlc_values, min_v_hlc_values ] :
            x[om_key] = []

        # Loop over pulses
        for i, pulse in enumerate(pulses) :

            #
            # Compare to all other neightbours to find closest (in space or time) other pulse
            #

            # Init "closest pulse" variable. Also have a "HLC" specific version to somewhat mimic SRT seeding modes
            min_dr, min_dt, min_v = np.NaN, np.NaN, np.NaN
            min_dr_hlc, min_dt_hlc, min_v_hlc = np.NaN, np.NaN, np.NaN

            # Loop over all other DOMs
            for other_om_key, other_pulses in pulse_map.items() :

                # Skip same DOM, unless user allows self-coincidence
                if not allow_self_coincidence :
                    if om_key == other_om_key :
                        continue

                # Get the geometry for this other DOM
                other_om_geom = get_dom_geometry(frame=frame, om_key=other_om_key)

                # Now loop over the pulses on this other DOM
                for j, other_pulse in enumerate(other_pulses) :

                    # Skip same pulse
                    if (om_key == other_om_key) and (i == j) : 
                        continue

                    # Check if this other pulses is hlc
                    hlc, atwd = parse_pulse_flags(other_pulse)

                    #  Get distance and time between this pulse and our current pulse of interest
                    dr = (other_om_geom.position - om_geom.position).magnitude
                    dt = np.abs( other_pulse.time - pulse.time )

                    # Also get "speed" between pulses (causality proxy). Unit is "speed of light"
                    v = ( ( (dr/I3Units.meter) / (dt/I3Units.second) ) / speed_of_light ) if dt != 0. else np.inf # Speed

                    # Record the minimal values
                    if min_dr is np.NaN :
                        min_dr = dr
                        min_dt = dt
                        min_v = v
                    else :
                        min_dr = min(min_dr, dr)
                        min_dt = min(min_dt, dt)
                        min_v = min(min_v, v)

                    #TODO record whether min dr and dt are from the same pulse?

                    # Also compare only to HLCs (similar to SRT algorithm)
                    if hlc :
                        if min_dr_hlc is np.NaN :
                            min_dr_hlc = dr
                            min_dt_hlc = dt
                            min_v_hlc = v
                        else :
                            min_dr_hlc = min(min_dr_hlc, dr)
                            min_dt_hlc = min(min_dt_hlc, dt)
                            min_v_hlc = min(min_v_hlc, v)

            # print("min dr/t/v = %s, %s, %s" % (min_dr, min_dt, min_v))
            # print("min hlc dr/t/v = %s, %s, %s" % (min_dr_hlc, min_dt_hlc, min_v_hlc))

            # Record results for this pulse
            min_dr_values[om_key].append( min_dr )
            min_dt_values[om_key].append( min_dt )
            min_v_values[om_key].append( min_v )
            min_dr_hlc_values[om_key].append( min_dr_hlc )
            min_dt_hlc_values[om_key].append( min_dt_hlc )
            min_v_hlc_values[om_key].append( min_v_hlc )

        # Check have one value per pulse
        assert len(min_dr_values[om_key]) == len(pulses)
        assert len(min_dr_values[om_key]) == len(pulses)
        assert len(min_v_values[om_key]) == len(pulses)
        assert len(min_dr_hlc_values[om_key]) == len(pulses)
        assert len(min_dt_hlc_values[om_key]) == len(pulses)
        assert len(min_v_hlc_values[om_key]) == len(pulses)

    # Done
    return min_dr_values, min_dt_values, min_v_values, min_dr_hlc_values, min_dt_hlc_values, min_v_hlc_values

