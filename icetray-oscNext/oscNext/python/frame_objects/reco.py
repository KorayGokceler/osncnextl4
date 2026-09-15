'''
Functions related to computations we makes on
Retro Reco frame objects

Etienne Bourbeau, Kayla Leonard, Tom Stuttard
'''

import copy, numbers

import numpy as np


#
# Energy conversion
#

def convert_EM_to_hadronic_cascade_energy(E_em):
    '''
    Convert EM cascade energy to hadronic equivalent (e.g. the energy of hadronic cascade
    it would required to produce the same light as an EM cascade of the energy provided).
    '''

    # Redone hadronic factor
    E_o = 0.18791678
    m   = 0.16267529
    f0  = 0.30974123

    max_possible_Hd_cascade_energy = 10000.

    # Check inputs
    assert np.all(E_em >= 0.), "Negative EM cascade energy found"
    #TODO check EM energy is not out of max range for interpolation


    #
    # produce a calibration curve 
    # 
    # Since the conversion factor is easier to derive
    # when you start with the cascade energy, it makes
    # sense to obtain the curve of E_EM as a function 
    # of E_HD, and then interpolate that curve at the 
    # observed value of E_EM
    #
    HD_cascade_range = np.linspace(0.0,max_possible_Hd_cascade_energy,500001)
    
    # Total energy of the EM cascade as a fct of the total Hadronic cascade energy
    # (in the case where both yield the same amount of visible energy)
    #y = HD_cascade_range/E_o
    #EM_cascade_energy = y*E_o+E_o*(f0-1)*(y**(1-m))

    # The low-energy tail of the relationship must be stabilized 
    # around the threshold energy E_o (otherwise the ratio will blow
    # up and anyway, we enter an energy regime where the initial 
    # scaling becomes invalid (too little energy to produce the neutral
    # pions that constitute the cherenkov throughput of the cascade)
    #
    # This means that for energies below E_threshold, we fix the energy 
    # of the hadronic cascade. For some reason, that threshold has been
    # set to  2.71828183 GeV... We set it instead to 0.2 GeV, pending
    # further checks
    E_threshold = 0.2 #2.71828183

    y = (HD_cascade_range/E_o)*(HD_cascade_range>E_threshold) + (E_threshold/E_o)*(HD_cascade_range<=E_threshold)

    F_em = 1-y**(-m)

    EM_cascade_energy= HD_cascade_range*(F_em + (1-F_em)*f0)

    #
    # Now we have a calibration curve of EM cascade energy vs. HD cascade energy. 
    # 
    # All that is left is to interpolate the inverse of that curve to retrieve
    # the hadronic cascade energy for our reconstructed EM energy values:
    HD_casc_interpolated = np.interp(x=E_em,xp=EM_cascade_energy,fp=HD_cascade_range)

    assert np.all(HD_casc_interpolated <= max_possible_Hd_cascade_energy), "Hadronic cascade energy out or range"

    return HD_casc_interpolated


def convert_retro_reco_energy_to_neutrino_energy(em_cascade_energy, track_length) :
    '''
    Function to convert from the RetroReco fitted variables:
      a) EM cascade energy
      b) Track length
    To the underlying neutrino properties:
      a) Cascade energy (energy of all particles EXCEPT the outgoing muon)
      b) Outgoing muon energy
      c) Initial neutrino energy

    We use:
      - `convert_EM_to_hadronic_cascade_energy` to convert from EM to hadronic cascade energy
      - `GMS_LEN2EN` to convert track length to energy
      - Simple multiplicative fudge factors to correct the track length and cascade energy to 
        best match the neutrino properties, based on the oscNext MC (weighted to Honda flux 
        and nufit 2.0)

    We see good agreement in total energy for nue/mu CC, and also for nutau CC and NC events 
    but with an energy bias that matches the expectation due to the missing energy from final 
    state neutrinos (~25% missing energyfor nutau CC, 50% for NC).

    Agreement is worse for:
      - Very low energy (<5 GeV), where there seems to be a floor in reco energy
      - High energy (>100 GeV), where we seem to underestimate energy, although stats are bad 
        here so hard to compute percentiles.

    We also get good agreement for the track <-> muon length.

    Good data-MC agreement is observed in all cases.
    '''

    from retro.i3processing.retro_recos_to_i3files import GMS_LEN2EN
    # from retro.utils.cascade_energy_conversion import em2hadr

    # Convert EM to hadronic energy
    # cascade_hadronic_energy = em2hadr(em_cascade_energy) # This is a function provided by RetroReco (possible copied from pegleg) giving similar but not identical results to `convert_EM_to_hadronic_cascade_energy`) 
    cascade_hadronic_energy = convert_EM_to_hadronic_cascade_energy(em_cascade_energy)
    
    # Apply a fudge factor for overall cascade energy
    cascade_energy = 1.7 * cascade_hadronic_energy

    # Apply a fudge factor to the length
    track_length = 1.45 * track_length

    # Recompute track energy from fudged length, using GMS tables
    track_energy = GMS_LEN2EN(track_length) #TODO Think this is for water, not ice

    # Combine into a total energy
    total_energy = cascade_energy + track_energy

    return cascade_energy, track_energy, total_energy, track_length


TRACK_M_PER_GEV = 15 / 3.3 # This is copied from the function 'const_en2len' in RetroReco (for use cases where that package is not installed)

def convert_track_energy_to_length(energy_GeV) :
    '''
    Convert track energy to track length, under the assumption of a constant energy loss vs length for a muon travelling in ice
    '''
    return energy_GeV * TRACK_M_PER_GEV

def convert_track_length_to_energy(length_m) :
    '''
    Inverse of convert_track_energy_to_length
    '''
    return length_m / TRACK_M_PER_GEV


#
# Event display
#

def create_final_level_reco_I3Particle(frame) :
    '''
    Create an I3Particle containing the final level reconsturcted parameters.
    Make sure it is compatible with steamshovel for event displays.
    '''

    #TODO Directly add this to the L7 files

    from icecube import dataclasses
    from icecube.oscNext.selection.oscNext_L7 import L7_FINAL_RECO_X_KEY, L7_FINAL_RECO_Y_KEY, L7_FINAL_RECO_Z_KEY, L7_FINAL_RECO_TIME_KEY, L7_FINAL_RECO_ZENITH_KEY, L7_FINAL_RECO_AZIMUTH_KEY, L7_FINAL_RECO_TRACK_LENGTH_KEY, L7_FINAL_RECO_TOTAL_ENERGY_KEY

    particle = dataclasses.I3Particle()

    particle.pos = dataclasses.I3Position(frame[L7_FINAL_RECO_X_KEY].value, frame[L7_FINAL_RECO_Y_KEY].value, frame[L7_FINAL_RECO_Z_KEY].value)
    particle.dir = dataclasses.I3Direction(frame[L7_FINAL_RECO_ZENITH_KEY].value, frame[L7_FINAL_RECO_AZIMUTH_KEY].value) 
    particle.time = frame[L7_FINAL_RECO_TIME_KEY].value
    particle.energy = frame[L7_FINAL_RECO_TOTAL_ENERGY_KEY].value
    particle.length = frame[L7_FINAL_RECO_TRACK_LENGTH_KEY].value

    # particle.type = dataclasses.I3Particle
    particle.shape = dataclasses.I3Particle.ParticleShape.StartingTrack
    particle.location_type = dataclasses.I3Particle.LocationType.InIce
    particle.fit_status = dataclasses.I3Particle.FitStatus.OK

    return particle



#
# Track energy losses
#

def calc_track_light_profile(
    frame, 
    pulse_map_name,
    particle_key,
    max_distance_backward=100., # [m] 
    max_distance_forward=500., # [m] 
    num_segments=120,
    output_key=None,
    start_margin=None,
    stop_margin=None,
) :
    '''
    Compute the number of hits in a number of cylindrical semgments along a track

    Also compute dN/dx along the track
    '''

    from icecube import dataclasses


    #
    # Prepare
    #

    # Defaults
    if output_key is None :
        output_key = "TrackLightProfile"

    # Get the geometry
    omgeo = frame[str("I3Geometry")].omgeo

    # Track segment definition
    #TODO guarantee a segment at 0
    track_boundaries_1d = np.linspace( -max_distance_backward, max_distance_forward, num_segments+1 )  # Distances from the reco vertex along the reco trajectory
    track_centers_1d = 0.5 * (track_boundaries_1d[1:] + track_boundaries_1d[:-1])

    # Get pulses or similar opbject (handle any mask)
    pulse_map = frame[pulse_map_name]
    if hasattr(pulse_map, "apply") :
        pulse_map = pulse_map.apply(frame)

    # Get the particle
    assert particle_key in frame, "Could not find particle '%s' in the frame" % particle_key
    particle = frame[particle_key]

    # Get track information
    track_start = particle.pos
    track_direction = particle.dir
    track_length = particle.length
    track_stop = track_start + ( track_direction * track_length )


    #
    # Process
    #

    # This code makes a series of cylinder segments along the track, and counts hits in each

    #TODO vectorize/jit?

    vertex = np.array([track_start.x, track_start.y, track_start.z])
    dir_vec = np.array([track_direction.x, track_direction.y, track_direction.z])
    # dir_vec = -np.array(
    #     [
    #         np.sin(theta) * np.cos(phi),
    #         np.sin(theta) * np.sin(phi),
    #         np.cos(theta),
    #     ]
    # )

    track_boundaries_3d = (
        track_boundaries_1d[:, None] * dir_vec[None, :] + vertex
    )

    a_plane, b_plane, c_plane = dir_vec
    d_plane = -(
        a_plane * track_boundaries_3d[:, 0]
        + b_plane * track_boundaries_3d[:, 1]
        + c_plane * track_boundaries_3d[:, 2]
    )

    n_photons_in_segments = np.zeros_like(track_boundaries_1d[:-1])

    # Loop over OMs
    for om_key, pulses in pulse_map.items():

        # Get position of hit OM
        if isinstance(om_key, dataclasses.ModuleKey) :
            om_key = icetray.OMKey(om_key.string, om_key.om)
        pos = omgeo[om_key].position

        # Loop over pulses on this OM
        for ip, p in enumerate(pulses):

            pulse_coord = np.array([pos.x, pos.y, pos.z])

            # Calc disatcen to each boundary
            dist_to_boundaries = np.abs(
                np.dot(
                    np.array(pulse_coord),
                    np.array([a_plane, b_plane, c_plane]),
                )
                + d_plane
            ) / np.sqrt(a_plane**2 + b_plane**2 + c_plane**2)

            # Check if pulse is within a segment
            within_boundaries = np.where( dist_to_boundaries < np.diff(track_boundaries_1d)[0] )[0]
            if len(within_boundaries) == 2:

                # Determine which segment
                starting_bin = within_boundaries[0]

                # Get num photons (proxy) from charge
                if hasattr(p, "charge") :
                    num_photons = np.round(p.charge, 0)
                    if num_photons == 0:
                        num_photons = 1
                elif hasattr(p, "npe") :
                    num_photons = p.npe
                elif hasattr(p, "wavelength") : # e.g. a photon
                    num_photons = 1.

                # Increment counter
                n_photons_in_segments[starting_bin] += num_photons


    #
    # Calculate dN/dx
    #

    # This code calculates the average number of hits per unit length (dN/dx), a proxy for dE/dx.
    # Only compute this using segments within the length of the track, possibly with some margin 
    # to help mitigate uncertainty in the start/end point reco, and also avoid light from the breakup 
    # of the nucleus at the start.

    # Get start/end of the track in local coords, possibly with a margin at either end
    track_start_local = 0.
    track_stop_local = track_length
    if start_margin is not None :
        assert (start_margin > 0.) and (start_margin < 0.5)
        track_start_local += ( track_length * start_margin )
    if stop_margin is not None :
        assert (stop_margin > 0.) and (stop_margin < 0.5)
        track_stop_local -= ( track_length * stop_margin )

    # Get segments within the length
    within_track_mask = ( track_boundaries_1d >= track_start_local ) & ( track_boundaries_1d <= track_stop_local )

    # Check found match (won't if track is shorter than a single segment)
    dN_dx = np.NaN
    if np.sum(within_track_mask == True) >= 2 :

        # Also get the mask for the segment centers, rather than edges
        # This inmvovles dropping the last element and making the last True element False
        centers_mask = within_track_mask[:-1].copy()
        centers_mask[ np.where(centers_mask==True)[0][-1] ] = False

        # Compute dN / dx within the selected track region
        total_num_photons = np.sum( n_photons_in_segments[centers_mask] )
        total_length = track_boundaries_1d[within_track_mask][-1] - track_boundaries_1d[within_track_mask][0]
        dN_dx = float(total_num_photons) / total_length


    #
    # Store in frame
    #

    # Store the segment data
    segment_output = dataclasses.I3MapStringVectorDouble()

    segment_output[str("segment_boundaries")] = track_boundaries_1d #TODO store once only (S frame) ?
    segment_output[str("segment_centers")] = track_centers_1d #TODO store once only (S frame) ?
    segment_output[str("segment_num_photons")] = n_photons_in_segments
    segment_output[str("within_track_mask")] = within_track_mask.astype(float)
    frame[str(output_key+"_Segments")] = segment_output

    # Also store the input track information
    frame[str(output_key+"_TrackStart")] = track_start
    frame[str(output_key+"_TrackStop")] = track_stop
    frame[str(output_key+"_TrackDir")] = track_direction
    frame[str(output_key+"_TrackLength")] = dataclasses.I3Double(track_length)
    frame[str(output_key+"_dN_dx")] = dataclasses.I3Double(dN_dx)


    return True



#
# Pulse pre-processing
#


def reco_precleaned_dom_stats(frame, input_key, output_key, late_t_threshold=False) :
    '''
    Dedicated pulse pre-cleaning for use prior to reconstruction.
    Designed to mitigate known data-MC issues in pulse data and best approximate the underlying PE hits.

    Specifically:
      - Removes pulses that arrive significantly after the first pulse on a given DOM (to remove e.g. after-pulsing, noise, unmodelled coincident events, ...)
      - Quantise time to units of 1 ns - addresses issue that MC only ever has SuperDST pulses (which are only at 1 ns fidelity), whereas in real data there are often re-extracted waveforms with <ns precision
      - Use "per DOM stats" rather than pulse-level data:
        - First pulse time only (good proxy for first MCPE time, robust to wavedeform issues)
        - Sum charge on DOM
      - Quantise charge to "num PEs" analog
    '''



    print("reco_precleaned_dom_stats...")

    from icecube import dataclasses, icetray
    from icecube.oscNext.frame_objects.pulses import parse_pulse_flags

    # Grab input pules 
    input_pulse_map = frame[input_key].apply(frame)

    # Init output pulses
    assert output_key not in frame, "'%s' already exists in frame" % output_key
    output_pulse_map = dataclasses.I3RecoPulseSeriesMap()

    #TODO store rejected pulses??


    #
    # Loop over DOMs
    #

    for om_key, input_pulses in input_pulse_map.items() :

        # Skip empty
        if len(input_pulses) == 0 :
            continue


        #
        # Late t pulse rejection
        #

        #TODO also cut early pulses?

        # Pulses >XXX ns after the trigger are probably noise, coincidences, etc, and have bad data-MC agreement. 
        # Find that they can pull the reco (weird long tracks get added to try and account for them in Retro's LLH), so remove them...

        if late_t_threshold not in [None, False] :

            assert isinstance(late_t_threshold, numbers.Number)

            output_pulses = []

            # Get trigger time
            trigger_time = None
            trig_id = 1011 # DeepCore #TODO steerable
            trig_hierarchy = frame['I3TriggerHierarchy']
            for trig in trig_hierarchy:
                if trig.key.config_id == trig_id: 
                    trigger_time = trig.time
            assert trigger_time is not None

            # Get time for every pulse w.r.t. trigger and cut on it
            for pulse in input_pulses :
                t = pulse.time -  trigger_time
                # print(t)
                # print(trigger_time, pulse.time, t, late_t_threshold)
                if t < late_t_threshold :
                    output_pulses.append(pulse)
                # else :
                #     print("cut")

            else :
                output_pulses = input_pulses


        #
        # Late dt0 pulse rejection
        #

        # Pulses >1000 ns after the first pulse on a DOM are virtually always from noise, coincident events, afterpulsing, etc, not the signal
        # Remove them...

        output_pulses = []

        late_dt0_threshold = 1000. #TODO try 300 ns to avoid ATWD->FADC transition?? Doing this though would really require ALL SLC hits to be dropped to really remove the issue...

        # Need at least 2 pulses for a cut on dt0 
        if len(input_pulses) > 1 :

            # Calc dt0 for every pulse and cut on it
            t0 = input_pulses[0].time
            for pulse in input_pulses :
                dt0 = pulse.time - t0
                if dt0 < late_dt0_threshold :
                    output_pulses.append(pulse)

        else :
            output_pulses = input_pulses


        #
        # Pulse merging
        # 

        # In pass2 data, wavedeform sometimes splits pulses in data but not MC
        # Merge to avoid this


        #TODO



        #
        # Quantise time
        #

        # SuperDST pulses have times with an integer number of ns, whereas some re-extracted waveforms have finer grained times
        # However, the re-extraction never seems to happen in MC, not yet sure why by rounding times here to avoid possible bias

        #TODO don't apply this to the PE tables in retros

        new_output_pulses = []

        for input_pulse in output_pulses :
            output_pulse = copy.deepcopy(input_pulse)
            output_pulse.time = float(round(input_pulse.time))
            new_output_pulses.append(output_pulse)

        output_pulses = new_output_pulses


        #
        # Reject small pulses
        #

        #TODO? Could remove small pulses due to:
        # - poor data-MC near discriminatorm and 160 ns bump
        # - poor data-MC near discriminator threshold in SPE distribution


        #
        # Get per DOM stats
        #

        # (a) Only use first pulse time (possibly has already undergone merging)
        # (b) Take sum of remaining charge, and round to integer number of PEs
        # (c) Combine pulse flags (&)

        # Get time of first pulse
        first_time = output_pulses[0].time

        # We want to use charge as  proxy for "num PEs"
        # The mean of the SPE distribution is not 1 (the mean of the gaussian peak is calibrated at one, but the mean of the overall distribution is <1 due to the low charge exponental component)
        # Therefore need to correct the chrge by the mean of the PSE distribution
        # dom_mean_charge = dataclasses.mean_spe_charge(frame['I3DetectorStatus'].dom_status[om_key], frame['I3Calibration'].dom_cal[om_key]) #TODO This just returns a hard-coded value of 0.86, so doesn't take into account (a) indivudual DOM SPE fits and (b) the change in SOE template shape in more recent fits
        dom_mean_charge = 0.82 # Since `dataclasses.mean_spe_charge` is out-of-date, for now have tuned an overall correction for the current oscNext GCD file. #TODO In future, want to get indidivual means for each DOM, for bot ATWD and FADC 

        # Loop over pulses to:
        #  - Sum the charge (applying correction for mean of SP distribution)
        #  - Make a combined pulse flags object
        charge_sum = 0.
        dom_hlc, dom_slc, dom_atwd, dom_fadc = False, False, False, False
        for pulse in output_pulses :
            # Get flags
            hlc, atwd, fadc = parse_pulse_flags(pulse)
            # Sum charge
            # charge_sum += pulse.charge / ( hlc_gaus_mean_corrected if hlc else slc_gaus_mean_corrected )
            charge_sum += pulse.charge / dom_mean_charge

            # Update flags
            if hlc :
                dom_hlc = True # Mark as True if there are ANY HLC pulses
            else :
                dom_slc = True # Mark as True if there are ANY HLSLCC pulses
            if atwd :
                dom_atwd = True # Mark as True if there are ANY ATWD pulses
            if fadc :
                dom_fadc = True # Mark as True if there are ANY FADC pulses

        # Round the charge sum to represent PEs (there can only be integer numbers of PEs)
        charge_sum = max( float(round(charge_sum)), 1.)

        # Compile flags
        flags = 0
        if dom_hlc :
            flags = flags | dataclasses.I3RecoPulse.PulseFlags.LC # This means ">=1 HLC on DOM"
        if dom_atwd :
            flags = flags | dataclasses.I3RecoPulse.PulseFlags.ATWD # This means at least one pulse uses the ATWD
        if dom_fadc :
            flags = flags | dataclasses.I3RecoPulse.PulseFlags.FADC # This means at least one pulse uses the FADC
        if dom_slc :
            flags = flags | 8  # This means ">=1 SLC on DOM". There is no "SLC" flag, so abusing a free bit to make one

        # Create new "pulse" object, even though it is a "per DOM" statistic
        dom_stats = dataclasses.I3RecoPulse()
        dom_stats.time = first_time
        dom_stats.charge = charge_sum
        dom_stats.flags = flags

        # Write to the output map
        output_pulse_map[om_key] = dataclasses.I3RecoPulseSeries([dom_stats])

    # Write to frame
    frame[output_key] = output_pulse_map


#
# GraphNeT
#

def calc_sigma_from_kappa(kappa) :
    '''
    Convert kappa to sigma: kappa ~ 1/sigma^2)    #TODO reference
    '''
    return np.sqrt( 1. / kappa )

