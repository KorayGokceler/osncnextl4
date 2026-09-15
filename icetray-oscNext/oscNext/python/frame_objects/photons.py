'''
Variables related to photons

Tom Stuttard, Etienne Bourbeau
'''

#TODO finish updating

import numpy as np

import numbers

from icecube import icetray, dataclasses

from scipy.constants import speed_of_light


#
# Photon speed
#

SPEED_OF_LIGHT = speed_of_light * icetray.I3Units.meter / icetray.I3Units.second
REFRACTIVE_INDEX_ICE = 1.33 # Note that in reality this is slightly depth-dependent
SPEED_OF_LIGHT_ICE = SPEED_OF_LIGHT / REFRACTIVE_INDEX_ICE

SUPERLUMINAL_THRESHOLD = SPEED_OF_LIGHT_ICE


def calc_photon_speed(vertex_pos, vertex_time, hit_pos, hit_time, speed_of_light_units=True) :
    '''
    Calc photon speed, given some vertex
    '''

    assert isinstance(vertex_pos, dataclasses.I3Position),'vertex_pos must be of type dataclasses.I3Position'
    assert isinstance(vertex_time, numbers.Number),'vertex_time must be a number'

    displacement = (vertex_pos - hit_pos).magnitude
    time_taken = hit_time - vertex_time

    if time_taken == 0 :
        speed = np.NaN
    else:
        speed = displacement / time_taken

    # Express value w.r.t. speed of light, if requested
    if speed_of_light_units :
        speed = ( speed / (icetray.I3Units.meter/icetray.I3Units.second) ) / speed_of_light


    return displacement, time_taken, speed


def calc_photon_speed_metrics(speed_values) :
    '''
    Compute some scalar photon speed metrics
    These are the kind of variables a classifier could use

    Inputs arguments are the following:

    speed_values: photon speeds

    superluminal_threshold: threshold speed we will consider to be superluminal

    trim_unphysical: ignore pulses that have unphysical speeds
    '''

    # Check inputs
    assert isinstance( speed_values, (collections.abc.Sequence, np.ndarray) )
    assert len(speed_values) > 0
    assert all([ isinstance(s, numbers.Number) for s in speed_values ])

    # Defaults
    if superluminal_threshold is None :
        superluminal_threshold = SPEED_OF_LIGHT_ICE

    # Mask off events with no speed calculated (e.g. when time taken was 0)
    speed_values_masked = speed_values[np.isfinite(speed_values)]

    # Look at whether photon speeds are physical (e.g. in range [0, c])
    unphysical_slow_mask = speed_values_masked < 0. # Too slow
    unphysical_fast_mask = speed_values_masked > SPEED_OF_LIGHT # Too fast
    unphysical_mask = unphysical_slow_mask | unphysical_fast_mask

    # Remove unphysical cases (such as noise hits that are not causally connected to the vertex)
    if trim_unphysical :
        speed_values_masked = speed_values_masked[~unphysical_mask]

    # Num/fraction hits faster than expected for vertex direct light
    superluminal_speed_values = speed_values_masked[speed_values_masked > superluminal_threshold]
    num_superluminal = len(superluminal_speed_values)
    num_total = len(speed_values_masked)
    fraction_superluminal = float(num_superluminal) / float(num_total) if num_total > 0 else np.NaN

    # Fraction of pulses with unphysical speeds (usefull to distinguish noise events)
    unphysical_fraction_slow = float(unphysical_slow_mask.sum()) / float(num_total) if num_total > 0 else np.NaN
    unphysical_fraction_fast = float(unphysical_fast_mask.sum()) / float(num_total) if num_total > 0 else np.NaN
    unphysical_fraction = float(unphysical_mask.sum()) / float(num_total) if num_total > 0 else np.NaN

    # Grab max some other simple metrics
    # Some of these can catpure the "spread" of speeds
    speed_max = np.max(speed_values_masked) if speed_values_masked.size > 0 else np.NaN # Physicaly cuts can give us zero photons
    speed_mean = np.mean(speed_values_masked) if speed_values_masked.size > 0 else np.NaN
    speed_median = np.median(speed_values_masked) if speed_values_masked.size > 0 else np.NaN
    speed_sigma = np.std(speed_values_masked) if speed_values_masked.size > 0 else np.NaN

    # Return the values
    return num_superluminal, fraction_superluminal, speed_max, speed_mean, speed_median, speed_sigma, unphysical_fraction_slow, unphysical_fraction_fast, unphysical_fraction


def photon_speed_metrics(
    frame,
    pulses, 
    vertex_particle,
    output_prefix=None,
) :
    '''
    Icetray module to compute variables related to the 
    apparent photon speeds from pulses

    Inputs arguments are the following:

    pulses: name of a valid pulse series

    vertex_particle: name of I3Particle frame object that has the vertex position and time information 

    output_perfix: a prefix for the output frame object names

    '''

    #
    # Check inputs
    #

    # Check pulses exist
    assert frame.Has(pulses), 'ERROR: pulse series {} not found'.format(pulses)

    # Handle the alternative vertex params vs I3Particle definitions
    if vertex_particle is None :
        assert (vertex_x is not None) and frame.Has(vertex_x)
        assert (vertex_y is not None) and frame.Has(vertex_y)
        assert (vertex_z is not None) and frame.Has(vertex_z)
        assert (vertex_time is not None) and frame.Has(vertex_time)
        vertex_pos = dataclasses.I3Position(frame[vertex_x].value, frame[vertex_y].value, frame[vertex_z].value)
        vertex_time = frame[vertex_time].value
    else :
        vertex_pos = vertex_particle.pos
        vertex_time = vertex_particle.time


    #
    # Loop over pulses
    #

    # Get OM geometry map
    om_geo = frame["I3Geometry"].omgeo

    # Prepare containers
    displacement = []
    time_taken = []
    speed = []

    # Loop over OMs
    for om, pulses in frame[pulses].apply(frame).items() :

        # Get OM position
        om_pos = (om_geo[om].position)

        # loop over pulses
        for pulse in pulses :

            # Compute "speed" of hit releative to vertex
            d, t, s = calc_photon_speed(vertex_pos=vertex_pos, vertex_time=vertex_time, hit_pos=om_pos, hit_time=pulse.time)

            # Fill containers
            displacement.append(d)
            time_taken.append(t)
            speed.append(s)

    # numpy-ify
    displacement = np.array(displacement)
    time_taken = np.array(time_taken)
    speed = np.array(speed)

    # Compute some derived quantities
    num_superluminal, fraction_superluminal, speed_max, speed_mean, speed_median, speed_sigma, unphysical_fraction_slow, unphysical_fraction_fast, unphysical_fraction = calc_photon_speed_metrics(speed, superluminal_threshold=superluminal_threshold, trim_unphysical=trim_unphysical)

    # Normalise to speed of light (vacuum)
    displacement /= SPEED_OF_LIGHT
    time_taken /= SPEED_OF_LIGHT
    speed /= SPEED_OF_LIGHT
    speed_max /= SPEED_OF_LIGHT
    speed_mean /= SPEED_OF_LIGHT
    speed_median /= SPEED_OF_LIGHT
    speed_sigma /= SPEED_OF_LIGHT


    #
    # Fill frame objects
    #

    if output_prefix is None :
        output_prefix = ""
    else :
        output_prefix += "_"

    # Put all the "1 per event" metrics into a map
    results = dataclasses.I3MapStringDouble()
    results["num_superluminal"] = num_superluminal
    results["fraction_superluminal"] = fraction_superluminal
    results["speed_max"] = speed_max
    results["speed_mean"] = speed_mean
    results["speed_median"] = speed_median
    results["speed_sigma"] = speed_sigma
    results["unphysical_fraction_slow"] = unphysical_fraction_slow
    results["unphysical_fraction_fast"] = unphysical_fraction_fast
    results["unphysical_fraction"] = unphysical_fraction
    frame[output_prefix+"PhotonSpeedMetrics"] = results

    # Also add the arrays
    frame[output_prefix+'PhotonDisplacement'] = dataclasses.I3VectorDouble(displacement)
    frame[output_prefix+'PhotonTimeTaken'] = dataclasses.I3VectorDouble(time_taken)
    frame[output_prefix+'PhotonSpeed'] = dataclasses.I3VectorDouble(speed)




#
# Photon direction
#


def calc_photon_direction(vertex_pos, hit_pos) :
    '''
    Calc photon direction vector, given some vertex
    '''

    assert isinstance(hit_pos, dataclasses.I3Position),'hit_pos must be of type dataclasses.I3Position'
    assert isinstance(vertex_pos, dataclasses.I3Position),'vertex_pos must be of type dataclasses.I3Position'

    travel_vector = hit_pos - vertex_pos
    return dataclasses.I3Direction(travel_vector.x, travel_vector.y, travel_vector.z)


def calc_photon_direction_metrics(direction_values) :
    '''
    Compute some scalar photon direction metrics
    '''

    #TODO take particle direction as input and calculate angle w.r.t. particle direction
    #TODO weight each photon by 1/displacement (e.g. reduce impact of photons that have likely scattered more)

    # Check inputs
    assert isinstance( direction_values, (collections.abc.Sequence, np.ndarray) )
    assert len(direction_values) > 0
    assert all([ isinstance(d, dataclasses.I3Direction) for d in direction_values ])

    # Get components
    direction_x_values = [ d.x for d in direction_values ]
    direction_y_values = [ d.y for d in direction_values ]
    direction_z_values = [ d.z for d in direction_values ]

    # Get mean amnd std of direction
    mean_direction = dataclasses.I3Direction( np.mean(direction_x_values), np.mean(direction_y_values), np.mean(direction_z_values) )
    std_direction_x, std_direction_y, std_direction_z = np.std(direction_x_values), np.std(direction_y_values), np.std(direction_z_values)

    # Get mean and std of "opening angle" relative to mean   #TODO also relative to reco/true particle direction
    angle_to_mean = [ d.angle(mean_direction) for d in direction_values ]
    mean_angle_to_mean = np.mean(angle_to_mean)
    std_angle_to_mean = np.mean(angle_to_mean)

    return mean_direction, (std_direction_x, std_direction_y, std_direction_z), mean_angle_to_mean, std_angle_to_mean
