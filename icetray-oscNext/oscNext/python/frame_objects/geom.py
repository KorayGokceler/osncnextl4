'''
Geometry-related variables

Tom Stuttard
'''

import numpy as np
import collections


#
# Coord system
#

# Define depth of coord sys origin
# https://wiki.icecube.wisc.edu/index.php/Coordinate_system#Definition_of_.22depth.22_in_IceCube
I3_COORD_SYS_ORIGIN_DEPTH_m = 1948.07


def get_depth_from_z(z) :
    '''
    Convert a vertical coordinate z (in the IceCube coord system) to a depth below the ice surface
    '''
    return I3_COORD_SYS_ORIGIN_DEPTH_m - z


def get_z_from_depth(depth) :
    '''
    Inverse of `get_depth_from_z`
    '''
    return I3_COORD_SYS_ORIGIN_DEPTH_m - depth


#
# Geometry definitions
#

# Define some reference string positions
STRING_36_XY_m = (46.29, -34.88) # Central in DeepCore
STRING_88_XY_m = (47.3, -57.0) # Central in DeepCore (numbering correct for V55 geom, could change in future iterations)

# Fiducial volume definitions
# Note that there are multiple definitions ued by different algorithms, and in some cases it is geometric, in other cases as list of fiducial vs veto OMs is used
ICECUBE_FIDUCIAL_RADIUS_m = 600.
ICECUBE_FIDUCIAL_ORIGIN_m = (0., 0., np.NaN) #TODO what z?
ICECUBE_FIDUCIAL_DEPTH_m = (2460, 1450)
ICECUBE_FIDUCIAL_Z_m = (get_z_from_depth(ICECUBE_FIDUCIAL_DEPTH_m[0]), get_z_from_depth(ICECUBE_FIDUCIAL_DEPTH_m[1]))

DEEPCORE_FIDUCIAL_RADIUS_m = 150.
DEEPCORE_FIDUCIAL_ORIGIN_m = (STRING_36_XY_m[0], STRING_36_XY_m[1], -330.) #TODO check z
DEEPCORE_FIDUCIAL_DEPTH_m = (ICECUBE_FIDUCIAL_DEPTH_m[0], 2100) # Lower edge matches IceCube
DEEPCORE_FIDUCIAL_Z_m = (get_z_from_depth(DEEPCORE_FIDUCIAL_DEPTH_m[0]), get_z_from_depth(DEEPCORE_FIDUCIAL_DEPTH_m[1]))

# Natural geometry
DUST_LAYER_TOP_DEPTH_m = 1950.
DUST_LAYER_BOTTOM_DEPTH_m = 2100.
DUST_LAYER_TOP_Z = get_z_from_depth(DUST_LAYER_TOP_DEPTH_m)
DUST_LAYER_BOTTOM_Z = get_z_from_depth(DUST_LAYER_BOTTOM_DEPTH_m)
BEDROCK_Z = -830.0

def get_fiducial_volume(radius, z_min, z_max) :
    ''' Get fiducial volume [m^3] given some cylinder definition '''
    return np.pi * np.square(radius) * (z_max - z_min)

# Define fiducial volumes
ICECUBE_FIDUCIAL_VOLUME_m3 = get_fiducial_volume(radius=ICECUBE_FIDUCIAL_RADIUS_m, z_min=np.min(ICECUBE_FIDUCIAL_Z_m), z_max=np.max(ICECUBE_FIDUCIAL_Z_m))
DEEPCORE_FIDUCIAL_VOLUME_m3 = get_fiducial_volume(radius=DEEPCORE_FIDUCIAL_RADIUS_m, z_min=np.min(DEEPCORE_FIDUCIAL_Z_m), z_max=np.max(DEEPCORE_FIDUCIAL_Z_m))

# Define the strings in the fiducial volume   #TODO Get this dynamically from the same function used at lower levels
FIDUCIAL_STRINGS = [
    36, # Central IceCube string
    26, 27, 37, 46, 45, 35, # IceCube strings on the outer permineter of DeepCore
    79, 80, 81, 82, 83, 84, 85, 86, # DeepCore strings
]

# Define anisotropy in IceCube ice
ANISOTROPY_AXIS_deg = 130. # See Fig 7 in https://tc.copernicus.org/preprints/tc-2022-174/tc-2022-174.pdf


#
# Basic geometric variables
#

def calc_rho_36(x,y) :
    '''
    Radial distance from string 36 (approximately central within DeepCore)
    '''
    return np.sqrt( (x-46.29) ** 2 + (y+34.88) ** 2 )


def calc_stopping_point(vertex, direction, length) :
    '''
    Determine the particle stopping point in the detector
    ''' 

    if length in [np.NaN, None] :
        return np.NaN
    else :
        return vertex + (direction * length)


def calc_particle_stopping_point(particle) :
    return calc_stopping_point(vertex=particle.pos, direction=particle.dir, length=particle.length)


def get_deepcore_containment(x, y, z) :

    # Get radial point
    rho_36 = calc_rho_36(x=x, y=y)

    # Compute containment flag
    containment_bool = rho_36 < DEEPCORE_FIDUCIAL_RADIUS_m # Radial
    containment_bool = containment_bool and (z <= np.max(DEEPCORE_FIDUCIAL_Z_m) ) and (z >= np.min(DEEPCORE_FIDUCIAL_Z_m) ) # Vertical

    return containment_bool


def get_icecube_containment(x, y, z) :

    # Get radial point
    rho_36 = calc_rho_36(x=x, y=y)

    # Compute containment flag
    containment_bool = rho_36 < ICECUBE_FIDUCIAL_RADIUS_m # Radial
    containment_bool = containment_bool and (z <= np.max(ICECUBE_FIDUCIAL_Z_m) ) and (z >= np.min(ICECUBE_FIDUCIAL_Z_m) ) # Vertical

    return containment_bool


def angular_comparison(dir1, dir2) :
    '''
    Angular difference between two direction vectors
    '''

    #TODO the L5 linefit vs corridor cut angular comparison should be updated to use this

    from icecube.dataclasses import I3MapStringDouble, I3Direction

    # Check inputs
    assert isinstance(dir1, I3Direction)
    assert isinstance(dir2, I3Direction)

    # Output container
    outdict = I3MapStringDouble()

    # Angular difference (dot product)
    outdict['angle_diff']       = np.arccos( dir1 * dir2 )

    # Zenith component diff
    outdict['zenith_diff']      = dir1.zenith - dir2.zenith
    outdict['abs_zenith_diff']  = abs(outdict['zenith_diff'])
    
    # Azimuth component diff (handle wrapping)
    outdict['azimuth_diff']     = dir1.azimuth - dir2.azimuth
    if outdict['azimuth_diff'] > np.pi : 
        outdict['azimuth_diff'] = outdict['azimuth_diff']  - np.pi
    elif outdict['azimuth_diff'] < - np.pi : 
        outdict['azimuth_diff'] = outdict['azimuth_diff']  + np.pi
    outdict['abs_azimuth_diff'] = abs(outdict['azimuth_diff'])
 
    return outdict


def vertex_to_om_and_string_distance(om_geom, vertex) :
    '''
    Get distance from vertex to OM/string
    '''

    from icecube.dataclasses import I3Position

    vertex_om_distance = (om_geom.position - vertex ).magnitude
    vertex_string_distance = ( I3Position(om_geom.position.x, om_geom.position.y, vertex.z) - vertex ).magnitude

    return vertex_om_distance, vertex_string_distance


def compute_event_vertex_geometry_params(
    frame, 
    vertex,
    pulse_series_name,
    output_key,
):
    ''' 
    Compute several variable that try to capture the
    geometry of the event relative to the DOMs.

    It is hoped that these variables are sensitive to ice/detector systematics,
    and thus can be used as control variables to verify these are modelled well
    '''


    from icecube.icetray import OMKey
    from icecube.dataclasses import I3MapStringDouble, I3Particle
    import numpy as np

    # Check frame contents
    assert not frame.Has(output_key), "Output variable already exists : %s" % output_key 
    assert frame.Has('I3Geometry'), "Could not find geometry"
    assert frame.Has(pulse_series_name), "Could not find the pulse series : %s" % pulse_series_name

    # Vertex could either be an objectm, or a key to find an object in the frame
    if isinstance(vertex, str) :
        assert frame.Has(vertex), "Could not find the vertex object : %s" % vertex
        vertex = frame[vertex]

    # Get vertex (either the object is a position, or is a particle with a position)
    if isinstance(vertex, I3Particle) :
        vertex = vertex.pos

    # Get pulse series
    pulse_series_map = frame[pulse_series_name]
    if hasattr(pulse_series_map, "apply") :
        pulse_series_map = pulse_series_map.apply(frame)


    # 
    # Init variables
    #

    closest_string_distance = np.inf
    closest_hit_string_distance = np.inf

    closest_om_distance = np.inf
    closest_hit_om_distance = np.inf
    
    charge_per_string = {}
    charge_per_om = {}

    distance_to_fiducial_strings = collections.OrderedDict()
    for s in FIDUCIAL_STRINGS :
        distance_to_fiducial_strings[s] = None

    #TODO finite track DCA
    #TODO closest above OM? or something using direction between particle and PMT face?
    #TODO depth?
    #TODO Something sensitive to angle, and thus hole ice
    #TODO weighted sum of distance to all OMs?


    #
    # Calculate variables
    #

    # Get OM geometry
    om_geometry_map = frame['I3Geometry'].omgeo

    # Loop over OMs
    for om_key, om_geom in om_geometry_map.items() :

        # Only use standard IceCube DOMs (not IceTop, special devices, etc)
        if om_geom.omtype != om_geom.IceCube :
            continue

        # Only use DOMs not on the bad DOM list
        #TODO

        # Check if this OM was hit
        om_was_hit = (om_key in pulse_series_map) and (len(pulse_series_map[om_key]) > 0)

        # Get distance from vertex to OM/string
        vertex_om_distance, vertex_string_distance = vertex_to_om_and_string_distance(om_geom=om_geom, vertex=vertex)

        # Record distance to closest string and closest OM for this event
        if vertex_string_distance < closest_string_distance :
            closest_string_distance = vertex_string_distance
        if vertex_om_distance < closest_om_distance :
            closest_om_distance = vertex_om_distance

        # Record distance to closest hit string and closest hit OM for this event
        if om_was_hit : 
            if vertex_string_distance < closest_hit_string_distance :
                closest_hit_string_distance = vertex_string_distance
            if vertex_om_distance < closest_hit_om_distance :
                closest_hit_om_distance = vertex_om_distance

        # Record distance to the fiducial strings
        if om_key.string in distance_to_fiducial_strings :
            if distance_to_fiducial_strings[om_key.string] is None :
                distance_to_fiducial_strings[om_key.string] = vertex_string_distance

        # Tally charge per string/OM
        if om_was_hit :

            om_charge = np.sum([ p.charge for p in pulse_series_map[om_key] ])

            if om_key.string in charge_per_string :
                charge_per_string[om_key.string] += om_charge
            else :
                charge_per_string[om_key.string] = om_charge

            assert om_key not in charge_per_om
            charge_per_om[om_key] = om_charge

    # Brighest OM/string
    brightest_string = max(charge_per_string, key=charge_per_string.get)
    _, brightest_string_distance = vertex_to_om_and_string_distance(om_geom=om_geometry_map[OMKey(brightest_string,1)], vertex=vertex)

    brightest_om = max(charge_per_om, key=charge_per_om.get)
    brightest_om_distance, _ = vertex_to_om_and_string_distance(om_geom=om_geometry_map[brightest_om], vertex=vertex)


    #
    # Checks
    #

    assert np.isfinite(closest_string_distance)
    assert np.isfinite(closest_hit_string_distance)
    assert np.isfinite(brightest_string_distance)
    assert np.isfinite(closest_om_distance)
    assert np.isfinite(closest_hit_om_distance)
    assert np.isfinite(brightest_om_distance)

    for s, d in distance_to_fiducial_strings.items() :
        assert d is not None


    # 
    # Write to frame
    #

    output_map = I3MapStringDouble()

    output_map["closest_string_distance"] = closest_string_distance
    output_map["closest_hit_string_distance"] = closest_hit_string_distance
    output_map["brightest_string_distance"] = brightest_string_distance

    output_map["closest_om_distance"] = closest_om_distance
    output_map["closest_hit_om_distance"] = closest_hit_om_distance
    output_map["brightest_om_distance"] = brightest_om_distance

    for s, d in distance_to_fiducial_strings.items() :
        output_map["distance_to_string_%i"%s] = d

    frame[output_key] = output_map


#
# Extracting DOM-based geometry information from GCD
#

def get_dom_geometry(frame, om_key) :
    '''
    Extract the geometry for this DOM
    '''

    from icecube import icetray, dataclasses

    # Convert from ModuleKey to OMKey if necessary
    if isinstance(om_key, dataclasses.ModuleKey) : 
        om_key = icetray.OMKey(om_key.string, om_key.om, 0) #TODO this assumes only a single PMT, so needs updating to support Upgrade
    assert isinstance(om_key, icetray.OMKey)

    # Grab geometry
    return frame[str("I3Geometry")].omgeo[om_key]
                


def is_dom_in_deepcore_fiducial(frame, om_key, use_dom_list=True) :
    '''
    Check if DOM is part of the fiducial volume of DeepCore

    Two different definitions:
        1) use_dom_list == True : Use the hardcoded list of fiducial DOMs (as used by the DeepCoreFilter)
        2) use_dom_list == False : Use geometry to decide, e.g. is a DOM within the fiducial volume
    '''

    #TODO does this exactly correspond to the definition of fiducial in the DeepCore filter?

    # Get the DOM positions
    dom_geom = get_dom_geometry(frame=frame, om_key=om_key)

    # Check if the DOM is in the fiducial region
    if use_dom_list :
        from icecube.DeepCore_Filter import DOMList
        dc_fid_list = list(DOMList.DeepCoreFiducialDOMsDict['IC86']) #TODO Upgrade compatible
        return ( om_key in dc_fid_list )
    else :
         # Geometric definition. Re-using the DeepCore containment function more generally used for checking vertices)
        return get_deepcore_containment(x=dom_geom.position.x, y=dom_geom.position.y, z=dom_geom.position.z)

