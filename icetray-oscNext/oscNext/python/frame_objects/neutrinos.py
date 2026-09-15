'''
Tools for weighting neutrino simulations

Handles neutrino flux and oscillation calculations

Tom Stuttard, Michael Larson
'''

import numpy as np

from icecube import dataclasses
from icecube.icetray import I3Bool
import icecube.oscNext.frame_objects.simulation # Cannot use 'from'' due to circular dependence
from icecube.oscNext.frame_objects.weighting import WEIGHT_DICT_KEY, WEIGHT_KEY
from icecube.oscNext.frame_objects.geom import calc_stopping_point, calc_rho_36, get_icecube_containment, get_deepcore_containment
from icecube.oscNext.tools.misc import UpdateFrameObject

# Weight dict keys
NU_ONE_WEIGHT_KEY = "OneWeight"
NU_NORM_ONE_WEIGHT_KEY = "NormalizedOneWeight"
NU_NUM_EVENTS_KEY = "NEvents"
NU_FLUX_E_KEY = "flux_e"
NU_FLUX_MU_KEY = "flux_mu"
NU_GEN_RATIO_KEY = "gen_ratio"
NU_PROB_FROM_E_KEY = "prob_from_nue"
NU_PROB_FROM_MU_KEY = "prob_from_numu"
NU_NO_OSC_WEIGHT_KEY = "weight_no_osc"
NU_PRIMARY_NEUTRINO_ENERGY = "PrimaryNeutrinoEnergy"
NU_SINGLE_POWERLAW_STEM = "SinglePowerLawFlux"
NU_SINGLE_POWERLAW_FLUX_KEY = NU_SINGLE_POWERLAW_STEM + "_flux"
NU_SINGLE_POWERLAW_WEIGHT_KEY = NU_SINGLE_POWERLAW_STEM + "_weight"
NU_SINGLE_POWERLAW_NORM_KEY = NU_SINGLE_POWERLAW_STEM + "_norm"
NU_SINGLE_POWERLAW_INDEX_KEY = NU_SINGLE_POWERLAW_STEM + "_index"
STARTING_NEUTRINO_KEY = "MCDeepCoreStartingEvent"


#
# Oscillations
#

def calc_osc_prob_prob3(barger_prop, prop_height, e, cz, pdg, params) :
    '''
    Calculate oscillation probabilities using prob3
    '''
    NuE = 1 ; NuMu = 2 ; NuTau = 3
    kSquared = True
    kNuType = np.array(np.sign(pdg), dtype=np.int)
    out_nu = (  NuE*(np.abs(pdg) == 12) + NuMu*(np.abs(pdg) == 14) + NuTau*(np.abs(pdg) == 16) )
    osc_prob = np.zeros([len(e), 2])
    for i in range(len(e)):
        barger_prop.SetMNS(params['sin2_theta12'], params['sin2_theta13'], params['sin2_theta23'],
                           params['dm21'], params['dm32'], params['deltacp'],
                           float(e[i]), kSquared, int(kNuType[i]))
        barger_prop.DefinePath(float(cz[i]), prop_height)
        barger_prop.propagate(int(kNuType[i]))
        osc_prob[i,:] = [barger_prop.GetProb(NuE, int(out_nu[i])),
                         barger_prop.GetProb(NuMu, int(out_nu[i]))]
    return osc_prob


def calc_baseline_from_coszen(cz, r=6371., h=15., d=1.) :
    '''
    cz = cos(zenith) in radians, to be converted to path length in km
    r = Radius of Earth, in km
    h = Production height in atmosphere, in km
    d = Depth of detector, in km
    '''
    return -r*cz +  np.sqrt( (r*cz)**2 - r**2 + (r+h+d)**2 )


def calc_numu_survival_prob_simple(coszen, energy, theta, deltam2, prop_height, detector_depth) :
    '''
    Simple 2-flavour numu surival oscillation equation in vacuum
    Is a pretty good approximation of atmospheric neutrino oscillations
    '''

    # Convert coszen to baseline
    baseline = calc_baseline_from_coszen(cz=coszen, h=prop_height, d=detector_depth)

    # Calculate oscillations
    return 1. - ( np.square(  np.sin(2.*theta) ) *  np.square( np.sin( ( 1.27 * deltam2 * baseline ) / energy ) ) )



#
# Weighting 
#

class NeutrinoWeighter(object) :
    '''
    A class for adding weights to neutrino events that take into account flux and oscillations
    Can be used as an i3 module
    Note that the weight be be valid for the single i3 file used (e.g. have not normalised by total number of files) 
    '''

    #TODO Should pre-compute and spline osc probs

    def __init__(self, 
        is_nugen=False, # Indicate if generator is NuGen (otherwise GENIE assumed)
        gen_ratio=None, # The fraction of events that are nu (rather than nubar)
        prob3=False, # Use prob3 for the osc calculation (alternative is a simple 2 flavour calculation in vacuum)
        add_truth_to_frame=False, # Also store the truth particle into the frame
    ) :

        import os

        # Store args
        self.is_nugen = is_nugen
        self.gen_ratio = gen_ratio
        self.prob3 = prob3
        self.add_truth_to_frame = add_truth_to_frame

        # Try to get the flux weighting tools
        try :
            import nuflux
        except Exception as e :
            raise Exception("The required `nuflux` project is missing")

        # Try to get the oscillation probability calculator
        if self.prob3 :
            try :
                from icecube import prob3
            except Exception as e :
                raise Exception("The required `prob3` project is missing (https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/olivas/prob3). This has not been ported to the latest (GitHub) IceTray yet...")

        # Use default gen_ratio for this type if none provided
        #TODO Should store this in the data (I frame or weight dict)
        if self.gen_ratio is None :
            if self.is_nugen: 
                self.gen_ratio = 0.5
            else: 
                self.gen_ratio = 0.7

        # Check gen ratio
        assert (self.gen_ratio >= 0.) and (self.gen_ratio <= 1.), "Invalid `gen_ratio` value (must be in between 0 and 1) : %s" % self.gen_ratio


        #
        # Init flux service
        #

        #TODO redirect annoying output from these
        self.lowe_flux_service = nuflux.makeFlux("IPhonda2014_spl_solmin")
        self.highe_flux_service = nuflux.makeFlux("honda2006")
        self.highe_flux_service.knee_reweighting_model = 'gaisserH3a_elbert'


        #
        # Init oscillations
        #

        if self.prob3 :
            self.earth_model = os.path.expandvars('$I3_SRC/prob3/resources/oscillations/PREM_10layer.dat') # TODO 4 layer model for speed?
            self.barger_prop = prob3.BargerPropagator(self.earth_model, self.detector_depth) #TODO Use vacuum? Should be faster and good enough for most cases
            self.barger_prop.UseMassEigenstates(False)
            self.barger_prop.SetOneMassScaleMode(False)
            self.barger_prop.SetWarningSuppression(True)

        self.detector_depth = 2.
        self.prop_height = 20.

        self.osc_params = { # NuFit global fit values #TODO think this is v2.0, check
            "dm21": 7.49e-5,
            "dm31": 2.526e-3,
            "sin2_theta12": 0.308,
            "sin2_theta13": 0.02163,
            "sin2_theta23": 0.440,
            "deltacp": 289.*np.pi/180.,
        }

        self.osc_params["dm32"] = self.osc_params["dm31"] - self.osc_params["dm21"]

        #TODO Store osc params in frame


    def __call__(self, frame, overwrite=False) :

        from icecube.oscNext.frame_objects.simulation import get_primary_neutrino
        from icecube import dataclasses

        # Check the weight dict is there (should be, as genie-icetray writes it)
        assert WEIGHT_DICT_KEY in frame, "Could not find weight dict '%s' in the frame" % WEIGHT_DICT_KEY

        # Grab the weight dict
        with UpdateFrameObject(frame,WEIGHT_DICT_KEY) as weight_dict :

            # Bail out if a weight has already been written
            if overwrite == False :
                if WEIGHT_KEY in weight_dict :
                    return


            #
            # Store weighting parameters
            #

            weight_dict["detector_depth"] = self.detector_depth
            weight_dict["prop_height"] = self.prop_height
            for k,v in self.osc_params.items() :
                weight_dict[k] = v


            #
            # Get neutrino information
            #

            true_neutrino = get_primary_neutrino(frame)

            true_energy = weight_dict[NU_PRIMARY_NEUTRINO_ENERGY]
            true_zenith = true_neutrino.dir.zenith
            true_azimuth = true_neutrino.dir.azimuth
            true_pdg_code = true_neutrino.pdg_encoding

            #TODO grab from weight dict
            gen_ratio = self.gen_ratio

            # Handle nu vs nubar 
            nue, numu = dataclasses.I3Particle.ParticleType.NuE, dataclasses.I3Particle.ParticleType.NuMu
            if true_neutrino.pdg_encoding < 0:
                nue, numu = dataclasses.I3Particle.ParticleType.NuEBar, dataclasses.I3Particle.ParticleType.NuMuBar
                gen_ratio = 1. - gen_ratio

            # Store the gen ratio in the frame
            weight_dict[NU_GEN_RATIO_KEY] = gen_ratio

            # Also add the primary neutrino to the frame if does not already exist
            if self.add_truth_to_frame :
                if icecube.oscNext.frame_objects.simulation.IN_ICE_PRIMARY_KEY not in frame :
                    frame[icecube.oscNext.frame_objects.simulation.IN_ICE_PRIMARY_KEY] = true_neutrino


            #
            # Get flux
            #

            if true_neutrino.energy < 1e5 : 
                flux_service = self.lowe_flux_service
            else: 
                flux_service = self.highe_flux_service

            weight_dict[NU_FLUX_E_KEY] = flux_service.getFlux(nue, true_energy, np.cos(true_zenith)) # Not providing azimuth (e.g. using azimuthally averaged flux)
            weight_dict[NU_FLUX_MU_KEY] = flux_service.getFlux(numu, true_energy, np.cos(true_zenith)) # Not providing azimuth (e.g. using azimuthally averaged flux)


            #
            # Get Oscillation probabilities
            #

            if self.prob3 :

                # Use prob3 to calculate the oscillation probabilities
                # This will return "nue -> this nu" and "numu -> this nu"
                osc_probs = calc_osc_prob_prob3(
                    barger_prop=self.barger_prop,
                    prop_height=self.prop_height,
                    e=[true_energy,], 
                    cz=[np.cos(true_zenith),],
                    pdg=[true_pdg_code,], 
                    params=self.osc_params,
                )

                # Get prob this neutrino came from either nue/mu (assuming negligible unoscillated nutau flux)
                prob_from_nue  = osc_probs[0,0]
                prob_from_numu = osc_probs[0,1]

            else :

                # Simple analytic calculation
                # 2-flavour assumption (neglects nue oscillations), vacuum
                # This is a good approximation of atmospheric neutrino oscillations in DeepCore energy range
                prob_mu_mu = calc_numu_survival_prob_simple(
                    coszen=np.cos(true_zenith),
                    energy=true_energy,
                    theta=np.arcsin(np.sqrt(self.osc_params["sin2_theta23"])),
                    deltam2=self.osc_params["dm31"],
                    prop_height=self.prop_height,
                    detector_depth=self.detector_depth,
                )

                # Get prob this neutrino came from either nue/mu (assuming negligible unoscillated nutau flux)
                if np.abs(true_pdg_code) == 12 :
                    prob_from_nue = 1.
                    prob_from_numu = 0.

                elif np.abs(true_pdg_code) == 14 :
                    prob_from_nue = 0.
                    prob_from_numu = prob_mu_mu

                elif np.abs(true_pdg_code) == 16 :
                    prob_from_nue = 0.
                    prob_from_numu = 1. - prob_mu_mu # All disappearing numu go to nutau


            # Stash in weight dict
            weight_dict[NU_PROB_FROM_E_KEY]  = prob_from_nue
            weight_dict[NU_PROB_FROM_MU_KEY] = prob_from_numu

            # Adding a "no oscillation" case
            prob_from_nue_without_osc  = 1. if np.abs(true_pdg_code) == 12 else 0.
            prob_from_numu_without_osc = 1. if np.abs(true_pdg_code) == 14 else 0.


            #
            # Compute overall weight
            #

            # Note that this is NOT normalised by number of files


            # Start from OneWeight, and normalise it (for this file, a user will also need to 
            # normalise based on the number of files they use)
            normed_one_weight = weight_dict[NU_ONE_WEIGHT_KEY] / (weight_dict[NU_NUM_EVENTS_KEY] * weight_dict[NU_GEN_RATIO_KEY] )

            # Apply flux and oscillations
            weight_with_osc = normed_one_weight * ( (weight_dict[NU_FLUX_E_KEY]*prob_from_nue) + (weight_dict[NU_FLUX_MU_KEY]*prob_from_numu) ) 

            # Also add a "no oscillations" case
            weight_without_osc = normed_one_weight * ( (weight_dict[NU_FLUX_E_KEY]*prob_from_nue_without_osc) + (weight_dict[NU_FLUX_MU_KEY]*prob_from_numu_without_osc) ) 

            # Write to the frame
            weight_dict[NU_NORM_ONE_WEIGHT_KEY] = normed_one_weight
            weight_dict[WEIGHT_KEY] = weight_with_osc
            weight_dict[NU_NO_OSC_WEIGHT_KEY] = weight_without_osc



def single_powerlaw_flux( norm, spectral_index, energy_GeV ) :
    '''
    Calculate a single power law flux
    In general, we use the following units for flux (and hence `norm`) : GeV^-1 cm^-2 s^-1 sr^-1
    Norm is defined as the flux at 1 GeV
    '''
    #TODO move to a dedicated `flux.py`?
    return norm * np.power( energy_GeV, spectral_index ) 


def add_single_powerlaw_flux_weight( frame, norm, spectral_index, overwrite=False ) :
    '''
    Function to add a single power law weight to the frame
    No oscillations included
    '''

    # Check the required inputs are there
    assert WEIGHT_DICT_KEY in frame, "Could not find weight dict '%s' in the frame" % WEIGHT_DICT_KEY
    assert icecube.oscNext.frame_objects.simulation.IN_ICE_PRIMARY_KEY in frame, "Could not find truth particle '%s' in the frame" % icecube.oscNext.frame_objects.simulation.IN_ICE_PRIMARY_KEY

    # Grab the particle energy
    true_energy_GeV = frame[icecube.oscNext.frame_objects.simulation.IN_ICE_PRIMARY_KEY].energy
    
    # Open the weight dict for editing
    with UpdateFrameObject(frame,WEIGHT_DICT_KEY) as weight_dict :

        # Bail out if already written
        if overwrite == False :
            if NU_SINGLE_POWERLAW_WEIGHT_KEY in weight_dict :
                return

        # Compute flux
        flux = single_powerlaw_flux( norm=norm, spectral_index=spectral_index, energy_GeV=true_energy_GeV ) 

        # Compute weight
        weight = weight_dict[NU_ONE_WEIGHT_KEY] * flux / ( weight_dict[NU_NUM_EVENTS_KEY] * weight_dict[NU_GEN_RATIO_KEY] )

        # Store everything
        weight_dict[NU_SINGLE_POWERLAW_FLUX_KEY] = flux
        weight_dict[NU_SINGLE_POWERLAW_WEIGHT_KEY] = weight
        weight_dict[NU_SINGLE_POWERLAW_NORM_KEY] = norm
        weight_dict[NU_SINGLE_POWERLAW_INDEX_KEY] = spectral_index



#
# Truth
#

def neutrino_extra_truth_info(frame, output_key=None, overwrite=False) :
    '''
    Add a bunch of extra truth info beyond just the initial neutrino properties
    '''

    # Defaults
    if output_key is None :
        output_key = icecube.oscNext.frame_objects.simulation.EXTRA_TRUTH_INFO

    # Check if already computed
    if frame.Has(str(output_key)) and (not overwrite) :
        return True


    #
    # Grab frame objects
    #

    # Get MC tree
    mc_tree = frame[str("I3MCTree")]

    # Get the primary
    primary_nu = frame[icecube.oscNext.frame_objects.simulation.IN_ICE_PRIMARY_KEY]

    # This code is specific to neutrino interactions
    assert np.abs(primary_nu.pdg_encoding) in [12, 14, 16]

    # Get all secondaries
    secondaries = mc_tree.get_daughters(primary_nu)

    # Get weight dict
    weight_dict = frame[WEIGHT_DICT_KEY]


    #
    # Get outgoing lepton properties
    #

    # This code assumes (but checks) that the outgoing lepton from the primary vertex is the first child in the tree (which it is for GENIE)

    # Init variables
    outgoing_lepton_energy = np.NaN 
    outgoing_lepton_length = np.NaN 
    outgoing_lepton_endpoint = None 

    # Get outgoing lepton energy from GENIE dicts, for use later when checking the particle in the MC tree
    #TODO

    # Here initiaising a "canonical" track/cascade energy.
    # This is what people normally mean by this, e.g. the "track energy/length" is the energy/length of the muon from the numu CC vertex (or 0 for any 
    # other flavor-interaction combo), and the "cascade energy" is the remainder of the neutrino energy.
    # Using 0 rather than NaN for null, as ML training prefers this.
    # Caveats:
    #  - This does't account for visible energy (e.g. invisible final state partice ssuch as neutirnos carrying away energy with no Cherekov light)
    #  - This doesn't include any tau decay products, since they are not included in the I3MCTree currently
    #  - There can also be "tracks" from the nucelus break up, which can be energetic and can exceed the muon 
    #    from the numu CC vertex. But this is rare, and not including this here, but note that "max_muon_energy" 
    #    might exceed "track_energy" for this reason.
    canonical_track_energy = 0.
    canonical_track_length = 0.

    # This depends on the interaction...
    if weight_dict[str("InteractionType")] == 1 : # CC

        # This depends on the flavor...
        if np.abs(primary_nu.pdg_encoding) == 12 :

            #TODO muon stochastic losses not included here, could try and do this but need to avoid double counting

            # nue CC
            outgoing_e = secondaries[0]
            assert np.abs(outgoing_e.pdg_encoding) == 11
            outgoing_lepton_energy = outgoing_e.energy
            assert np.isclose(primary_nu.pos.x, outgoing_e.pos.x) and np.isclose(primary_nu.pos.y, outgoing_e.pos.y) and np.isclose(primary_nu.pos.z, outgoing_e.pos.z), "Electron in CC nue event does not originate from interaction vertex!?!?"

        elif np.abs(primary_nu.pdg_encoding) == 14 :

            # numu CC
            outgoing_mu = secondaries[0]
            assert np.abs(outgoing_mu.pdg_encoding) == 13
            outgoing_lepton_energy = outgoing_mu.energy
            assert np.isclose(primary_nu.pos.x, outgoing_mu.pos.x) and np.isclose(primary_nu.pos.y, outgoing_mu.pos.y) and np.isclose(primary_nu.pos.z, outgoing_mu.pos.z), "Muon in CC numu event does not originate from interaction vertex!?!?"

            outgoing_lepton_length = outgoing_mu.length
            outgoing_lepton_endpoint = calc_stopping_point(vertex=primary_nu.pos, direction=outgoing_mu.dir, length=outgoing_mu.length)

            canonical_track_energy = outgoing_mu.energy
            canonical_track_length = outgoing_mu.length

        elif np.abs(primary_nu.pdg_encoding) == 16 :

            # nutau CC
            outgoing_tau = secondaries[0]
            assert np.abs(outgoing_tau.pdg_encoding) == 15
            outgoing_lepton_energy = outgoing_tau.energy
            assert np.isclose(primary_nu.pos.x, outgoing_tau.pos.x) and np.isclose(primary_nu.pos.y, outgoing_tau.pos.y) and np.isclose(primary_nu.pos.z, outgoing_tau.pos.z), "Tau in CC nutau event does not originate from interaction vertex!?!?"

            # tau length not stored currently

    elif weight_dict[str("InteractionType")] == 2 : # NC

        # Outgoing particle is neutrino
        outgoing_nu = secondaries[0]
        assert outgoing_nu.pdg_encoding == primary_nu.pdg_encoding

    # Update canonical values
    canonical_cascade_energy = primary_nu.energy - canonical_track_energy

    # Check energy matches the expectation, given the inelasticity of the event and the "El" variable in GENIE (need to account for rest mass though)
    #TODO

    #TODO Handle other interation channels (elastic, coherent, IMD, ...)


    #
    # Get energy loss particles
    #

    #TODO note that tau decay particles are not stored currently

    # Get EM secondaries
    em_secondaries = []
    for p in secondaries :
        if p.type in icecube.oscNext.frame_objects.simulation.EM_PARTICLE_TYPES :
            em_secondaries.append(p)

    # Get muon secondaries
    muon_secondaries = []
    for p in secondaries :
        if p.type in icecube.oscNext.frame_objects.simulation.MUON_PARTICLE_TYPES :
            muon_secondaries.append(p)

    # Get tau secondaries
    tau_secondaries = []
    for p in secondaries :
        if p.type in icecube.oscNext.frame_objects.simulation.TAU_PARTICLE_TYPES :
            tau_secondaries.append(p)

    # Get hadronic secondaries
    hadr_secondaries = []
    for p in secondaries :
        if p.type in icecube.oscNext.frame_objects.simulation.HADRONIC_PARTICLE_TYPES :
            hadr_secondaries.append(p)

    # Get muon stochastic losses
    # Do NOT include these in any energy sum, since ther enegy is already counted in the muon itself
    muon_stochastics = []
    for p in muon_secondaries :
        for s in mc_tree.get_daughters(p) :
           muon_stochastics.append(s)  #TODO distinguish EM vs hadronic? is mostly EM though, at least a GeV-scale

    # Get total energy loss in the various processes
    em_energy_total = np.sum([ p.energy for p in em_secondaries ])
    muon_energy_total = np.sum([ p.energy for p in muon_secondaries ])
    tau_energy_total = np.sum([ p.energy for p in tau_secondaries ])
    hadr_energy_total = np.sum([ p.energy for p in hadr_secondaries ])
    muon_stochastics_energy_total = np.sum([ p.energy for p in muon_stochastics ])

    # Get total visible secondary energy
    # - Don't include tau in this, since its decay particles are not included in the tree so don't know how much of the energy is visible
    # - Don't include muon stochastics, since would double count the muon energy
    visible_energy = em_energy_total + muon_energy_total + hadr_energy_total

    # Get highest energy particle of each type
    max_em_energy = np.max([ p.energy for p in em_secondaries]) if len(em_secondaries) > 0 else np.NaN
    max_hadr_energy = np.max([ p.energy for p in hadr_secondaries]) if len(hadr_secondaries) > 0 else np.NaN
    max_muon_energy = np.max([ p.energy for p in muon_secondaries]) if len(muon_secondaries) > 0 else np.NaN

    # Also get length
    muon_length_total = np.sum([ p.length for p in muon_secondaries ]) if len(muon_secondaries) > 0 else np.NaN
    max_muon_length = np.max([ p.length for p in muon_secondaries]) if len(muon_secondaries) > 0 else np.NaN


    #
    # Get geometry variables
    #

    # Vertex radial distance
    vertex_rho36 = calc_rho_36(x=primary_nu.pos.x, y=primary_nu.pos.y)

    # End point radial distance
    outgoing_lepton_endpoint_rho36 = np.NaN if outgoing_lepton_endpoint is None else calc_rho_36(x=outgoing_lepton_endpoint.x, y=outgoing_lepton_endpoint.y)

    # Starting containment
    dc_starting_containment = int( get_deepcore_containment(x=primary_nu.pos.x, y=primary_nu.pos.y, z=primary_nu.pos.z) )
    ic_starting_containment = int( get_icecube_containment(x=primary_nu.pos.x, y=primary_nu.pos.y, z=primary_nu.pos.z) )

    # Stopping containment
    dc_stopping_containment = np.NaN if outgoing_lepton_endpoint is None else int( get_deepcore_containment(x=outgoing_lepton_endpoint.x, y=outgoing_lepton_endpoint.y, z=outgoing_lepton_endpoint.z) )
    ic_stopping_containment = np.NaN if outgoing_lepton_endpoint is None else int( get_icecube_containment(x=outgoing_lepton_endpoint.x, y=outgoing_lepton_endpoint.y, z=outgoing_lepton_endpoint.z) )


    #
    # Store results
    #

    # Delete old objects if overwriting
    if overwrite and frame.Has(str(output_key)) :
        frame.Delete(str(output_key))

    # Finalise variables
    if outgoing_lepton_endpoint is None :
        outgoing_lepton_endpoint = dataclasses.I3Position(np.NaN, np.NaN, np.NaN)

    # Store extra truth info dict
    results = dataclasses.I3MapStringDouble()

    results[str("outgoing_lepton_length")] = outgoing_lepton_length
    results[str("outgoing_lepton_energy")] = outgoing_lepton_energy

    results[str("em_energy_total")] = em_energy_total
    results[str("muon_energy_total")] = muon_energy_total
    results[str("tau_energy_total")] = tau_energy_total
    results[str("hadr_energy_total")] = hadr_energy_total
    results[str("muon_stochastics_energy_total")] = muon_stochastics_energy_total

    results[str("max_em_energy")] = max_em_energy
    results[str("max_hadr_energy")] = max_hadr_energy
    results[str("max_muon_energy")] = max_muon_energy

    results[str("muon_length_total")] = muon_length_total
    results[str("max_muon_length")] = max_muon_length

    results[str("visible_energy")] = visible_energy

    results[str("canonical_cascade_energy")] = canonical_cascade_energy
    results[str("canonical_track_energy")] = canonical_track_energy
    results[str("canonical_track_length")] = canonical_track_length

    results[str("vertex_rho36")] = vertex_rho36

    results[str("outgoing_lepton_endpoint_x")] = outgoing_lepton_endpoint.x
    results[str("outgoing_lepton_endpoint_y")] = outgoing_lepton_endpoint.y
    results[str("outgoing_lepton_endpoint_z")] = outgoing_lepton_endpoint.z
    results[str("outgoing_lepton_endpoint_rho36")] = outgoing_lepton_endpoint_rho36

    results[str("dc_starting_containment")] = dc_starting_containment
    results[str("dc_stopping_containment")] = dc_stopping_containment
    results[str("ic_starting_containment")] = ic_starting_containment
    results[str("ic_stopping_containment")] = ic_stopping_containment

    frame[str(output_key)] = results

    # Also store the DC starting containment in the old style for backwards compatibility
    if frame.Has(STARTING_NEUTRINO_KEY) and overwrite :
        frame.Delete(STARTING_NEUTRINO_KEY)
    frame[STARTING_NEUTRINO_KEY] = I3Bool(bool(dc_starting_containment))

    return True
