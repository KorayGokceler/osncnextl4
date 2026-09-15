'''
Various physics helper functions for e.g. kinematics
'''

import numpy as np

from icecube.oscNext.frame_objects.weighting import WEIGHT_DICT_KEY
from icecube.dataclasses import I3Particle


def get_energy_and_momentum(particle) :
    '''
    Return a particle's energy and 3-momentum vector (e.g. 4-momentum)
    '''

    # Get total energy
    E = particle.total_energy # This is E_kinetic + m

    # print(particle.total_energy, particle.kinetic_energy, particle.mass, particle.kinetic_energy+particle.mass)

    # Get direction unit vector
    u = np.array([particle.dir.x, particle.dir.y, particle.dir.z])
    u /= np.sqrt(np.sum(np.square(u)))

    # Get 3-momentum
    p3_mag =  np.sqrt( np.square(E) - np.square(particle.mass) ) # Magnitude
    p3 = u * p3_mag

    return E, p3


def get_nutau_interaction_products(frame) :
    '''
    Get the tau decay particles for a nutau CC interaction
    '''

    #TODO generalise to e.g. also muon decays

    # Get the truth info from the frame
    mc_tree = frame["I3MCTree_clsim"] # This is the version with secondaries from propagation addded - required for tau decays
    weight_dict = frame[WEIGHT_DICT_KEY]

    # Get the primary neutrino
    nu = mc_tree[0]
    assert np.abs(nu.pdg_encoding) == 16, "Primary is a not a nutau"

    # Only relevent for CC interactions     #TODO what about other rare channels?
    tau, tau_decay_products = None, None
    if weight_dict["InteractionType"] == 1 :

        # Get the interaction final state
        secondaries = mc_tree.get_daughters(nu)

        # Get the primary outgoing lepton, in this case a tau
        tau = secondaries[0]
        assert np.abs(tau.pdg_encoding) == 15 

        # Get the tau decay products
        tau_decay_products = get_tau_decay_products(mc_tree, tau)

    return nu, tau, tau_decay_products


def get_tau_decay_products(mc_tree, tau) :
    '''
    Get the decay particles for a tau
    '''

    assert isinstance(tau, I3Particle)
    assert np.abs(tau.pdg_encoding) == 15

    # Get the tau decay products
    tau_decay_products = mc_tree.get_daughters(tau)
    # assert len(tau_decay_products) > 0

    #TODO Get polarization

    return tau_decay_products

