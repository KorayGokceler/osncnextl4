'''
Useful tools for GENIE simulation

This includes functions for handling GENIE systematic uncertainties

Tom Stuttard
'''

import sys
import numpy as np

import icecube
from icecube import icetray
from icecube.oscNext.frame_objects.weighting import WEIGHT_DICT_KEY

from icecube.dataclasses import I3VectorDouble, I3Double


#
# Globals
#

# This is the main GENIE result output
GENIE_RESULT_DICT_KEY = "I3GENIEResultDict"
GENIE_MEC_FLAG_KEY = "GENIE_MEC"

# Some backwards compatibility handling here
# We used to use genie_icetray project to generate GENIE MC, but more recently switched to genie_reader
# oscNext GENIE  step1 files used genie_icetray, and so we need to support reading its data (even in modern IceTray)
# In practise, this means need to be able to load "I3GENIEResultDict" from the frame
# In old code, do this by importing genie_icetray
# In modern IceTray, instead import this from simclasses        #TODO support loading the new output object from genie-reader in this code too
try :
    from icecube import genie_icetray
except ImportError as e :
    from icecube import simclasses
    assert hasattr(simclasses, GENIE_RESULT_DICT_KEY), "Could not find simclasses.%s" % GENIE_RESULT_DICT_KEY

# Define useful variables from GENIE events pertaining to the interaction
# See GENIE manual (https://arxiv.org/pdf/1510.05494.pdf, section 7.6.2)
# This is a nice resource for interaction variables descriptions: https://danielscully.co.uk/thesis/
GENIE_INTERACTION_VARS = [
    "tgt", # Target PDG code
    "Z", # Nuclear target Z
    "A", # Nuclear target A
    "x", # Bjorken x
    # "xs", # Alternative x calc
    "y", # Inelasticity
    # "ys", # Alternative y calc
    "Q2", # Momentum transfer, Q^2, in GeV^2
    # "Q2s", # Alternative Q2 calc
    "W", # Hadronic invariant mass, W
    # "Ws", # Alternative W calc
    "xsec", # cross section
    "diffxsec", # differential cross section
    "qel", # quasielastic event
    "res", # resonance event
    "dis", # DIS event
    "imd", # Inverse muon decay event
    "coh", # Coherent meson production event
    "dfr", # Diffractive event
    "nuel", # nu-e elastic scattering event
    "cc", # charged current
    "nc", # neutral current
    "charm", # charm produced
]
GENIE_INTERACTION_VARS = [ GENIE_RESULT_DICT_KEY+"."+v for v in GENIE_INTERACTION_VARS ]

INTERACTIONS = { "qel":0, "res":1, "dis":2, "other":3 }

# Define variables neeed to compute GENIE systematics
GENIE_SYSTS = (
    'AhtBY', 'BhtBY', 'CV1uBY', 'CV2uBY',
    'MaCCQE', 'MaCCRES', 'MaCOHpi', 'MaNCEL', 'MaNCRES'
)

GENIE_SYST_INDS = (0, 1, 2, 3)



#
# GENIE systematics
#

# Define GENIE variables that are written to the frame (by ExtractGENIESystematics)
GENIE_SYST_FRAME_OBJECTS = ["GENIEWeight"]
for genie_syst in GENIE_SYSTS:
    for ind in GENIE_SYST_INDS:
        frame_obj = 'GENIE_rw_' + genie_syst + '_%d' % ind
        GENIE_SYST_FRAME_OBJECTS.append(frame_obj)


class ExtractGENIESystematics(icetray.I3Module):
    '''
    I3 module for extracting information required for GENEIE systematics handling in PISA
    '''

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox("OutBox")
        self.AddParameter("GENIEResultDict_Name", "Name of GENIEResult object", None)

    def Configure(self):
        self.genie_name = self.GetParameter("GENIEResultDict_Name")
        if self.genie_name is None:
            self.genie_name = GENIE_RESULT_DICT_KEY

    def DAQ(self, frame): #TODO Should this be P frame?
        if frame.Has(self.genie_name):
            genie_res = frame[self.genie_name]
            MCweight = frame["I3MCWeightDict"]
            if "GENIEWeight" in MCweight:
                wgt_zero = MCweight["GENIEWeight"]
            else:
                raise KeyError(
                    'Did not find a GENIE weight in frame!'
                )
                # TODO: or do we `return True` here?
                #return True
            frame['GENIEWeight'] = I3Double(wgt_zero)
            for key in genie_res.keys():
                # TODO: ensure consistency of systematics here with those in
                # `GENIE_SYSTS`
                if 'rw' in key:
                    syst_values = genie_res[key]
                    for i in GENIE_SYST_INDS:
                        frame["GENIE_"+key+'_%i'%i] = I3Double(syst_values[i])
        else:
            raise KeyError(
                'Did not find a GENIE weight in frame!'
            )
        self.PushFrame(frame)
        return True


#TODO Document this, including links to GENIE docs or IceCube wiki...
def fit_genie_syst(in_yvalues, genie_weight):
    # TODO: re-vectorise
    # are these the number of std. devs.?
    rw_xvalues  = np.array([-2, -1, 0, 1, 2])
    if sum(in_yvalues) == 4.0:
        return np.array([0, 0])
    in_yvalues = np.array(in_yvalues)
    yvalues = np.concatenate(
        (in_yvalues[:2]/genie_weight, [1.], in_yvalues[2:]/genie_weight)
    )
    # returns  the coefficients from low to high order (as opposed to np.polyfit)
    fitcoeff = np.polynomial.polynomial.polyfit(
        rw_xvalues, yvalues, deg=2
    )
    linear_fit_coefft, quadratic_fit_coefft = fitcoeff[1:]
    return linear_fit_coefft, quadratic_fit_coefft


def calc_genie_systematics(hdf5_events_file, group, dummy_only=False) :
    '''
    Calculate GENIE systematics for a single group
    '''

    #TODO Document this, including links to GENIE docs or IceCube wiki...

    if not dummy_only:
        # only read this once
        genie_weight = group.genie_weight.read()

    for genie_syst in GENIE_SYSTS:
        # same polynomial fits to all GENIE systematics
        linear_fit = np.zeros(group.event_id.shape)
        quad_fit = np.zeros(group.event_id.shape)

        if not dummy_only:
            # Get the information pertainting to the GENIE axial mass
            # systematics that have been written to the hdf5 file
            genie_rw_syst = []
            # collect all weight variations for this systematic
            for i in GENIE_SYST_INDS:
                genie_rw_syst_i = getattr(
                    group,
                    SIMPLE_STD_GENIE_MAPPINGS['rw_' + genie_syst + '_%d' % i]
                ).read()
                genie_rw_syst.append(genie_rw_syst_i)

            genie_rw_syst = np.array(genie_rw_syst)

            # obtain linear and quadratic fit coefficients on an
            # event-by-event basis
            # TODO: vectorise
            for event_id, gw in enumerate(genie_weight):
                rw_syst_list = genie_rw_syst[:, event_id]
                # get the linear and quadratic terms from a poly fit
                linear_fit[event_id], quad_fit[event_id] = fit_genie_syst(
                    in_yvalues=rw_syst_list,
                    genie_weight=gw
                )

        # add the fit coefficients to the events
        hdf5_events_file.create_array(
            where=group, name='linear_fit_%s' % genie_syst.lower(),
            obj=linear_fit, title=''
        )
        hdf5_events_file.create_array(
            where=group, name='quad_fit_%s' % genie_syst.lower(),
            obj=quad_fit, title=''
        )


def calc_genie_systematics_for_group(hdf5_events_file, group, dummy_only=False) :
    '''
    Calculate GENIE systematics for a single group
    '''

    #TODO Document this, including links to GENIE docs or IceCube wiki...

    if not dummy_only:
        # only read this once
        genie_weight = group.genie_weight.read()

    for genie_syst in GENIE_SYSTS:
        # same polynomial fits to all GENIE systematics
        linear_fit = np.zeros(group.event_id.shape)
        quad_fit = np.zeros(group.event_id.shape)

        if not dummy_only:
            # Get the information pertainting to the GENIE axial mass
            # systematics that have been written to the hdf5 file
            genie_rw_syst = []
            # collect all weight variations for this systematic
            for i in GENIE_SYST_INDS:
                genie_rw_syst_i = getattr(
                    group,
                    SIMPLE_STD_GENIE_MAPPINGS['rw_' + genie_syst + '_%d' % i]
                ).read()
                genie_rw_syst.append(genie_rw_syst_i)

            genie_rw_syst = np.array(genie_rw_syst)

            # obtain linear and quadratic fit coefficients on an
            # event-by-event basis
            # TODO: vectorise
            for event_id, gw in enumerate(genie_weight):
                rw_syst_list = genie_rw_syst[:, event_id]
                # get the linear and quadratic terms from a poly fit
                linear_fit[event_id], quad_fit[event_id] = fit_genie_syst(
                    in_yvalues=rw_syst_list,
                    genie_weight=gw
                )

        # add the fit coefficients to the events
        hdf5_events_file.create_array(
            where=group, name='linear_fit_%s' % genie_syst.lower(),
            obj=linear_fit, title=''
        )
        hdf5_events_file.create_array(
            where=group, name='quad_fit_%s' % genie_syst.lower(),
            obj=quad_fit, title=''
        )


#
# GENIE data quality
#

def filter_bad_genie_events(frame) :
    '''
    Have found some unphysical GENIE events, use this function to purge them
    '''

    #
    # Error 1: Outgoing nu has higher energy than incoming nu for MEC NC events
    #

    # Have removed this check now...

    # error1 = False

    # # Grab incoming nu particle
    # nu_in = frame["I3MCTree"][0]
    # assert np.abs(nu_in.pdg_encoding) in [12, 14, 16] 

    # # Look at NC events only
    # genie_dict = frame[GENIE_RESULT_DICT_KEY]
    # if genie_dict["nc"] :

    #     # Get the outgoing nu
    #     nu_out = frame["I3MCTree"][1]
    #     assert np.abs(nu_out.pdg_encoding) in [12, 14, 16] 

    #     # Look for cases where the outgoing nu is higher energy than the incoming nu
    #     if nu_out.energy > nu_in.energy :
    #         error1 = True

    # # Bail if bad event
    # if error1 :
    #     return False


    #
    # Error 2: Outgoing lepton is not the expected type
    #

    error2 = False

    # Get the primary
    primary = frame[icecube.oscNext.frame_objects.simulation.IN_ICE_PRIMARY_KEY]

    # Get outgoing lepton
    mc_tree = frame[str("I3MCTree")]
    secondaries = mc_tree.get_daughters(primary)  
    outgoing_lepton = secondaries[0] # Assumption that it is always the first particle - TODO verify this

    # Check outputgong lepton c.f. primary
    if frame[WEIGHT_DICT_KEY]["InteractionType"] == 1 : # CC
        if primary.pdg_encoding == 12 :
            if outgoing_lepton.pdg_encoding != 11 :
                error2 = True
        elif primary.pdg_encoding == -12 :
            if outgoing_lepton.pdg_encoding != -11 :
                error2 = True
        elif primary.pdg_encoding == 14 :
            if outgoing_lepton.pdg_encoding != 13 :
                error2 = True
        elif primary.pdg_encoding == -14 :
            if outgoing_lepton.pdg_encoding != -13 :
                error2 = True
        elif primary.pdg_encoding == 16 :
            if outgoing_lepton.pdg_encoding != 15 :
                error2 = True
        elif primary.pdg_encoding == -16 :
            if outgoing_lepton.pdg_encoding != -15 :
                error2 = True

    elif frame[WEIGHT_DICT_KEY]["InteractionType"] == 2 : # NC
        if outgoing_lepton.pdg_encoding != primary.pdg_encoding :
            error2 = True

    # For these weird events, there are only 2 entries in the MC tree. Enforce this, in case there are other categories
    if error2 :
        assert len(mc_tree) == 2, "Found bad GENIE event with num particles != 2? !?!?? Might be a new category, needs investigation..."

    # Bail if bad event
    if error2 :
        return False


    # If here, nothing was bad
    return True


#
# Version compatibility
#

def handle_genie_result_dict_versions(frame):
    '''
    Handling differing form for the GENIE result object

    Takes the more recent form in genie-reader and converts it to the form used
    by the older genie-icetray (which was used for oscNext MC.
    '''

    # Check if old-style GENIE dict not present
    if GENIE_RESULT_DICT_KEY not in frame :

        # Can't find it, try and make it from the new-style object...

        # Get the new object
        assert "I3GenieResult" in frame
        genie_result = frame["I3GenieResult"]

        # Make an empty old object
        from icecube.simclasses import I3GENIEResultDict
        genie_result_old = I3GENIEResultDict()

        # Copy variable across
        for attr_name in dir(genie_result) :
            if not attr_name.startswith("_") :
                genie_result_old[attr_name] = getattr(genie_result, attr_name)

        # Add old object to frame
        frame[GENIE_RESULT_DICT_KEY] = genie_result_old


def flag_mec_events(frame, output_key=GENIE_MEC_FLAG_KEY) :
    '''
    MEC interaction events are not flagged in MC produced using genie-icetray 
    (or older versions of genie-reader). Here we add a flag manually.
    '''

    # Skip if flag already written
    if output_key in frame :
        return True

    # Read the result dict
    assert GENIE_RESULT_DICT_KEY in frame, "Could not find GENIE result dict : %s" % GENIE_RESULT_DICT_KEY
    genie_result_dict = frame[GENIE_RESULT_DICT_KEY]

    # Determine if event is MEC
    mec_flag = genie_result_dict["hitnuc"] in [2000000200, 2000000201, 2000000202] # Particle codes for pp, nn, and np clusters

    # Check MEC event doesn't also have any of the other interaction flags
    if mec_flag :
        assert not genie_result_dict["dis"], "GENIE events was flagged as 'mec' but also found 'dis' flag"
        assert not genie_result_dict["res"], "GENIE events was flagged as 'mec' but also found 'res' flag"
        assert not genie_result_dict["qel"], "GENIE events was flagged as 'mec' but also found 'qel' flag"
        assert not genie_result_dict["coh"], "GENIE events was flagged as 'mec' but also found 'coh' flag"
        assert not genie_result_dict["dfr"], "GENIE events was flagged as 'mec' but also found 'dfr' flag"
        assert not genie_result_dict["imd"], "GENIE events was flagged as 'mec' but also found 'imd' flag"
        assert not genie_result_dict["nuel"], "GENIE events was flagged as 'mec' but also found 'nuel' flag"

    # Write to the frame
    frame[output_key] = icetray.I3Bool(mec_flag)
