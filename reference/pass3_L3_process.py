#!/usr/bin/env python

from optparse import OptionParser

from icecube import dataio, dataclasses, simclasses
from icecube.icetray import I3Tray, pypick
from icecube import icetray
from icecube.online_filterscripts.online_filters.greco.grecovariables import DeepCoreCleaning, DeepCoreCuts


def pass_DC_filter(frame):
    """Check whether the DeepCoreFilter passed."""
    if frame.Has("OfflineFilterMask") and "OfflineSupDeepCoreFilter_24" in frame["OfflineFilterMask"]:
        return frame['OfflineFilterMask']['OfflineSupDeepCoreFilter_24'].prescale_passed
    elif frame.Has("FilterMask") and "DeepCoreFilter_13" in frame["FilterMask"]:
        return frame['FilterMask']['DeepCoreFilter_13'].prescale_passed
    return False

@icetray.traysegment
def oscNext_L3(tray, name,
               If=lambda frame: True):
    """Add oscNext L3 bool to frame."""

    #Apply DC filter.
    tray.Add(pass_DC_filter)

    # Run the DeepCore noise cleaning and create the "SRTTWSplitInIcePulsesDC" pulse map (if not there already)
    tray.Add(DeepCoreCleaning, name + "_DeepCoreCleaning",
             uncleaned_pulses = "SplitInIcePulses",
             If = pypick(lambda f: If(f) and not f.Has("SRTTWSplitInIcePulsesDC")),
            )

    # If L3 bools don't exist, calculate them
    tray.Add(DeepCoreCuts, name + "_DeepCoreCuts",
             useNamePrefix = False,
             If = pypick(lambda f: If(f) and not f.Has("IC2018_LE_L3_bools")),
            )

    # Make cuts to remove events suffering from data quality issues do to detector issues, etc
    def make_data_quality_cut(frame):
        if frame.Has('LIDErrata_osc_data_quality'):
            N_LIDErrata = len(frame['LIDErrata_osc_data_quality'])
        else:
            N_LIDErrata = 0
        if frame.Has("OfflineFilterMask") and "SLOPFilter_24" in frame["OfflineFilterMask"]:
            slop_bool = frame['OfflineFilterMask']['SLOPFilter_24'].prescale_passed
        elif frame.Has("FilterMask") and "SlopFilter_13" in frame["FilterMask"]:
            slop_bool = frame['FilterMask']['SlopFilter_13'].prescale_passed
        else:
            slop_bool = False
        # accept event if no errata in frame and it has NOT passed the SLOP filter
        frame["Data_quality_bool"] = icetray.I3Bool(N_LIDErrata == 0 and not slop_bool)
    tray.Add(make_data_quality_cut)

    #Store L3 boolean.
    def add_bool(frame):
        L3_bool = frame['IC2018_LE_L3_bools']['IC2018_LE_L3_Full'] and frame["Data_quality_bool"].value
        frame['L3_oscNext_bool'] = icetray.I3Bool(L3_bool)
    tray.Add(add_bool)

    # Delete keys that may have been created but should not be kept
    tray.AddModule("Delete", name + "_cleanup",
                   Keys = [name + "_DeepCoreCutsNChAbove200",
                           "SplitInIcePulses_STW_" + name + "_DeepCoreCutsNoiseEngine",
                           name + "_DeepCoreCutsSRTTWSplitInIcePulsesDCHitStatistics",
                           name + "_DeepCoreCutsTWSplitInIcePulsesDCTimeRange",
                           "SplitInIcePulses_STW_" + name + "_DeepCoreCutsNoiseEngineTimeRange",
                           name + "_DeepCoreCutsSplitInIcePulsesHitStatistics",
                           name + "_DeepCoreCutsNoiseEngineNoCharge_bool",
                           "SplitInIcePulses_STW_ClassicRT_" + name + "_DeepCoreCutsNoiseEngine",
                           name + "_DeepCoreCutsTWSplitInIcePulsesDC",
                           name + "_DeepCoreCutsTWRTVetoSeries",
                           "BadOMSelection",
                           "SRTTW_SplitInIcePulses",
                           "NoiseEngine_bool",
                           "TWSplitInIcePulses",
                           "TWSplitInIcePulsesTimeRange",
                          ],
                  )


usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("-o", "--outdir", dest="OUTDIR", help="Write output to OUTFILE")
parser.add_option("-i", "--infile", dest="INFILE", help="Read input from INFILE (.i3 format)")
parser.add_option("-g", "--gcdfile", default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz", dest="GCDFILE", help="Read in GCD file")
(options,args) = parser.parse_args()

gcd, infile, outdir = options.GCDFILE, options.INFILE, options.OUTDIR
if 'Level2' in infile.split('/')[-1]:
    outfile = infile.split('/')[-1].replace('Level2', 'L3')
elif 'OfflineFiltered' in infile.split('/')[-1]:
    outfile = infile.split('/')[-1].replace('OfflineFiltered', 'L3')
else:
    raise NameError(f"Infile {infile} doesn't follow naming convention.")


tray = I3Tray()
tray.Add(dataio.I3Reader, "reader", FilenameList=[gcd, infile])
tray.Add(oscNext_L3)
tray.Add("I3Writer", "writer",
         filename = outdir+outfile)
tray.Add("TrashCan", "can")
tray.Execute()
tray.Finish()
