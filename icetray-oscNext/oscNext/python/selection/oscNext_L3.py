'''
The oscNext level 3 event selection tray segment.
Based on level3-filter-lowen/python/LowEnergyL3TraySegment.py

Tom Stuttard
'''

from icecube import dataclasses
from icecube import icetray
from icecube.icetray import I3Units
from icecube import NoiseEngine
from icecube.DeepCore_Filter import DOMS
from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService
from icecube.oscNext.selection.globals import *


#
# Output frame objects
#

# Define frame objects
L3_2018_VARS_KEY = "IC2018_LE_L3_Vars" #XXX: note, this object had "2012" in name before
L3_2018_ALL_BOOLS_KEY = "IC2018_LE_L3_bools"

# Make a list of objects to store in HDF5
L3_HDF5_KEYS = [
    L3_2018_VARS_KEY,
    L3_2018_ALL_BOOLS_KEY,
    L3_CUT_BOOL_KEY,
    "SRTTWOfflinePulsesDCFid",
]


#
# Tray segments
#

def PassedDCFilter(frame, year):
        # year = str(12)
        if frame.Has("FilterMask"):
            if frame["FilterMask"].get("DeepCoreFilter_"+year).condition_passed:
                return True
            # end if()
        # end if()
        return False
    # end PassedDCFilter()


@icetray.traysegment
def DataQualityCuts(tray, name, pulses, make_cut=False) :
    '''
    Make cuts to remove events suffering from data quality issues do to detector issues, etc
    '''

    from icecube.filterscripts.flaringDOMFilter import FlaringDOMFilter

    ##########################
    # Run flaring DOM analysis
    # 
    # The analysis runs on all
    # DOMs, not just the known
    # offenders.
    ##########################
    
    tray.AddModule(FlaringDOMFilter, 
               pulsesName=pulses,
               mode="CutAllEvents",
               sendErrataMoni=False,
               errataName='LIDErrata_osc_data_quality'
              )
    
    def add_errata_length(frame):
        '''Put length of errata vector into the frame as a number.
        '''
        if frame.Has('LIDErrata_osc_data_quality'):
            frame['N_LIDErrata'] = icetray.I3Int(len(frame['LIDErrata_osc_data_quality']))
        else:
            frame['N_LIDErrata'] = icetray.I3Int(0)
    tray.AddModule(add_errata_length, "errata_length")
    

    ##################################################################
    # Place bool in frame indicating whether it passed the SLOP filter
    ##################################################################
    
    def add_slop_bool(frame):
        slop_bool = False
        if (frame.Has("FilterMask") and
            frame["FilterMask"].has_key("SlopFilter_13") and
            frame["FilterMask"].get("SlopFilter_13").condition_passed):
            slop_bool = True
        frame['SLOP_passed_bool'] = icetray.I3Bool(slop_bool)
    tray.AddModule(add_slop_bool, "add_slop_bool")


    ######################
    # Overall cut decision
    ######################

    def make_data_quality_cut(frame,make_cut=False) :

        # Define the bool (will be True if event should be kept)
        frame["Data_quality_bool"] = icetray.I3Bool(
            ( frame["N_LIDErrata"].value == 0 ) * # True if no errata in frame   
            ( not frame["SLOP_passed_bool"].value ) # accept event if it has NOT passed the SLOP filter
        )

        # Make the cut
        if make_cut :
            return frame["Data_quality_bool"].value
        else :
            return True

    tray.AddModule( make_data_quality_cut, "L3_make_data_quality_cut", make_cut=make_cut )


def purge_greco_filter(frame) :
    '''
    From 2019 onwards, we have a problem that a version of this code 
    was copy-pasted into the GRWCO online filter, which runs at L2.

    See https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/meta-projects/combo/trunk/filterscripts/python/grecovariables.py

    This means that similar (but not nessecarily identical) versions 
    of these variable are already present in the code, and need removing
    before we write our own

    This is a bit of a mass, different variable names should really be used
    TODO Solve this better
    '''

    problem_keys = [
        L3_2018_VARS_KEY,
        L3_2018_ALL_BOOLS_KEY,
    ]
    #TODO others?

    for k in problem_keys :
        if k in frame :
            frame.Delete(k)


@icetray.traysegment
def DeepCoreCuts(
    tray, 
    name, 
    Iffy = lambda f: True,
    splituncleaned='SplitInIcePulses',
    Pulses = 'SRTTWOfflinePulsesDC' ,  ### Pulsemap Generated in L2
    alttws=False,
    DoXYCut=False,
    year='12',
):
    '''
    This is directly lifted from Based on level3-filter-lowen/python/LowEnergyL3TraySegment.py
    '''

    # Remove GRECO filter
    tray.Add(purge_greco_filter, "purge_greco_filter")

    icetray.load("static-twc",False)
    def RunAlternateTWs(frame):
        if alttws:
           return True
        return False

    from icecube.common_variables import hit_multiplicity, hit_statistics, time_characteristics
    for tmpPulses in [splituncleaned, Pulses]:
        tray.AddModule( hit_multiplicity.I3HitMultiplicityCalculator, 
                        tmpPulses + "HMcalc", 
                        PulseSeriesMapName=tmpPulses, 
                        OutputI3HitMultiplicityValuesName=tmpPulses+"HitMultiplicity")
        tray.AddModule( hit_statistics.I3HitStatisticsCalculator, 
                        tmpPulses + "HScalc",
                        PulseSeriesMapName=tmpPulses, 
                        OutputI3HitStatisticsValuesName=tmpPulses + "HitStatistics" )
        tray.AddModule( time_characteristics.I3TimeCharacteristicsCalculator, 
                    tmpPulses + "TCcalc", 
                    PulseSeriesMapName=tmpPulses, 
                    OutputI3TimeCharacteristicsValuesName=tmpPulses + "TimeCharacteristics")
    del tmpPulses
    

    # Data quality (don't cut, include it in the main L3 cut)
    tray.AddSegment( DataQualityCuts, "L3_DataQualityCuts", pulses=Pulses, make_cut=False )

    
    ### Run NoiseEngine ###
    tray.AddSegment( NoiseEngine.WithCleaners, "NoiseEnginess",
                     HitSeriesName = splituncleaned,
                     If = lambda f: PassedDCFilter(f, year=year)
                     )
    ### also no charge version as well
    # will reuse the output from previous segment
    tray.AddModule("NoiseEngine",name+"NoiseEngine",
                HitSeriesName = splituncleaned+"_STW_ClassicRT_NoiseEnginess",
                OutputName = "NoiseEngineNoCharge_bool",
                ChargeWeight = False,
                If= lambda f: PassedDCFilter(f, year=year)
        )
    DOMList = DOMS.DOMS( "IC86EDC")

    ### Need to generate TWC Pulses // No longer present in L2 ###

    tray.AddModule('I3StaticTWC<I3RecoPulseSeries>', name + '_StaticTWC_DC',
                   InputResponse = splituncleaned,
                   OutputResponse = 'TWOfflinePulsesDC',
                   TriggerConfigIDs = [1011],
                   TriggerName = "I3TriggerHierarchy",
                   WindowMinus = 5000,
                   WindowPlus = 4000,
                   If = lambda f: PassedDCFilter(f, year=year) and not f.Has("TWOfflinePulsesDC")
                   )

    tray.AddModule( "I3OMSelection<I3RecoPulseSeries>", "GenDCInTWFidPulses",
                    selectInverse  = True,
                    InputResponse  = 'TWOfflinePulsesDC',
                    OutputResponse = 'TWOfflinePulsesDCFid',
            OutputOMSelection = 'Baddies',
                    OmittedKeys    = DOMList.DeepCoreFiducialDOMs,
                    If = lambda f: PassedDCFilter(f, year=year) and not f.Has("TWOfflinePulsesDCFid")
                    )

    # #############################
    # The goal here is to reduce the noise events by looking
    # for events which have tightly time correlated hits in the
    # DC Fiducial volume.
    # #############################


    tray.AddModule( "I3OMSelection<I3RecoPulseSeries>", "GenDCPulses",
                    selectInverse  = True,
                    InputResponse  = splituncleaned,
                    OutputResponse = 'OfflinePulsesDCFid',
                    OmittedKeys    = DOMList.DeepCoreFiducialDOMs,
                    If = lambda f: PassedDCFilter(f, year=year)
                    )

    tray.AddModule( "I3OMSelection<I3RecoPulseSeries>", "GenICVetoPulses_0",
                    selectInverse     = False,
                    InputResponse     = splituncleaned,
                    OutputResponse    = "OfflinePulsesICVeto",
                    OutputOMSelection = "BadOM_ICVeto_0",
                    OmittedKeys       = DOMList.DeepCoreFiducialDOMs,
                    If = lambda f: PassedDCFilter(f, year=year)
                    )

    tray.AddModule("I3DeepCoreVeto<I3RecoPulse>", "deepcore_filter_pulses",
                   InputFiducialHitSeries = "OfflinePulsesDCFid",
                   InputVetoHitSeries     = "OfflinePulsesICVeto",
                   DecisionName           = "DCFilterPulses",
                   MinHitsToVeto          = 2,
                   VetoChargeName         = "DCFilterPulses_VetoPE",
                   VetoHitsName           = "DCFilterPulses_VetoHits",
                   If = lambda f: PassedDCFilter(f, year=year)
                   )
    
    # #############################
    # Run a dynamic time window cleaning over the
    # StaticTW cleaned recopulses in the DeepCore
    # fiducial region.
    # #############################

    tray.AddModule( "I3TimeWindowCleaning<I3RecoPulse>", "DynamicTimeWindow175", 
                    InputResponse  = "TWOfflinePulsesDCFid",
                    OutputResponse = "DCFidPulses_DTW175", 
                    TimeWindow     = 175,
                   If = lambda f: PassedDCFilter(f, year=year) and RunAlternateTWs(f)
                    )

    tray.AddModule( "I3TimeWindowCleaning<I3RecoPulse>", "DynamicTimeWindow200", 
                    InputResponse  = "TWOfflinePulsesDCFid",
                    OutputResponse = "DCFidPulses_DTW200", 
                    TimeWindow     = 200,
                   If = lambda f: PassedDCFilter(f, year=year) and RunAlternateTWs(f)
                    )

    tray.AddModule( "I3TimeWindowCleaning<I3RecoPulse>", "DynamicTimeWindow250",
                    InputResponse  = "TWOfflinePulsesDCFid",
                    OutputResponse = "DCFidPulses_DTW250",
                    TimeWindow     = 250,
                    If = lambda f: PassedDCFilter(f, year=year) and RunAlternateTWs(f)
                    )    
    tray.AddModule( "I3TimeWindowCleaning<I3RecoPulse>", "DynamicTimeWindow300",
                    InputResponse  = "TWOfflinePulsesDCFid",
                    OutputResponse = "DCFidPulses_DTW300",
                    TimeWindow     = 300,
                    If = lambda f: PassedDCFilter(f, year=year)
                    )

    # #############################
    # Put all the MicroCount variables
    # into a map for elegance
    # #############################

    def MicroCount(frame):
      # if If(frame):
        MicroValuesHits = dataclasses.I3MapStringInt()
        MicroValuesPE   = dataclasses.I3MapStringDouble()

        DTWs = ["DTW175","DTW200","DTW250", "DTW300"]
        for DTW in DTWs:
            totalCharge = 0
            if not frame.Has("DCFidPulses_" + DTW):
                MicroValuesHits["STW9000_" + DTW] = 0
                MicroValuesPE["STW9000_" + DTW] = totalCharge
                continue
            # end if()

            for omkey in frame["DCFidPulses_" + DTW].apply(frame):
                for pulse in omkey[1]:
                    totalCharge += pulse.charge
                # end for()
            # end for()
            MicroValuesHits["STW9000_" + DTW] = len(frame["DCFidPulses_" + DTW].apply(frame))
            MicroValuesPE["STW9000_" + DTW] = totalCharge
        # end for()
        frame["MicroCountHits"] = MicroValuesHits
        frame["MicroCountPE"] = MicroValuesPE
        return
 #     return
    # end MicroCount()

    # #############################
    # Add the MicroCount MapStringDouble object to the frame
    # #############################

    tray.AddModule( MicroCount, "MicroCount", If = lambda f: PassedDCFilter(f, year=year))

    # ##############################
    # Rewrite the QR_Box method from the
    # slc-veto project into python so I don't
    # have to faff about with a code review.
    # ##############################

    def CalculateC2QR6AndFirstHitVertexZ( frame, PulseSeries):
#      if If(frame):

        totalCharge = 0
        totalChannels = 0.0
        timeFirstHit = []
        timeCharge  = {}
        timeOMKey   = {}

        geometry = frame[ "I3Geometry"]
        vertexZ = float("nan")
        vertexY = float("nan")
        vertexX = float("nan")
        firstHitTime = float("inf") # This is a correct way to define infine number


        for omkey in frame[ PulseSeries].apply(frame):
            cur_times = []
            totalChannels += 1.0
            for pulse in omkey[1]:
                timeCharge[pulse.time] = pulse.charge
                timeOMKey[pulse.time]  = omkey[0]
                cur_times.append(pulse.time)
                totalCharge += pulse.charge
                if pulse.time < firstHitTime:
                    firstHitTime = pulse.time
                    vertexZ = geometry.omgeo[ omkey[0]].position.z
                    vertexY = geometry.omgeo[ omkey[0]].position.y
                    vertexX = geometry.omgeo[ omkey[0]].position.x
                # end if()    
            timeFirstHit.append( min(cur_times) )                                 
            # end for()
        # end for()

        frame[ "VertexGuessZ"] = dataclasses.I3Double( vertexZ)
        frame[ "VertexGuessY"] = dataclasses.I3Double( vertexY)
        frame[ "VertexGuessX"] = dataclasses.I3Double( vertexX)
        frame[ "VertextGuessTime"] = dataclasses.I3Double( firstHitTime)
        
        orderedTimeCharge = sorted(timeCharge.items())
        orderedTimeOMKey  = sorted(timeOMKey.items())
        orderedTimeFirstHit = sorted(timeFirstHit)

        redTimeCharge = orderedTimeCharge[2:]

        # Check to see whether there is at least one pulse
        # in the cleaned dataset. Normally this condition is
        # impossible to not satisfy,
        # but isolated HLC hits change the game.

        if len( redTimeCharge) < 1:
            print(" In the limit that zero/zero goes to large zero in the \n denominator, C2QR6 will be zero.")
            frame["C2QR6"] = dataclasses.I3Double(0)
        else: 
            totalCharge2 = 0
            C2QR6        = 0
            startTime    = redTimeCharge[0][0]

            for time, charge in redTimeCharge:
                if (time - startTime) < 600:
                    C2QR6 += charge
                # end if()
                totalCharge2 += charge
            # end for()

            C2QR6 = C2QR6/totalCharge2

            if C2QR6 > 0:
                frame["C2QR6"] = dataclasses.I3Double(C2QR6)
            else:
                print(str(frame["I3EventHeader"].event_id) + " has somehow received a negative value of C2QR6")
                print(" value is :", C2QR6)
                return
            # end if()
        # end if()
        # Next part gets calculates hit ratio
        redTimeFirstHit = orderedTimeFirstHit[2:]
        if len(redTimeFirstHit)<1:  
            print(" In the limit that zero/zero goes to large zero in the \n denominator, C2HR6 will be zero.")      
            frame["C2HR6"] = dataclasses.I3Double(0)
        else:
            C2HR6 = 0.0
            startTimeFirstHit = redTimeFirstHit[0]
            for time in redTimeFirstHit :
                if (time - startTimeFirstHit)<  600.0:
                    C2HR6+=1.0  
            # bizzare type conversion to avoind stumbling                                  
            C2HR6 = float(C2HR6)/float(len(redTimeFirstHit)) 
            if C2HR6 > 0:
                frame["C2HR6"] = dataclasses.I3Double(C2HR6)
            else:
                print(str(frame["I3EventHeader"].event_id) + " has somehow received a negative value of C2HR6")
                print(" value is :", C2HR6)
                return
        # end if()
        return
#      return
    # end CalculateC2QR6AndFirstHitVertexZ

    # #############################
    # Actually add the C2QR6 and
    # position of the first hit from the cleaned
    # pulse series to the frame
    # ##############################

    tray.AddModule( CalculateC2QR6AndFirstHitVertexZ, "CalcC2QR6AndFirstHitVertexZ", PulseSeries = Pulses, If = lambda f: PassedDCFilter(f, year=year))

    # ##############################
    # Rewrite the NAbove method from the
    # slc-veto project into python so I don't
    # have to faff about with a code review.
    # ##############################

    def CalculateNAbove200( frame, PulseSeries, ConfigID):
      #if If(frame):
        # Loop over all the triggers looking
        # for the times of all the selected triggers

        triggerTimes = []
        for trigger in frame[ "I3TriggerHierarchy"]:
            if trigger.key.config_id == ConfigID:
                triggerTimes.append( trigger.time)
            # end if()
        # end for()

        # Check that there is at least one of the
        # selected triggers.

        if len(triggerTimes) == 0:
            frame[ "QAbove200"] = dataclasses.I3Double( 0)
            frame[ "NchAbove200"] = dataclasses.I3Double( 0)
            return
        # end if()

        triggerTimes.sort()
        earliestTriggerTime = triggerTimes[0]

        chargeCounter = 0
        hitCounter = 0.0
        geometry = frame[ "I3Geometry"]
        try:
          for omkey in frame[ PulseSeries]:
            isHit=0.0
            if not geometry.omgeo[ omkey[0]].position.z > -200 * I3Units.m:
                continue
            # end if()
            for pulse in omkey[1]:
                if (pulse.time - earliestTriggerTime) < 0 and (pulse.time - earliestTriggerTime) > -2000:
                    chargeCounter += pulse.charge
                isHit = 1.0
            hitCounter+=isHit
        except TypeError:
          for omkey in frame[ PulseSeries].apply(frame):
            isHit=0.0
            if not geometry.omgeo[ omkey[0]].position.z > -200 * I3Units.m:
                continue
            # end if()
            for pulse in omkey[1]:
                if (pulse.time - earliestTriggerTime) < 0 and (pulse.time - earliestTriggerTime) > -2000:
                    chargeCounter += pulse.charge
                    isHit = 1.0
            hitCounter+=isHit
        frame[ "QAbove200"] = dataclasses.I3Double( chargeCounter)
        frame[ "NChAbove200"] = dataclasses.I3Double( hitCounter)
        return
    #  return
    # end CalculateNAbove200()

    # #############################
    # Actually add the NAbove200 variable
    # to the frame
    # ##############################

    tray.AddModule( CalculateNAbove200, "CalcNAbove200",
                    PulseSeries = splituncleaned,
                    ConfigID    = 1011,
                    If = lambda f: PassedDCFilter(f, year=year)
                    )

    # #############################
    # Jacob Daughhetee has shown that a ratio of 
    # SRT cleaned hits within DC to outside of 
    # DC can be a good bkg identifier as well.
    # Run the standard L2 SRT cleaning.
    # #############################


    tray.AddModule( "I3OMSelection<I3RecoPulseSeries>", "GenDCCFidSRTPulses",
                    selectInverse     = True,
                    InputResponse     = Pulses,
                    OutputResponse    = "SRTTWOfflinePulsesDCFid",
                    OutputOMSelection = "BadOM1",
                    OmittedKeys       = DOMList.DeepCoreFiducialDOMs,
                    If = lambda f: PassedDCFilter(f, year=year)
                    )

    tray.AddModule( "I3OMSelection<I3RecoPulseSeries>", "GenICVetoSRTPulses",
                    selectInverse     = False,
                    InputResponse     = Pulses,
                    OutputResponse    = "SRTTWOfflinePulsesICVeto",
                    OutputOMSelection = "BadOM2",
                    OmittedKeys       = DOMList.DeepCoreFiducialDOMs,
                    If = lambda f: PassedDCFilter(f, year=year)
                    )
    # ##############################
    # Suggestion by the WIMP group to
    # look at the RT cluster size of pulses
    # in the IC Veto region
    # ##############################

    classicSeededRTConfigService = I3DOMLinkSeededRTConfigurationService(
        treat_string_36_as_deepcore = False,
        allowSelfCoincidence    = True,
        useDustlayerCorrection  = False,
        dustlayerUpperZBoundary = 0*I3Units.m,
        dustlayerLowerZBoundary = -150*I3Units.m,
        ic_ic_RTTime           = 1000*I3Units.ns,
        ic_ic_RTRadius         = 250*I3Units.m
        )


    tray.AddModule( "I3StaticTWC<I3RecoPulseSeries>", "TWRTVetoPulses",
                    InputResponse    = splituncleaned,
                    OutputResponse   = "TWRTVetoSeries",
                    TriggerConfigIDs = [1011],
                    TriggerName      = "I3TriggerHierarchy",
                    WindowMinus      = 5000,
                    WindowPlus       = 0,
                    If = lambda f: PassedDCFilter(f, year=year)
                    )

    tray.AddModule( "I3OMSelection<I3RecoPulseSeries>", "GenRTVetoPulses",
                    selectInverse     = False,
                    InputResponse     = "TWRTVetoSeries",
                    OutputResponse    = "TWRTVetoSeries_ICVeto",
                    OutputOMSelection = "BadOM3",
                    OmittedKeys       = DOMList.DeepCoreFiducialDOMs,
                    If = lambda f: PassedDCFilter(f, year=year)
                    )

    tray.AddModule("I3RTVeto_RecoPulseMask_Module", "rtveto",
                   STConfigService         = classicSeededRTConfigService,
                   InputHitSeriesMapName   = "TWRTVetoSeries_ICVeto",
                   OutputHitSeriesMapName  = "RTVetoSeries250",
                   Streams                 = [icetray.I3Frame.Physics],
                   If = lambda f: PassedDCFilter(f, year=year)
                  )

    #tray.AddModule( "I3SeededRTHitCleaningModule<I3RecoPulse>", "RTVeto250",
    #                InputResponse  = "TWRTVetoSeries_ICVeto",
    #                OutputResponse = "RTVetoSeries250",
    #                Seeds          = "OneHitAutoSeed",
    #                RTRadius       = 250*I3Units.m,
    #                If = lambda f: PassedDCFilter(f, year=year)
    #                )

    def CountRTVetoSeriesNChannel( frame ):
#      if If(frame):
        totalCharge = 0
        totalHits = 0.0
        if frame.Has("RTVetoSeries250"):
            for omkey in frame["RTVetoSeries250"].apply(frame):
                totalHits+=1.0
                for pulse in omkey[1]:
                    totalCharge += pulse.charge
                # end for()
            # end for()
        # end if()
        frame["RTVetoSeries250PE"] = dataclasses.I3Double( totalCharge)
        frame["RTVetoSeries250Hits"] = dataclasses.I3Double( totalHits)
        return
 #     return
    # end CountRTVetoSeriesNChannel()

    tray.AddModule( CountRTVetoSeriesNChannel, "CountRTVetoSeries",If = lambda f: PassedDCFilter(f, year=year))

    # ##############################
    # Create a variable that gets pushed
    # to the frame that checks whether
    # the event passed the IC2011 LE L3
    # straight cuts and cleaning.
    # ##############################

    def passIC2018_LE_L3(frame):

        IC2012_LE_L3_FrameObjects = set([ "DCFilterPulses_VetoPE",
                                      "MicroCountHits",
                                      "MicroCountPE",
                                      "QAbove200",
                                      "NChAbove200", 
                                      "NoiseEngine_bool",
                                      "NoiseEngineNoCharge_bool", 
                                      "C2QR6",
                                      "C2HR6",
                                      "RTVetoSeries250PE",
                                      "RTVetoSeries250Hits",
                                      "SRTTWOfflinePulsesDCFid",
                                      "SRTTWOfflinePulsesICVeto",
                                      "VertexGuessZ",
                                      "N_LIDErrata",
                                      "SLOP_passed_bool",
                                      "Data_quality_bool"
                                      ])

        # Check all expected frame objects are found
        frameObjects = set( frame.keys() )
        missingFrameObjects = IC2012_LE_L3_FrameObjects.difference(frameObjects)
        assert len(missingFrameObjects) == 0, "%s are missing from the frame\n%s" % ( list(missingFrameObjects), str(frame) )

        # Count the number of SRT cleaned PEs in the DeepCore fiducial and
        # IceCube Veto region for use in the Ratio Cut variable. Also,
        # count the number of PE for hits satisying the DeepCore Filter
        # `speed of light' criteria.

        totalChargeFiducial = 0
        totalChargeVeto     = 0
        
        totalHitsFiducial   = len(frame["SRTTWOfflinePulsesDCFid"])            
        for omkey in frame["SRTTWOfflinePulsesDCFid"]:
            for pulse in omkey[1]:
                totalChargeFiducial += pulse.charge
            # end for()
        # end for()
        
        totalHitsVeto       = len(frame["SRTTWOfflinePulsesICVeto"])
        for omkey in frame["SRTTWOfflinePulsesICVeto"]:
            for pulse in omkey[1]:
                totalChargeVeto += pulse.charge
            # end for()
        # end for()

        LE_L3_Vars = dataclasses.I3MapStringDouble()
        LE_L3_Vars["N_LIDErrata"]        = frame["N_LIDErrata"].value
        LE_L3_Vars["SLOP_passed"]        = frame["SLOP_passed_bool"].value
        LE_L3_Vars["Data_quality"]       = frame["Data_quality_bool"].value
        LE_L3_Vars["NoiseEngine"]        = frame["NoiseEngine_bool"].value
        LE_L3_Vars["NoiseEngineNoCharge"]= frame["NoiseEngineNoCharge_bool"].value
        LE_L3_Vars["STW9000_DTW175PE"]   = frame["MicroCountPE"].get("STW9000_DTW175")
        LE_L3_Vars["STW9000_DTW200PE"]   = frame["MicroCountPE"].get("STW9000_DTW200")
        LE_L3_Vars["STW9000_DTW250PE"]   = frame["MicroCountPE"].get("STW9000_DTW250")
        LE_L3_Vars["STW9000_DTW300PE"]   = frame["MicroCountPE"].get("STW9000_DTW300")
        LE_L3_Vars["STW9000_DTW175Hits"] = frame["MicroCountHits"].get("STW9000_DTW175")
        LE_L3_Vars["STW9000_DTW200Hits"] = frame["MicroCountHits"].get("STW9000_DTW200")
        LE_L3_Vars["STW9000_DTW250Hits"] = frame["MicroCountHits"].get("STW9000_DTW250")
        LE_L3_Vars["STW9000_DTW300Hits"] = frame["MicroCountHits"].get("STW9000_DTW300")
        LE_L3_Vars["C2QR6"]              = frame["C2QR6"].value
        LE_L3_Vars["C2HR6"]              = frame["C2HR6"].value
        LE_L3_Vars["NAbove200PE"]        = frame["QAbove200"].value
        LE_L3_Vars["NAbove200Hits"]      = frame["NChAbove200"].value
        LE_L3_Vars["DCFiducialHits"]     = totalHitsFiducial
        LE_L3_Vars["ICVetoHits"]         = totalHitsVeto
        LE_L3_Vars["DCFiducialPE"]       = totalChargeFiducial
        LE_L3_Vars["ICVetoPE"]           = totalChargeVeto
        if totalHitsFiducial==0:
            # XXX: need to find a bit more elegant solution for this problem
            LE_L3_Vars['VetoFiducialRatioPE']  = float("inf")
            LE_L3_Vars['VetoFiducialRatioHits']= float("inf")
        else:  
            LE_L3_Vars['VetoFiducialRatioPE']= totalChargeVeto*1.0 /totalChargeFiducial
            LE_L3_Vars['VetoFiducialRatioHits']=totalHitsVeto*1.0/totalHitsFiducial
        LE_L3_Vars["CausalVetoPE"]       = frame["DCFilterPulses_VetoPE"].value
        LE_L3_Vars["CausalVetoHits"]     = frame["DCFilterPulses_VetoHits"].value
        LE_L3_Vars["VertexGuessZ"]       = frame["VertexGuessZ"].value
        LE_L3_Vars["VertexGuessY"]       = frame["VertexGuessY"].value
        LE_L3_Vars["VertexGuessX"]       = frame["VertexGuessX"].value
        LE_L3_Vars["RTVeto250PE"]        = frame["RTVetoSeries250PE"].value
        LE_L3_Vars["RTVeto250Hits"]      = frame["RTVetoSeries250Hits"].value
        LE_L3_Vars["NchCleaned"]         = len(frame["SRTTWOfflinePulsesDC"].apply(frame))
        LE_L3_Vars['CleanedFullTimeLength'] = ( frame[Pulses+'HitStatistics'].max_pulse_time - 
                                                frame[Pulses+'HitStatistics'].min_pulse_time)
        LE_L3_Vars['UncleanedFullTimeLength'] = ( frame[splituncleaned+'HitStatistics'].max_pulse_time - 
                                                 frame[splituncleaned+'HitStatistics'].min_pulse_time)          
        LE_L3_Vars['FullTimeLengthRatio']   = 1.*LE_L3_Vars['CleanedFullTimeLength']/LE_L3_Vars['UncleanedFullTimeLength'] 
        frame[L3_2018_VARS_KEY] = LE_L3_Vars
        # Calculation passing bools for RTVetoCut
        rt_veto_charge_pass  = ( (LE_L3_Vars["RTVeto250PE"] <  4.0 and LE_L3_Vars["DCFiducialPE"] <  100) or \
                                 (LE_L3_Vars["RTVeto250PE"] <  6.0 and LE_L3_Vars["DCFiducialPE"] >= 100 and LE_L3_Vars["DCFiducialPE"] < 150) or \
                                 (LE_L3_Vars["RTVeto250PE"] < 10.0 and LE_L3_Vars["DCFiducialPE"] >= 150 and LE_L3_Vars["DCFiducialPE"] < 200) or \
                                 (LE_L3_Vars["DCFiducialPE"] >= 200) )
        rt_veto_hit_pass     = ( (LE_L3_Vars["RTVeto250Hits"] <  4.0 and LE_L3_Vars["DCFiducialHits"] <  75) or \
                                 (LE_L3_Vars["RTVeto250Hits"] <  5.0 and LE_L3_Vars["DCFiducialHits"] >= 75 and LE_L3_Vars["DCFiducialHits"] < 100) or \
                                 (LE_L3_Vars["DCFiducialHits"] >= 100) )
        nch_pass             = ( LE_L3_Vars["NchCleaned"]  >= 5.9) ## just making sure that it is really >= 6           
        LE_L3_Vars["RTVetoCutCharge"]   = rt_veto_charge_pass
        LE_L3_Vars["RTVetoCutHit"]      = rt_veto_hit_pass
        
        # Calculating bools for L3 without RTVeto
        L3_charge_bool = (frame["NoiseEngine_bool"].value and 
                          frame["MicroCountHits"].get("STW9000_DTW300") > 2 and 
                          frame["MicroCountPE"].get("STW9000_DTW300") > 2 and 
                          frame["QAbove200"].value < 12 and 
                          totalChargeFiducial > 0 and 
                          frame["VertexGuessZ"].value < -120 and 
                          frame["DCFilterPulses_VetoPE"].value < 7.0 and 
                          LE_L3_Vars['VetoFiducialRatioPE'] < 1.5 and 
                          frame["C2QR6"].value > 0.4 ) 
        L3_hit_bool    = (frame["NoiseEngineNoCharge_bool"].value and 
                          frame["MicroCountHits"].get("STW9000_DTW300") > 2 and 
                          frame["NChAbove200"].value < 10 and 
                          totalHitsFiducial > 2 and 
                          frame["VertexGuessZ"].value < -120 and 
                          frame["DCFilterPulses_VetoHits"].value < 7.0 and 
                          LE_L3_Vars['VetoFiducialRatioHits'] < 1.5 and 
                          frame["C2HR6"].value > 0.37 and 
                          LE_L3_Vars['UncleanedFullTimeLength'] < 13000.0 and 
                          LE_L3_Vars['CleanedFullTimeLength'] < 5000.0 ) 

        # Get the data quality bool
        data_quality_bool = bool( LE_L3_Vars["Data_quality"] )

        ### Apply Track Energy Dependent Cuts // Send Failures to Expanded Fiducial Branch ### 
        frame["IC2012_LE_L3"]           = icetray.I3Bool(L3_charge_bool*rt_veto_charge_pass)
        frame["IC2012_LE_L3_No_RTVeto"] = icetray.I3Bool(L3_charge_bool)

        ## Storing all the 2018 L3 bools
        LE_L3_2018_bools =  dataclasses.I3MapStringBool()
        LE_L3_2018_bools["IC2018_LE_L3_Full"]             = (L3_hit_bool*rt_veto_hit_pass*nch_pass*data_quality_bool)
        LE_L3_2018_bools["IC2018_LE_L3_No_Nch"]           = (L3_hit_bool*rt_veto_hit_pass*data_quality_bool)
        LE_L3_2018_bools["IC2018_LE_L3_No_RTVeto"]        = (L3_hit_bool*nch_pass*data_quality_bool)
        LE_L3_2018_bools["IC2018_LE_L3_No_Nch_No_RTVeto"] = (L3_hit_bool*data_quality_bool)
        frame[L3_2018_ALL_BOOLS_KEY] = LE_L3_2018_bools
        if (not (frame["IC2012_LE_L3"].value or frame["IC2012_LE_L3_No_RTVeto"])) and rt_veto_charge_pass: 
            frame['ExpBranchL3Fodder'] = icetray.I3Bool(True)
        else: 
            frame['ExpBranchL3Fodder'] = icetray.I3Bool(False)

        # Store cut in oscNext format
        l3_cut = frame[L3_2018_ALL_BOOLS_KEY]["IC2018_LE_L3_Full"]
        frame[L3_CUT_BOOL_KEY] = icetray.I3Bool(l3_cut)


        return     

    tray.AddModule( passIC2018_LE_L3, "LE_L3_pass",If = lambda f: PassedDCFilter(f, year=year))

    # ############################################
    # Remove all the bad omselections created when
    # producing the MicroCount outputs as well as
    # other junk from L3 processing
    # ############################################

    tray.AddModule("Delete", "delete_badomselection",
                   Keys = ["BadOMSelection",
                           "BadOM1",
                           "BadOM2",
                           "BadOM3",
                           "BadOM_DCFid_0",
                           "BadOM_ICVeto_0",
                           "DCFidPulses_DTW175",
                           "DCFidPulses_DTW200",
                           "DCFidPulses_DTW250",
                           "DCFidPulses_DTW300",
                           "SplitInIcePulses_STW_NoiseEnginess",
                           "SplitInIcePulses_STW_ClassicRT_NoiseEnginess",
                           "SRTTWOfflinePulsesICVeto",
                           "SRTTWOfflinePulsesDCFid",
                           "DCFilterPulses_VetoPE",
                           "DCFilterPulses_VetoHits",
                           "DCFilterPulses",
                           "C2QR6",
                           "C2HR6",
                           "Baddies",
                           "RTVetoSeries250",
                           "RTVetoSeries250PE",
                           "RTVetoSeries250Hits", 
                           "TWRTVetoSeriesTimeRange",
                           "TWRTVetoSeries_ICVeto",
                           "OfflinePulsesDCFid",
                           "OfflinePulsesICVeto",
                           "VertexGuessX",
                           "VertexGuessY",
                           "VertexGuessZ",
                           "TWOfflinePulsesDCFid",
                           "NAbove200",
                           "MicroCountPE",
                           "MicroCountHits",
                           'SRTTWOfflinePulsesICVetoCleanedKeys',
                           'SRTTWOfflinePulsesDCFidCleanedKeys',
                           'TWOfflinePulsesDCFidCleanedKeys',
                           'TWRTVetoSeries_ICVetoCleanedKeys',
                           'OfflinePulsesDCFidCleanedKeys',
                           'OfflinePulsesICVetoCleanedKeys',
                           'N_LIDErrata',
                           'LIDErrata_osc_data_quality',
                           'SLOP_passed_bool'
                           ]
                   )


    return


@icetray.traysegment
def oscNext_L3( tray, name, 
                uncleaned_pulses,
                cleaned_pulses,
                year='12', # DeepCore filter year
                ):
    '''
    This is the main oscNext L3 tray segment

    Replaces DCL3MasterSegment
    '''

    # Only keep events passing DeepCore filter (L2)
    tray.AddModule(PassedDCFilter, year=year)

    # Apply L3 cuts
    tray.AddSegment( DeepCoreCuts, "L3DeepCoreCuts", splituncleaned=uncleaned_pulses, Pulses=cleaned_pulses, year=year, alttws=True)

