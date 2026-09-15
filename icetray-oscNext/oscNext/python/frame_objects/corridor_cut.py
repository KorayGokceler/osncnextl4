# Merged into "oscNext" project from the deprecated "veto_tools" sandbox project

from icecube import icetray, dataclasses
# from icecube import santa
from numpy import cos, sin, mod, sqrt, exp, sign, cross, pi, sqrt, zeros, abs, arange, array, arccos, isnan, mean
import os

class CorridorCut(icetray.I3ConditionalModule):

    def dgamma(self, z, uz, zc, dc, c, n_index):
        value = (n_index/sqrt(n_index**2 - 1))*sqrt(dc**2 + ((z-zc)**2)*(1-uz**2))
        return value

    def tgamma(self, z, uz, zc, dc, tc, tini, c, n_index):
        value = (tc) + ((z-zc)*uz+(n_index**2 - 1)*self.dgamma(z, uz, zc, dc, c, n_index)/n_index)/c
        return value

    def absval(self, a):
        if a.ndim > 1:
            if a.shape[1] == 3:
                a = a.T
        aux = 0
        for one_item in a:
            aux += one_item**2
        return sqrt(aux)
    
    def dist(self, x0, x1, x2):
        return self.absval(cross(x0 - x1, x0-x2))/self.absval(x2-x1)

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        
        # The default values are for direct hits coming from a track. But you can count also late hits by modifying them.
        # Suggestion: Use the default settings with DOM counts and find direct hits. For finding late hits, modify time window and switch to PE's. You might also want to increase the radius for the search.

        self.AddParameter('InputPulseSeries',
                          'Pulse series that will be used for hit search',
                          'OfflinePulses')
        self.AddParameter('NoiselessPulseSeries',
                          'Noise cleaned pulses',
                          'SRTTWOfflinePulsesDC')
        self.AddParameter('OutputPulseSeries',
                          'Name for the mask containing the causal hits',
                          'CorridorPulses')
        self.AddParameter('HitCounter',
                          'Name for the  counts output',
                          'CorridorCounter')
        self.AddParameter('OutputTrack',
                          'Name for the output track',
                          'CorridorCutTrack')
        self.AddParameter('Radius',
                          'Radius for the search',
                          75. * icetray.I3Units.m)
        self.AddParameter('WindowMinus',
                          '',
                          -75. * icetray.I3Units.ns)
        self.AddParameter('WindowPlus',
                          '',
                          250. * icetray.I3Units.ns)
        self.AddParameter('ZenithSteps',
                          'Size of the zenith step in radians',
                          0.02)
        self.AddParameter('ChargeThreshold',
                          'Charge threshold for the hit to be considered',
                          0.3)
        self.AddParameter('PEmode',
                          'Turn on to count PEs instead of DOMs',
                          False)
        self.AddOutBox('OutBox')

    def Geometry(self, gframe):
        geometry = gframe['I3Geometry']
        self.stringx = zeros(87)
        self.stringy = zeros(87)
        for i in range(1, 87):
            self.stringx[i] = geometry.omgeo[icetray.OMKey(i,1)].position.x
            self.stringy[i] = geometry.omgeo[icetray.OMKey(i,1)].position.y
        self.PushFrame(gframe)

        # Dictionary with all the relevant things
        self.dcstrings      = range(79, 87)
        self.stringdict     = {}
        self.vertexdict     = {}
        self.vertexdict_end = {}
        self.anglesdict     = {}
        for one_string in self.dcstrings:
            self.anglesdict[one_string] = self.azimuthAngles(one_string)
            self.stringsInRange(one_string)


    def stringsInRange(self, icstring):
        self.stringdict[icstring] = []
        for one_x2 in self.vertexdict_end[icstring]:
            strings_one_angle = []
            # Formula for line to point distance
            dist_to_track = (abs((one_x2[0] - self.vertexdict[icstring][0])*(self.vertexdict[icstring][1] - self.stringy) - 
                             (self.vertexdict[icstring][0]- self.stringx)*(one_x2[1] -  self.vertexdict[icstring][1]))
                             /sqrt((one_x2[0] - self.vertexdict[icstring][0])**2 + (one_x2[1] -  self.vertexdict[icstring][1])**2))

            dist_to_endpoint = sqrt((one_x2[0] - self.stringx)**2 + (one_x2[1] - self.stringy)**2)

            temp_strings = arange(0,87)

            edcbool = array([True]*87)
            edcbool[self.extendedDC] = False
            edcbool[0] = False
            # Removing the zeros I put in self.stringx
            mask = (dist_to_track < self.radius) * (dist_to_endpoint < 500.) * edcbool

            self.stringdict[icstring].append(temp_strings[mask])


    def getVertex(self, icstring):
        nodes = array([36 , 0, 0])
        if icstring == 81:
            enodes = [45, 46]
        elif icstring == 82:
            enodes = [46, 37]
        elif icstring == 83:
            enodes = [37, 27]
        elif icstring == 84:
            enodes = [27, 26]
        elif icstring == 85:
            enodes = [26, 35]
        elif icstring == 86:
            enodes = [45, 35]
        elif icstring == 79 or icstring == 80:
            return array([ self.stringx[icstring], self.stringy[icstring]])

        nodes[1:] = enodes

        x = self.stringx[nodes].mean()
        y = self.stringy[nodes].mean()

        return array([x, y])


    def azimuthAngles(self, icstring):
        # New corridors added
        if icstring == 81:
            pairs = [[31, 41], [75, 76], [72, 73], [40, 50], [6, 13], [1,2]]
        elif icstring == 82:
            pairs = [[75, 76], [73, 74], [50, 40], [21,30], [29,30], [30, 39], [6, 13],  [2, 3],  [41, 31],    [78, 72]]
        elif icstring == 83:
            pairs = [[73, 74], [40, 30], [21, 30], [21, 29], [29, 30], [6, 13],  [2, 3],   [31, 22], [75, 66],   [59, 50], [4, 5], [74, 67]]
        elif icstring == 84:
            pairs = [[30, 40], [5, 6],   [2, 3],   [14, 15], [31, 22], [68, 75], [73, 74],   [74, 67], [4, 5], [3, 4]]
        elif icstring == 85:
            pairs = [[6,13], [5,6],    [1,2],    [31,22],  [75, 68], [72, 73], [40, 30],   [59, 50]]
        elif icstring == 86:
            pairs = [[1,2],    [31, 41], [75, 68], [72, 73], [50, 40], [5, 6]]
        elif icstring == 79: # Using 84 + 85
            pairs = [[30, 40], [5, 6],   [2, 3],   [14, 15], [31, 22], [68, 75], [73, 74],   [74, 67], [4, 5], [3, 4], [6,13], [1,2], [72, 73], [59, 50]]
        elif icstring == 80: # Using 84 + 83
            pairs = [[30, 40], [5, 6],   [2, 3],   [14, 15], [31, 22], [68, 75], [73, 74],   [74, 67], [4, 5], [3, 4], [21, 30], [21, 29], [29, 30], [6, 13], [75, 66],  [59, 50]]

        angles = zeros(len(pairs))
        self.vertexdict_end[icstring] = zeros([len(pairs),2])

        for item, onepair in enumerate(pairs):
            x = self.stringx[onepair].mean()
            y = self.stringy[onepair].mean()
            self.vertexdict[icstring] = self.getVertex(icstring)
            self.vertexdict_end[icstring][item,:] = array([x,y])
            az_angle = arccos((x - self.vertexdict[icstring][0])/
                                 sqrt((y - self.vertexdict[icstring][1])**2 + (x - self.vertexdict[icstring][0])**2))

            if (y - self.vertexdict[icstring][1]) < 0: #left side
                az_angle = 2*pi - az_angle

            angles[item] = az_angle

        return angles


    def Configure(self):
        self.input_pulses = self.GetParameter('InputPulseSeries')
        self.out_pulses   = self.GetParameter('OutputPulseSeries')
        self.track        = self.GetParameter('OutputTrack')
        self.nchannel     = self.GetParameter('HitCounter')
        self.radius       = self.GetParameter('Radius')
        self.wplus        = self.GetParameter('WindowPlus')
        self.wminus       = self.GetParameter('WindowMinus')
        self.zenith_steps = self.GetParameter('ZenithSteps')
        self.extendedDC   = array([36, 35, 45, 46, 37, 27, 26, 36, 79, 80, 81, 82, 83, 84, 85, 86])
        self.noiseless_hs = self.GetParameter('NoiselessPulseSeries')
        self.charge_th    = self.GetParameter('ChargeThreshold')
        self.pemode       = self.GetParameter('PEmode')

    def Physics(self, frame):
        if not self.input_pulses in frame:
            #print 'CorridorCut: No pulse series with name: ' + self.input_pulses
            self.PushFrame(frame)
            return True     
        try:
            ic_pulses = frame[self.input_pulses].apply(frame) 
        except:
            ic_pulses = frame.Get(self.input_pulses)

        geometry = frame['I3Geometry']

        noiseless_map = frame[self.noiseless_hs]
        if hasattr(noiseless_map, "apply") :
            noiseless_map = noiseless_map.apply(frame)
        if len(noiseless_map) == 0:
            #print 'CorridorCut: Noiseless pulses not available in frame'
            self.PushFrame(frame)
            return True
            

        # Get the CoG of pulses, push it towards the closest DC string
        noiseless_doms    = zeros([len(noiseless_map),4]) #string, om, z, time
        for domindex, one_dom in enumerate(noiseless_map):
            if one_dom[0].string in range(79, 86):
                noiseless_doms[domindex,:] = [one_dom[0].string, one_dom[0].om, geometry.omgeo[one_dom[0]].position.z, one_dom[1][0].time]
            else:
                noiseless_doms[domindex,:] = [0., 0., 0., 10E10]
    
        # Pick the earliest one
        arrival_times = noiseless_doms[:,3]
        main_string = noiseless_doms[arrival_times.argmin(),0]
        cogz = mean(noiseless_doms[noiseless_doms[:,0] == main_string, 2])
        tini = mean(noiseless_doms[noiseless_doms[:,0] == main_string, 3])
        
        '''
        cogx = noiseless_doms[:,0].mean()
        cogy = noiseless_doms[:,1].mean()
        dist_to_hqstrings = sqrt((cogx - self.stringx[79:87])**2 + (cogy - self.stringy[79:87])**2)
        main_string = 79 + dist_to_hqstrings.argmin()
        '''
        #print 'CC: New frame -  main string ', main_string
        #print noiseless_doms
        #raw_input()
        '''
        if sqrt((cogx - self.stringx[36])**2 + (cogy - self.stringy[36])**2) > 180:
            #print 'CorridorCut: CoG is too far from DC strings, skipping'
            frame[self.nchannel] = dataclasses.I3Double(0.)
            self.PushFrame(frame)
            return
        '''
        #cogz = noiseless_doms[:,2].mean()
        #tini = noiseless_doms[:,3].mean()
        #print 'CC: COGs: ', cogz, tini
        
        if cogz == 0:
            #print 'CorridorCut: No SANTA hits in reduced DC strings, no corridors defined, putting empty variables and skipping'
            new_pulse_map = dataclasses.I3RecoPulseSeriesMapMask(frame, self.input_pulses)
            new_pulse_map.set_none()
            output_track = dataclasses.I3Particle()
            output_track.fit_status = dataclasses.I3Particle.InsufficientHits  
            frame[self.track] = output_track
            frame[self.out_pulses] = new_pulse_map
            frame[self.nchannel] = dataclasses.I3Double(0.)
            self.PushFrame(frame)
            return     
        
        # Put all of the pulses in a huge numpy array. If several in one dom, take earliest one
        final_zenith  = None
        final_azimuth = None
        max_counts    = 0
        mc_error      = 9e10

        # The difference between PE mode and DOM mode is here
        if self.pemode:
            # In PE mode, every single hit goes to the list, there is no threshold (empty hits get 0 charge)
            pulses = 0
            for onedom in ic_pulses:
                for onehit in onedom[1]:
                    pulses += 1
            allpulses = zeros([pulses,8])
            pindex = 0
            for onedom in ic_pulses:
                for onehit in onedom[1]:
                    # Search for the first hit above the threshold, stop there, everything gets a weight 1
                    try:
                        allpulses[pindex] = [onedom[0].string, onedom[0].om, self.stringx[onedom[0].string], self.stringy[onedom[0].string], 
                                    geometry.omgeo[onedom[0]].position.z, onehit.time, onehit.charge, 0.]
                    except:
                        allpulses[pindex] = [onedom[0].string, onedom[0].om, self.stringx[onedom[0].string], self.stringy[onedom[0].string], 
                                    geometry.omgeo[onedom[0]].position.z, 10e10, 0,  0.]     
                    pindex += 1
                        
        # In DOM mode a dom can only be there once, and it takes the time of the first hit above threshold
        # Go through the first pulse in each dom, take the first one above threshold
        else:
            allpulses = zeros([len(ic_pulses),8])
            for index, onedom in enumerate(ic_pulses):
                take_this_hit = False
                hit_time      = 10e10
                charge        = 0
                for onehit in onedom[1]:
                    # Search for the first hit above the threshold, stop there, everything gets a weight 1
                    if onehit.charge > self.charge_th:
                        take_this_hit = True
                        hit_time = onehit.time
                        charge = 1
                        break
                # There are some hits in which wavedeform returns an empty hit, so we give them absurd times and charges, removed by next step
                allpulses[index] = [onedom[0].string, onedom[0].om, self.stringx[onedom[0].string], self.stringy[onedom[0].string], 
                                        geometry.omgeo[onedom[0]].position.z, hit_time, charge, 0.]

        # The rest of the code behaves the same
        allpulses = allpulses[allpulses[:,6] != 0, :]


        # For each event, one main string exists
        for az_index, azimuth in enumerate(self.anglesdict[main_string]):
            inrange_mask    = array([False]*allpulses.shape[0])
            for p_index in range(0,allpulses.shape[0]):
                if allpulses[p_index,0] in self.stringdict[main_string][az_index]:
                    inrange_mask[p_index] = True

            pulses_in_range =  allpulses[inrange_mask,:]
            #print 'AZIMUTH ', azimuth, ' - In range: ', pulses_in_range[:, 0:2]
            for zenith in arange(0.05, 1.2, self.zenith_steps):

                # Calculate the distance of each hit to the track hypothesis
                x1 = zeros([1,3])
                x1[0,0:2] = self.vertexdict[main_string]
                x1[0,2]   = cogz
                x2 = zeros([1,3])
                x2[0,0:3] = [sin(zenith)*cos(azimuth), sin(zenith)*sin(azimuth), cos(zenith)]
                x2 = x1+x2*600.
              
                pulses_dist = self.dist(pulses_in_range[:,2:5], x1, x2)
                #print azimuth, zenith, ' - dist - ', pulses_dist
                # Transforming coordinates
                new_theta = zenith - pi/2
                new_phi = mod((azimuth + pi), 2*pi)

                # Direction
                ux = cos(new_theta)*cos(new_phi)
                uy = cos(new_theta)*sin(new_phi)
                uz = sin(new_theta)

                # Point of closest approach
                zc = (cogz - 
                      uz*(self.vertexdict[main_string][0]*ux + self.vertexdict[main_string][1]*uy + cogz*uz) + 
                      uz*(pulses_in_range[:,2]*ux + pulses_in_range[:,3]*uy))/(1-uz**2)

                tc =  tini + (pulses_in_range[:,2]*ux + pulses_in_range[:,3]*uy + cogz*uz - 
                                     (self.vertexdict[main_string][0]*ux + self.vertexdict[main_string][1]*uy + 
                                      cogz*uz))/ ( dataclasses.I3Constants.c*(1-uz**2))

                dc = sqrt((self.vertexdict[main_string][0] + dataclasses.I3Constants.c*ux*(tc - tini) - pulses_in_range[:,2])**2 + 
                          (self.vertexdict[main_string][1] + dataclasses.I3Constants.c*uy*(tc - tini) - pulses_in_range[:,3])**2)

                tarrival = self.tgamma(pulses_in_range[:,4] , uz, zc, dc, tc, tini, dataclasses.I3Constants.c, dataclasses.I3Constants.n_ice)

                timeres = tarrival - pulses_in_range[:,5]
                #print 'Timeres - ', timeres
                # The 130 is hardcoded to take only the corridors (and not second layers)
                mask    =  (self.wminus < timeres)& (timeres < self.wplus) & (pulses_dist < 150.)

                # Summing the "charge" field, different for PE and DOM mode
                track_counts = sum(pulses_in_range[mask,6])
                distpass = pulses_dist[mask]
                
                if distpass.shape[0] > 0:
                    #print 'Found something: ', track_counts, ',  at: ', zenith, azimuth
                    tr2 = sum(distpass**2)/distpass.shape[0]
                    if track_counts > max_counts or (track_counts == max_counts and tr2 < mc_error):
                        max_counts    = track_counts
                        final_zenith  = zenith
                        final_azimuth = azimuth
                        mc_error      = tr2
                        veto_hits     = pulses_in_range[mask,0:2]
            #print 'Azimuth done!'
            #raw_input()
        ### Creating empty pulses and particle
        new_pulse_map = dataclasses.I3RecoPulseSeriesMapMask(frame, self.input_pulses)
        new_pulse_map.set_none()
        output_track = dataclasses.I3Particle()
        output_track.fit_status = dataclasses.I3Particle.InsufficientHits  
        ###      
        if max_counts > 0:
            output_track.pos.x = self.vertexdict[main_string][0]
            output_track.pos.y =  self.vertexdict[main_string][1]
            output_track.pos.z = cogz
            output_track.shape = dataclasses.I3Particle.InfiniteTrack
            #output_track.dir.set_direction(final_zenith, final_azimuth)
            output_track.dir = dataclasses.I3Direction(final_zenith, final_azimuth)
            output_track.fit_status = dataclasses.I3Particle.OK
            output_track.time  = tini

            for one_hit in veto_hits:
                new_pulse_map.set(icetray.OMKey(int(one_hit[0]), int(one_hit[1])), True)
            
            #print 'Something was found!'
            #raw_input()
 
        frame[self.track] = output_track
        frame[self.out_pulses] = new_pulse_map
        frame[self.nchannel] = dataclasses.I3Double(max_counts)
        self.PushFrame(frame)
