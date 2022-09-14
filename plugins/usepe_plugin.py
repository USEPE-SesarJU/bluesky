""" This plugin load the graph for the USEPE project. """
# Import the global bluesky objects. Uncomment the ones you need
from os import listdir
from os.path import join
from pathlib import Path
import configparser
import copy
import datetime
import math
import os
import pickle
import time

from bluesky import core, traf, stack, sim  # , settings, navdb,  scr, tools
from bluesky.tools import geo
from bluesky.tools.aero import nm
from bluesky.traffic.asas.detection import ConflictDetection
from bluesky.traffic.asas.statebased import StateBased
from usepe.city_model.dynamic_segments import dynamicSegments
from usepe.city_model.scenario_definition import createFlightPlan, createDeliveryFlightPlan, createSurveillanceFlightPlan
from usepe.city_model.strategic_deconfliction import initialPopulation, deconflictedPathPlanning, deconflictedDeliveryPathPlanning, deconflictedSurveillancePathPlanning
from usepe.city_model.utils import read_my_graphml, layersDict, wpt_bsc2wpt_graph, nearestNode3d
from usepe.segmentation_service.segmentation_service import segmentationService
import geopandas as gpd
import numpy as np
import pandas as pd


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2022'


active = False
usepeconfig = None
usepegraph = None
usepesegments = None
usepestrategic = None
usepeflightplans = None
usepedronecommands = None
usepewind = None
updateInterval = 1.0


# ## Initialisation function of your plugin. Do not change the name of this
# ## function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''

    # Configuration parameters
    config = {
        'plugin_name': 'USEPE',
        'plugin_type': 'sim',
        'update_interval': updateInterval,

        # The update function is called after traffic is updated.
        'update': update,

        # The preupdate function is called before traffic is updated.
        'preupdate': preupdate,

        # Reset
        'reset': reset }

    stackfunctions = {
        'USEPE': [
            'USEPE CONFIG/ON/OFF, [config_path]',
            'txt, [word]',
            usepe,
            'Set path to configuration file, or turn on/off the plugin.'
        ]
    }

    # init_plugin() should always return a configuration dict.
    return config, stackfunctions


def update():
    if sim.simt % 600 == 0:
        print( 'Simulation time: {}'.format( sim.simt ) )
    if active:
        if sim.simt > usepeconfig.getint( 'BlueSky', 'final_time' ):
            stack.stack( 'RESET' )
            stack.stack( 'QUIT' )

        usepegraph.update()
        usepesegments.update()
        usepestrategic.update()
        usepeflightplans.update()
        usepedronecommands.update()
    return


def preupdate():
    if active:
        usepegraph.preupdate()
        usepesegments.preupdate()
        usepestrategic.preupdate()
        usepeflightplans.preupdate()
        usepedronecommands.preupdate()
    return


def reset():
    if active:
        usepegraph.reset()
        usepesegments.reset()
        usepestrategic.reset()
        usepeflightplans.reset()
        usepedronecommands.reset()
    return


def usepe( cmd, args='' ):
    ''' USEPE command for the plugin
        Options:
        CONFIG: Set the configuration file, and initialise the various parts of the plugin.
        ON: Activate the plugin.
        OFF: Deactivate the plugin.
    '''

    global active
    global usepeconfig

    global usepegraph
    global usepesegments
    global usepeflightplans
    global usepestrategic
    global usepedronecommands
    global usepewind

    if cmd == 'CONFIG':
        if args == '':
            return False, f'"USEPE CONFIG" needs a valid path to configuration file.'

        config_path = args
        usepeconfig = configparser.ConfigParser()
        usepeconfig.read( config_path )

        graph_path = usepeconfig['BlueSky']['graph_path']
        flight_plan_csv_path = usepeconfig['BlueSky']['flight_plan_csv_path']

        initial_time = int( usepeconfig['BlueSky']['initial_time'] )
        final_time = int( usepeconfig['BlueSky']['final_time'] )

        usepegraph = UsepeGraph( graph_path )
        usepesegments = UsepeSegments()
        usepestrategic = UsepeStrategicDeconfliction( initial_time, final_time )
        usepeflightplans = UsepeFlightPlan( flight_plan_csv_path )
        usepedronecommands = UsepeDroneCommands()
        usepewind = UsepeWind()

        return True, f'The configuration file has been set.'

    elif cmd == 'ON':
        if usepeconfig is not None:
            # Activate the detection and resolution method, and logger
            # configuration_path = r"{}".format( usepeconfig['BlueSky']['configuration_path'] )
            # stack.stack( 'PCALL {} REL'.format( configuration_path ) )
            stack.stack( 'OP' )
            active = True
            return True, f'USEPE Plugin is now active'
        else:
            return False, f'The configuration file is not provided. First use "USEPE CONFIG config_path"'

    elif cmd == 'OFF':
        active = False
        return True, f'USEPE Plugin is now inactive'

    else:
        return False, f'Available commands are: CONFIG, ON, OFF'


class UsepeGraph( core.Entity ):
    ''' UsepeGraph new entity for BlueSky
    This class reads the graph that represents the city.
    '''

    def __init__( self, graph_path ):
        super().__init__()

        # self.graph = graph_path
        self.graph = read_my_graphml( graph_path )
        self.layers_dict = layersDict( usepeconfig )

    def update( self ):  # Not used
        # stack.stack( 'ECHO Example update: creating a graph' )
        return

    def preupdate( self ):  # Not used
        # print( self.graph.nodes() )
        return

    def reset( self ):  # Not used
        return


class UsepeSegments( core.Entity ):
    ''' UsepeSegments new entity for BlueSky
    This class contains the segments information.
    The initial set of segments is loaded.
    When the segments are updated, this class has methods to update: i)graph; ii) routes in the tactical phase;
    iii) routes in the strategic phase
    '''

    def __init__( self ):
        super().__init__()

        # with open( segment_path, 'rb' ) as f:
        #    self.segments = pickle.load( f )

        # Initialise class of dynamic segmentation - it provides the initial segments
        # Include the region for input
        self.region = "Hannover"

        self.segmentation_service = segmentationService( self.region )
        self.segmentation_service.export_cells()  # export .json file to "./data/examples"

        self.segments = pd.read_json( 'usepe/segmentation_service/data/examples/' + self.region + '.json', orient="records", lines=True )

        if not usepeconfig.getboolean( 'BlueSky', 'D2C2' ):
            self.referenceSegments()
        else:
            self.addCorridorSegments()

        usepegraph.graph, self.segments = dynamicSegments( usepegraph.graph, usepeconfig, self.segments, deleted_segments=None )

        self.wpt_dict = {}
        self.wpt_bsc = {}
        self.strategic_wind_updated = False

        # list with polygons printed: red segments
        self.poly_printed = []

        with self.settrafarrays():
            self.recentpath = np.array( [], dtype=np.ndarray )

        self.printRedSegments()

    def referenceSegments( self ):
        ref_speed = 20
        ref_capacity = 999
        reference_segments = self.segments.copy()
        reference_segments['class'] = reference_segments['class'].apply( lambda x: x if x == 'black' else 'white' )
        reference_segments['speed_max'] = reference_segments['class'].apply( lambda x: 0 if x == 'black' else ref_speed )
        reference_segments['capacity'] = reference_segments['class'].apply( lambda x: 0 if x == 'black' else ref_capacity )

        self.segments = reference_segments

    def addCorridorSegments( self ):
        # active_corridors = [1, 2, 3, 4]
        active_corridors = usepeconfig['Corridors']['corridors'].split( ' ' )

        for cor in active_corridors:
            name1 = 'COR{}'.format( str( cor ) )
            name2 = 'COR{}r'.format( str( cor ) )

            class_ = 'COR'
            lat_min = 0
            lat_max = 0
            lon_min = 0
            lon_max = 0
            z_min = 0
            z_max = 0
            speed_min = 0
            speed_max = int( usepeconfig['Corridors']['speed'] )
            capacity = 99
            occupancy = 0
            geovect = 'NSEW'
            parent = None
            new = False
            updated = False

            if name1 not in self.segments.index:
                self.segments.loc[name1] = [class_, lat_min, lat_max, lon_min, lon_max, z_min, z_max, speed_min,
                                            speed_max, capacity, occupancy, geovect, parent, new, updated]
            if name2 not in self.segments.index:
                self.segments.loc[name2] = [class_, lat_min, lat_max, lon_min, lon_max, z_min, z_max, speed_min,
                                            speed_max, capacity, occupancy, geovect, parent, new, updated]


    def create( self, n=1 ):
        super().create( n )

        path_recording_time = 300  # sec

        positions = math.ceil( path_recording_time / updateInterval )
        self.recentpath[-n:] = [np.empty( positions, dtype=tuple ) for _ in range( n )]

    def printRedSegments( self ):
        df_red_cell = self.segmentation_service.cells.loc[self.segmentation_service.cells["class"] == 'red']

        for pol in self.poly_printed:
            stack.stack( 'DEL pol{}'.format( pol ) )

        self.poly_printed = []

        for row in df_red_cell.iterrows():
            # print( 'POLY '.format( row[0], row[1]["geometry"] ) )
            try:
                shapelist = list( row[1]["geometry"].exterior.coords )
            except AttributeError:
                shapelist = list( row[1]["geometry"].coords )
            # print( shapelist )  # list of tuples
            aux = []
            for el in shapelist:
                aux.append( ( el[1], el[0] ) )
            # print( aux )
            aux = str( aux ).replace( "[", "" ).replace( "]", "" ).replace( "(", "" ).replace( ")", "" ).replace( ",", "" )
            # print( aux )
            # print( 'POLY pol{} {}'.format( row[0], aux ) )
            stack.stack( 'POLY pol{} {}'.format( row[0], aux )
                         )

            self.poly_printed += [row[0]]

    def dynamicSegments( self ):
        """
        TODO. Here we have to include the function which updates the segments
        """
        updated = False
        if not usepeconfig.getboolean( 'BlueSky', 'D2C2' ):
            return updated, self.segments

        update_interval = 300  # sec

        if sim.simt % update_interval == 0:
            '''Inputs provided for the update rules'''
            # waypoints for each drone + graph
            # usepeflightplans.route_dict  # dictionary containing the waypoints of each drone flying. Key - drone id, values - list of waypoints id
            # usepegraph.node  # dict containng the features of each waypoint key - waypoint id, value dict with features and its values
            # drone ids
            # traf.id

            # Go through all conflict pairs and sort the IDs for easier matching
            currentconf = [tuple( sorted( pair ) ) for pair in traf.cd.confpairs_unique]  # pairs of drones in conflict AT THIS MOMENT
            # historic conflicts?

            # for each pair in conflict, the latitude, longitude and altitude of the Closest Point of Approach (CPA)
            currentconf_loc = []
            for pair in currentconf:
                pair_index = traf.cd.confpairs.index( pair )
                drone_1_index = traf.id.index( pair[0] )

                dist_to_cpa = traf.cd.dcpa[pair_index]  # check units

                lat_cpa = traf.lat[drone_1_index] + ( dist_to_cpa * math.sin( traf.hdg[drone_1_index] * math.pi / 180 ) * 90 / 1E7 )
                lon_cpa = traf.lon[drone_1_index] + ( dist_to_cpa * math.cos( traf.hdg[drone_1_index] * math.pi / 180 ) * 90 / ( 1E7 * math.cos( traf.lat[drone_1_index] ) ) )
                alt_cpa = traf.alt[drone_1_index]

                currentconf_loc.append( ( lat_cpa, lon_cpa, alt_cpa ) )

            # for each pair in conflict, headings of each drone
            currentconf_hdg = []
            for pair in currentconf:
                pair_index = traf.cd.confpairs.index( pair )
                drone_1_index = traf.id.index( pair[0] )
                drone_2_index = traf.id.index( pair[1] )
                currentconf_hdg.append( ( traf.hdg[drone_1_index], traf.hdg[drone_2_index] ) )

            # value of the conflict frequency threshold, e.g., 1 conflict / (km^2 * hour)
            # usepeconfig['Segmentation']['conflict_threshold']

            # external file (csv, txt, cfg) that provides: area definition, event start time, event end time

            '''Update rules'''

            # WIND #
            wind_file = usepeconfig['Segmentation_service']['wind_path']
            if wind_file:
                if not self.strategic_wind_updated:
                    self.segmentation_service.update_wind_strat( wind_file, False )  # strategic update rules based on wind data
                    self.strategic_wind_updated = True
                self.segmentation_service.update_wind_tact( usepeflightplans.route_dict,
                                                        self.recentpath,
                                                        None,
                                                        traf.id,
                                                        usepegraph.graph )  # tactical update rules
            # else:
            #    print("No wind path specified, skipping all wind simulation!")

            # TRAFFIC #

            self.segmentation_service.update_traffic_strat( self.recentpath )
            self.segmentation_service.update_traffic_tact( self.recentpath )

            # CONFLICTS #
            self.segmentation_service.update_conflict( currentconf, currentconf_loc,
                                                       currentconf_hdg, update_interval )

            updated = True
            self.segmentation_service.export_cells()  # export .json file to "./data/examples"
            self.segments = pd.read_json( 'usepe/segmentation_service/data/examples/' + self.region + '.json', orient="records", lines=True )

            self.addCorridorSegments()
            print( 'Segments update completed.' )

        # The event rule is continuous
        self.segmentation_service.update_event( 'event', sim.simt )  # event update rule
        if self.segmentation_service.event_update:
            updated = True
            self.segmentation_service.event_update = False
            self.segmentation_service.export_cells()  # export .json file to "./data/examples"
            self.segments = pd.read_json( 'usepe/segmentation_service/data/examples/' + self.region + '.json', orient="records", lines=True )

            self.addCorridorSegments()

        segments = self.segments

        return updated, segments

    def calcRecentPath( self ):

        # drone path
        for i in range( self.recentpath.size ):

            acrte = traf.ap.route[i]
            iactwp = acrte.iactwp
            # actwpt = acrte.wpname[iactwp]
            # wpt_dict = self.wpt_dict[traf.id[i]]
            if acrte.wpalt[iactwp] < 0:
                actwpt_alt = traf.ap.alt[i]
            else:
                actwpt_alt = acrte.wpalt[iactwp]

            actwpt = nearestNode3d( usepegraph.graph,
                                    acrte.wplon[iactwp],
                                    acrte.wplat[iactwp],
                                    actwpt_alt,
                                    exclude_corridor=False )

            if iactwp == 0:
                i_prevactwpt = iactwp
            else:
                i_prevactwpt = iactwp - 1
                # prev_actwpt = list( wpt_dict.keys() )[list( wpt_dict.keys() ).index( acrte.wpname[iactwp] ) - 1]
            if acrte.wpalt[i_prevactwpt] < 0:
                actwpt_alt = traf.ap.alt[i]
            else:
                actwpt_alt = acrte.wpalt[i_prevactwpt]
            prev_actwpt = nearestNode3d( usepegraph.graph,
                                         acrte.wplon[i_prevactwpt],
                                         acrte.wplat[i_prevactwpt],
                                         acrte.wpalt[i_prevactwpt],
                                         exclude_corridor=False )

            temparr = np.empty_like( self.recentpath[i] )
            #=======================================================================================
            # currentpos = ( sim.simt, traf.lat[i], traf.lon[i], traf.alt[i],
            #                wpt_dict[acrte.wpname[iactwp]], wpt_dict[prev_actwpt] )
            #=======================================================================================

            currentpos = ( sim.simt, traf.lat[i], traf.lon[i], traf.alt[i],
                           actwpt, prev_actwpt )
            temparr[-1] = currentpos
            temparr[:-1] = self.recentpath[i][1:]
            self.recentpath[i][:] = temparr

    def calcWptDict( self ):

        for acid in traf.id:

            i = traf.id2idx( acid )
            acrte = traf.ap.route[i]
            wpt_bsc = acrte.wpname

            # If a new drone is created: update dict
            if acid not in self.wpt_dict:
                # Add entry in the dict
                wpt_route_graph = usepeflightplans.route_dict[acid]
                wpt_dict_acid = wpt_bsc2wpt_graph( acrte.wpname, wpt_route_graph )
                self.wpt_dict[acid] = wpt_dict_acid
            else:
                # If the list of waypoints in bluesky changes: update dict
                if wpt_bsc != self.wpt_bsc[acid]:
                    wpt_route_graph = usepeflightplans.route_dict[acid]
                    wpt_dict_acid = wpt_bsc2wpt_graph( acrte.wpname, wpt_route_graph )
                    self.wpt_dict[acid] = wpt_dict_acid
                else:
                    # print( 'No change wpt_dict' )
                    continue

            self.wpt_bsc[acid] = wpt_bsc


    def update( self ):  # Not used
        # stack.stack( 'ECHO Example update: import segments' )
        return

    def preupdate( self ):
        updated, self.segments = self.dynamicSegments()

        if updated:
            # TODO: Perform all the activities associated to the segmetns update
            df = self.segments[( self.segments['updated'] == True ) | ( self.segments['new'] == True )]
            if df.empty:
                pass
            else:
                # Print red segments
                self.printRedSegments()

                # 1st:  to update the graph
                usepegraph.graph, self.segments = dynamicSegments( usepegraph.graph, usepeconfig, self.segments, deleted_segments=None )

                # 2nd: to initialised the population of segments
                usepestrategic.initialisedUsers()

                # segments_df = pd.DataFrame.from_dict( self.segments, orient='index' )
                segments_df = self.segments
                # 3rd. To update the drones that are already flying
                for acid in traf.id:
                    print( acid )
                    idx = traf.id2idx( acid )

                    acrte = traf.ap.route[idx]
                    iactwp = acrte.iactwp
                    if iactwp >= ( len( acrte.wpname ) - 3 ):
                        continue
                    lat0 = acrte.wplat[iactwp]
                    lon0 = acrte.wplon[iactwp]
                    alt0 = acrte.wpalt[iactwp]

                    if alt0 < 0:
                        alt0 = traf.alt[idx]

                    latf = acrte.wplat[-1]
                    lonf = acrte.wplon[-1]
                    altf = acrte.wpalt[-1]

                    if altf < 0:
                        mask = usepeflightplans.flight_plan_df_back_up['ac'] == acid
                        altf = usepeflightplans.flight_plan_df_back_up[mask].iloc[0]['destination_alt']

                    orig = [lon0, lat0, alt0 ]
                    dest = [lonf, latf, altf ]
                    print( 'changed orig', orig, 'for', acid )
                    print( 'changed dest', dest, 'for', acid )

                    # We check which if the origin is in a no fly zone
                    cond = ( segments_df['lon_min'] <= lon0 ) & ( segments_df['lon_max'] > lon0 ) & \
                        ( segments_df['lat_min'] <= lat0 ) & ( segments_df['lat_max'] > lat0 ) & \
                        ( segments_df['z_min'] <= alt0 ) & ( segments_df['z_max'] > alt0 )

                    if segments_df[cond].empty:
                        segment_name_0 = 'N/A'
                        # origin is not within any segment
                        usepedronecommands.droneLanding( acid )
                        continue
                    else:
                        segment_name_0 = segments_df[cond].index[0]

                    # We check which is the destination is in a no fly zone
                    cond = ( segments_df['lon_min'] <= lonf ) & ( segments_df['lon_max'] > lonf ) & \
                        ( segments_df['lat_min'] <= latf ) & ( segments_df['lat_max'] > latf ) & \
                        ( segments_df['z_min'] <= altf ) & ( segments_df['z_max'] > altf )

                    if segments_df[cond].empty:
                        segment_name_f = 'N/A'
                        # origin is not within any segment
                        usepedronecommands.droneLanding( acid )
                        continue
                    else:
                        segment_name_f = segments_df[cond].index[0]

                    if ( self.segments['speed_max'][segment_name_0] == 0 ) | ( self.segments['speed_max'][segment_name_f] == 0 ):
                        # origin or destination is not allowed, so the drone lands
                        usepedronecommands.droneLanding( acid )
                        continue

                    rerouting = usepestrategic.updateStrategicDeconflictionDrone( acid, orig, dest )

                    if rerouting:
                        scn = usepedronecommands.rerouteDrone( acid )

                        acrte.wpstack[iactwp] = ['DEL {}'.format( acid ), scn]

                # 4th. To update the flight plans in the queue
                usepeflightplans.reprocessFlightPlans()

        self.calcWptDict()
        self.calcRecentPath()

        self.segmentation_service.cells["new"] = False
        self.segmentation_service.cells["updated"] = False

        return

    def reset( self ):  # Not used
        return


class UsepeStrategicDeconfliction( core.Entity ):
    ''' UsepeStrategicDeconfliction new entity for BlueSky
    This class implements the strategic deconfliction service.

     '''

    def __init__( self, initial_time, final_time ):
        """
        Create an initial data structure with the information of how the segments are populated.

        Create counters for delivery and backgroun drones
        """

        super().__init__()
        self.initial_time = initial_time
        self.final_time = final_time
        self.users = initialPopulation( usepesegments.segments, self.initial_time, self.final_time )

        # TODO: to include more drone purposes (e.g. emergency, etc.)
        self.delivery_drones = 0
        self.background_drones = 0
        self.surveillance_drones = 0

    def initialisedUsers( self ):
        """
        When the segments are updated, information of how the segments are populated is initialised
        for t > sim.simt
        """
        time = math.floor( sim.simt )
        print( time )

        new_users = initialPopulation( usepesegments.segments, self.initial_time, self.final_time )

        for key in new_users:
            if key in self.users:
                new_users[key][0:time] = self.users[key][0:time]

        self.users = new_users


    def strategicDeconflictionDrone( self, df_row, new=True ):
        """
        It receives as input a row of the flight plan buffer DataFrame, compute a flight plan
        without conflicts and adds a row to the flight plan processed DataFrame
        """
        row = df_row.iloc[0]
        orig = [row['origin_lon'], row['origin_lat'], row['origin_alt'] ]
        dest = [row['destination_lon'], row['destination_lat'], row['destination_alt'] ]
        departure_time = row['departure_s']

        if not new:
            name = row['ac']
        else:
            # TODO: add more purposes
            if row['purpose'] == 'delivery':
                self.delivery_drones += 1
                name = row['purpose'].upper() + str( self.delivery_drones )
            elif row['purpose'] == 'background':
                self.background_drones += 1
                name = row['purpose'].upper() + str( self.background_drones )
            elif row['purpose'] == 'surveillance':
                self.surveillance_drones += 1
                name = row['purpose'].upper() + str( self.surveillance_drones )

        # TODO: add all the drone types or read the information directly from BlueSky parameters
        if row['drone'] == 'M600':
            v_max = 18
            vs_max = 5
            safety_volume_size = 1
        elif row['drone'] == 'Amzn':
            v_max = 44
            vs_max = 8
            safety_volume_size = 1
        elif row['drone'] == 'W178':
            v_max = 42
            vs_max = 6
            safety_volume_size = 1

        # operation_id used when a premade scenario has been created to be later inserted during the flight
        op_id = None
        if 'operation_id' in row.keys():
            op_id = row['operation_id']

        ac = {'id': name, 'type': row['drone'], 'accel': 3.5, 'v_max': v_max, 'vs_max': vs_max,
              'safety_volume_size': safety_volume_size, 'purpose': row['purpose'], 'op_id': op_id}

        if ac['purpose'] == 'delivery':
            users, route, delayed_time = deconflictedDeliveryPathPlanning( orig, dest, dest, orig,
                                                                           departure_time, usepegraph.graph,
                                                                           self.users, self.initial_time,
                                                                           self.final_time,
                                                                           copy.deepcopy( usepesegments.segments ),
                                                                           usepeconfig, ac, hovering_time=30,
                                                                           only_rerouting=False )
        elif ac['purpose'] == 'surveillance':
            users, route, delayed_time = deconflictedSurveillancePathPlanning( orig, dest, dest, orig,
                                            departure_time, usepegraph.graph, self.users,
                                            self.initial_time, self.final_time,
                                            copy.deepcopy( usepesegments.segments ), usepeconfig,
                                            ac, only_rerouting=False, wait_time=row['operation_duration'] )
        else:
            users, route, delayed_time = deconflictedPathPlanning( orig, dest, departure_time,
                                                                   usepegraph.graph, self.users,
                                                                   self.initial_time, self.final_time,
                                                                   copy.deepcopy( usepesegments.segments ),
                                                                   usepeconfig, ac )

        df_row['delayed_time'] = delayed_time
        usepeflightplans.route_dict[name] = route
        usepeflightplans.ac_dict[name] = ac
        df_row['ac'] = name
        usepeflightplans.flight_plan_df_processed = pd.concat( 
            [usepeflightplans.flight_plan_df_processed, df_row] ).sort_values( by='delayed_time' )

        self.users = users

    def updateStrategicDeconflictionDrone( self, acid, orig, dest ):
        """
        When the segments are updated, a new flight plan is calculated (based on the new
        configuration of the airspace)
        Inputs:
                acid (str): name (callsign) of the drone
                orig (list): [lon, lat, alt] of the origin point. It is the coordinates of the active
                             wpt
                dest (list): [lon, lat, alt] of the destination point.

        Return:
                rerouting (bool): False if the current flight is a predetermined cleared path
        """

        departure_time = sim.simt

        name = acid

        ac = usepeflightplans.ac_dict[name]

        rerouting = True

        if ac['purpose'] == 'delivery':

            mask = usepeflightplans.flight_plan_df_back_up['ac'] == acid

            latf2 = usepeflightplans.flight_plan_df_back_up[mask].iloc[0]['origin_lat']
            lonf2 = usepeflightplans.flight_plan_df_back_up[mask].iloc[0]['origin_lon']
            altf2 = usepeflightplans.flight_plan_df_back_up[mask].iloc[0]['origin_alt']

            dest2 = [lonf2, latf2, altf2 ]

            if ( dest[0] == dest2[0] ) and ( dest[1] == dest2[1] ):  #  if delivery drone is already coming back
                users, route, delayed_time = deconflictedPathPlanning( orig, dest, departure_time,
                                                                       usepegraph.graph, self.users,
                                                                       self.initial_time, self.final_time,
                                                                       copy.deepcopy( usepesegments.segments ), usepeconfig,
                                                                       ac, only_rerouting=True )
            else:
                users, route, delayed_time = deconflictedDeliveryPathPlanning( orig, dest, dest, dest2,
                                                                               departure_time, usepegraph.graph,
                                                                               self.users, self.initial_time,
                                                                               self.final_time,
                                                                               copy.deepcopy( usepesegments.segments ),
                                                                               usepeconfig, ac, hovering_time=30,
                                                                               only_rerouting=True )
        elif ac['purpose'] == 'surveillance':
            mask = usepeflightplans.flight_plan_df_back_up['ac'] == acid
            fp_row = usepeflightplans.flight_plan_df_back_up[mask].iloc[0]

            latf_leave = fp_row['destination_lat']
            lonf_leave = fp_row['destination_lon']
            altf_leave = fp_row['destination_alt']

            dest_leave = [lonf_leave, latf_leave, altf_leave]

            latf_return = fp_row['origin_lat']
            lonf_return = fp_row['origin_lon']
            altf_return = fp_row['origin_alt']

            dest_return = [lonf_return, latf_return, altf_return]

            if ( dest[0] == dest_return[0] ) and ( dest[1] == dest_return[1] ):
                users, route, _ = deconflictedPathPlanning( orig, dest, departure_time,
                                                usepegraph.graph, self.users, self.initial_time,
                                                self.final_time, copy.deepcopy( usepesegments.segments ),
                                                usepeconfig, ac, only_rerouting=True )

            elif ( dest[0] == dest_leave[0] ) and ( dest[1] == dest_leave[1] ):
                users, route, _ = deconflictedSurveillancePathPlanning( orig, dest, dest,
                                                dest_return, departure_time, usepegraph.graph,
                                                self.users, self.initial_time, self.final_time,
                                                copy.deepcopy( usepesegments.segments ), usepeconfig, ac,
                                                only_rerouting=True, wait_time=fp_row['operation_duration'] )
            else:
                users = self.users
                route = usepeflightplans.route_dict[name]
                rerouting = False
        else:
            users, route, delayed_time = deconflictedPathPlanning( orig, dest, departure_time,
                                                                   usepegraph.graph, self.users,
                                                                   self.initial_time, self.final_time,
                                                                   copy.deepcopy( usepesegments.segments ), usepeconfig,
                                                                   ac, only_rerouting=True )

        usepeflightplans.route_dict[name] = route

        self.users = users

        return rerouting

    def update( self ):  # Not used
        return

    def preupdate( self ):  # Not used
        return

    def reset( self ):  # Not used
        return


class UsepePathPlanning( core.Entity ):  # Not used
    ''' UsepePathPlanning new entity for BlueSky '''

    def __init__( self ):
        super().__init__()

    def update( self ):  # Not used
        return

    def preupdate( self ):  # Not used
        return

    def reset( self ):  # Not used
        return


class UsepeDroneCommands( core.Entity ):
    ''' UsepeDroneCommands new entity for BlueSky
    This class is used to transform the route into BlueSky commands
    '''

    def __init__( self ):
        super().__init__()

    def createDroneCommands( self, row ):
        """
        Create the commands for a processed flight plan
        Inputs:
                row: row of the flight plan processed dataframe
        """
        route = usepeflightplans.route_dict[row['ac']]
        ac = usepeflightplans.ac_dict[row['ac']]
        # departure_time = str( datetime.timedelta( seconds=row['delayed_time'] ) )

        departure_time = str( datetime.timedelta( seconds=0 ) )  # Relative time is considered
        G = usepegraph.graph
        layers_dict = usepegraph.layers_dict

        scenario_name = f'scenario_traffic_drone_{ac["id"]}.scn'
        scenario_path = Path( 'USEPE/temp', scenario_name )
        scenario_file = open( Path( 'scenario', scenario_path ), 'w' )

        if ac['purpose'] == 'delivery':
            createDeliveryFlightPlan( route[0], route[1], ac, departure_time, G, layers_dict,
                                      scenario_file, scenario_path, hovering_time=30 )
        elif ac['purpose'] == 'surveillance':
            premade_scenario_path = Path( 'USEPE', 'exercise_3', 'surveillance_' + ac['op_id'] + '.scn' )
            createSurveillanceFlightPlan( route, ac, departure_time, G, layers_dict,
                scenario_file, scenario_path, premade_scenario_path )
        else:
            createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )

        scenario_file.close()

        stack.stack( f'PCALL {scenario_path} REL' )

    def rerouteDrone( self, acid ):
        """
        When the segments are updated, it is used to reroute the flights that have already departed
        """
        route = usepeflightplans.route_dict[acid]
        ac = usepeflightplans.ac_dict[acid]

        departure_time = str( datetime.timedelta( seconds=0 ) )  # Relative time is considered
        G = usepegraph.graph
        layers_dict = usepegraph.layers_dict

        scenario_name = f'scenario_traffic_drone_{ac["id"]}.scn'
        scenario_path = Path( 'usepe/temp', scenario_name )
        scenario_file = open( Path( 'scenario', scenario_path ), 'w' )

        if ac['purpose'] == 'delivery':
            if len( route ) == 2:
                createDeliveryFlightPlan( route[0], route[1], ac, departure_time, G, layers_dict,
                                          scenario_file, scenario_path, hovering_time=30 )
            else:
                createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )
        elif ac['purpose'] == 'surveillance':
            if len( route ) == 2:
                premade_scenario_path = Path( 'USEPE', 'exercise_3', 'surveillance_' + ac['op_id'] + '.scn' )
                createSurveillanceFlightPlan( route[0], route[1], ac, departure_time, G, layers_dict,
                    scenario_file, scenario_path, premade_scenario_path )
            else:
                createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )
        else:
            createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )

        scenario_file.close()

        text = f'PCALL {scenario_path} REL'

        return text

    def droneTakeOff( self ):
        """
        It goes over all the processed flight plan that departs in this time step
        """
        while not usepeflightplans.flight_plan_df_processed[usepeflightplans.flight_plan_df_processed['delayed_time'] <= sim.simt].empty:
            df_row = usepeflightplans.flight_plan_df_processed.iloc[[0]]
            row = df_row.iloc[0]
            self.createDroneCommands( row )
            usepeflightplans.flight_plan_df_processed = usepeflightplans.flight_plan_df_processed.drop( usepeflightplans.flight_plan_df_processed.index[0] )
            usepeflightplans.flight_plan_df_back_up = pd.concat( [usepeflightplans.flight_plan_df_back_up, df_row] )

    def droneLanding( self, acid ):
        print( ' Drone: {} is landing'.format( acid ) )
        stack.stack( 'SPD {} 0'.format( acid ) )
        stack.stack( '{} ATSPD 0 {} ALT 0'.format( acid, acid ) )
        stack.stack( '{} ATALT 0 DEL {}'.format( acid, acid ) )

    def update( self ):  # Not used
        return

    def preupdate( self ):
        self.droneTakeOff()
        return

    def reset( self ):  # Not used
        return


class UsepeFlightPlan( core.Entity ):
    ''' UsepeFlightPlan new entity for BlueSky
    This class contains all the information of the planned flights
    flight_plan_df: DataFrame with all the information passed to BlueSky
    flight_plan_df_buffer: DataFrame with all the flights that have not been planned yet (simt < planned_time_s)
    flight_plan_df_processed: DataFrame with all the flights that have been processed (simt > planned_time_s)
    '''

    def __init__( self, fligh_plan_csv_path ):
        super().__init__()

        self.flight_plan_df = pd.read_csv( fligh_plan_csv_path, index_col=0 ).sort_values( by=['planned_time_s'] )
        self.flight_plan_df_buffer = self.flight_plan_df[self.flight_plan_df['planned_time_s'] >= sim.simt]
        self.flight_plan_df_processed = pd.DataFrame( columns=list( self.flight_plan_df.columns ) +
                                                      ['delayed_time'] + ['ac'] )

        self.flight_plan_df_back_up = self.flight_plan_df_processed.copy()


        self.route_dict = {}
        self.ac_dict = {}
        # self.processFlightPlans()

    def processFlightPlans( self ):
        """
        To process the planned flight plans
        """
        # segments_df = pd.DataFrame.from_dict( usepesegments.segments, orient='index' )
        segments_df = usepesegments.segments
        while not self.flight_plan_df_buffer[self.flight_plan_df_buffer['planned_time_s'] <= sim.simt].empty:
            df_row = self.flight_plan_df_buffer.iloc[[0]]
            # print( df_row )

            row = df_row.iloc[0]
            orig = [row['origin_lon'], row['origin_lat'], row['origin_alt'] ]
            dest = [row['destination_lon'], row['destination_lat'], row['destination_alt'] ]

            # We check if the origin/destination is in a no fly zone
            # cond = ( segments_df['lon_min'] <= orig[0] ) & ( segments_df['lon_max'] > orig[0] ) & \
            #     ( segments_df['lat_min'] <= orig[1] ) & ( segments_df['lat_max'] > orig[1] ) & \
            #     ( segments_df['z_min'] <= orig[2] ) & ( segments_df['z_max'] > orig[2] )
            #
            # if segments_df[cond].empty:
            #     segment_name_0 = 'N/A'
            # else:
            #     segment_name_0 = segments_df[cond].index[0]

            orig_node = nearestNode3d( usepegraph.graph, lon=orig[0], lat=orig[1], altitude=orig[2] )
            segment_name_0 = usepegraph.graph.nodes[orig_node]['segment']

            # We check which is the destination is in a no fly zone
            # cond = ( segments_df['lon_min'] <= dest[0] ) & ( segments_df['lon_max'] > dest[0] ) & \
            #     ( segments_df['lat_min'] <= dest[1] ) & ( segments_df['lat_max'] > dest[1] ) & \
            #     ( segments_df['z_min'] <= dest[2] ) & ( segments_df['z_max'] > dest[2] )
            #
            # if segments_df[cond].empty:
            #     segment_name_f = 'N/A'
            # else:
            #     segment_name_f = segments_df[cond].index[0]

            dest_node = nearestNode3d( usepegraph.graph, lon=dest[0], lat=dest[1], altitude=dest[2] )
            segment_name_f = usepegraph.graph.nodes[dest_node]['segment']

            # print( usepesegments.segments['class'][segment_name_0] )
            # print( usepesegments.segments['class'][segment_name_f] )

            if ( segment_name_0 == 'N/A' ) | ( segment_name_f == 'N/A' ):
                # origin or destination is not within any segment
                self.flight_plan_df_buffer = self.flight_plan_df_buffer.drop( self.flight_plan_df_buffer.index[0] )
                print( 'Origin or destination is not within any segment' )
            else:
                if ( usepesegments.segments['speed_max'][segment_name_0] == 0 ) | ( usepesegments.segments['speed_max'][segment_name_f] == 0 ):
                    # origin or destination is not allowed, so the flight plan is rejected
                    self.flight_plan_df_buffer = self.flight_plan_df_buffer.drop( self.flight_plan_df_buffer.index[0] )
                    print( 'Origin or destination is not allowed, so the flight plan is rejected' )
                else:
                    usepestrategic.strategicDeconflictionDrone( df_row )
                    self.flight_plan_df_buffer = self.flight_plan_df_buffer.drop( self.flight_plan_df_buffer.index[0] )

    def reprocessFlightPlans( self ):
        """
        When the segments are updated, the processed flight plans are reprocessed according to the
        new airspace configuration
        """
        previous_df = self.flight_plan_df_processed.copy().sort_values( by=['planned_time_s'] )

        self.flight_plan_df_processed = pd.DataFrame( columns=list( self.flight_plan_df.columns ) +
                                                      ['delayed_time'] + ['ac'] )

        # segments_df = pd.DataFrame.from_dict( usepesegments.segments, orient='index' )
        segments_df = usepesegments.segments
        while not previous_df.empty:
            df_row = previous_df.iloc[[0]]
            print( df_row )

            row = df_row.iloc[0]
            orig = [row['origin_lon'], row['origin_lat'], row['origin_alt'] ]
            dest = [row['destination_lon'], row['destination_lat'], row['destination_alt'] ]

            # We check if the origin/destination is in a no fly zone
            # cond = ( segments_df['lon_min'] <= orig[0] ) & ( segments_df['lon_max'] > orig[0] ) & \
            #     ( segments_df['lat_min'] <= orig[1] ) & ( segments_df['lat_max'] > orig[1] ) & \
            #     ( segments_df['z_min'] <= orig[2] ) & ( segments_df['z_max'] > orig[2] )
            #
            # if segments_df[cond].empty:
            #     segment_name_0 = 'N/A'
            # else:
            #     segment_name_0 = segments_df[cond].index[0]

            orig_node = nearestNode3d( usepegraph.graph, lon=orig[0], lat=orig[1], altitude=orig[2] )
            segment_name_0 = usepegraph.graph.nodes[orig_node]['segment']

            # We check which is the destination is in a no fly zone
            # cond = ( segments_df['lon_min'] <= dest[0] ) & ( segments_df['lon_max'] > dest[0] ) & \
            #     ( segments_df['lat_min'] <= dest[1] ) & ( segments_df['lat_max'] > dest[1] ) & \
            #     ( segments_df['z_min'] <= dest[2] ) & ( segments_df['z_max'] > dest[2] )
            #
            # if segments_df[cond].empty:
            #     segment_name_f = 'N/A'
            # else:
            #     segment_name_f = segments_df[cond].index[0]

            dest_node = nearestNode3d( usepegraph.graph, lon=dest[0], lat=dest[1], altitude=dest[2] )
            segment_name_f = usepegraph.graph.nodes[dest_node]['segment']

            if ( segment_name_0 == 'N/A' ) | ( segment_name_f == 'N/A' ):
                # origin or destination is not within any segment
                previous_df = previous_df.drop( previous_df.index[0] )
            else:
                if ( usepesegments.segments['speed_max'][segment_name_0] == 0 ) | ( usepesegments.segments['speed_max'][segment_name_f] == 0 ):
                    # origin or destination is not allowed, so the flight plan is rejected
                    previous_df = previous_df.drop( previous_df.index[0] )
                else:
                    usepestrategic.strategicDeconflictionDrone( df_row, new=False )
                    previous_df = previous_df.drop( previous_df.index[0] )


    def update( self ):  # Not used
        return

    def preupdate( self ):
        self.processFlightPlans()
        return

    def reset( self ):  # Not used
        return


class UsepeWind( core.Entity ):
    ''' UsepeWind new entity for BlueSky
    This class import the wind data to BlueSky simulation
    '''

    def __init__( self ):
        super().__init__()

        # We call a pre-computed scenario with the wind information. We only consider one wind
        # snapshot for all the simulation, so we do this once

        wind_path = r"{}".format( usepeconfig['BlueSky']['wind_scenario_path'] )
        if wind_path:
            print( "Loading wind in the simulation. File {}".format( wind_path ) )
            stack.stack( 'PCALL {} REL'.format( wind_path ) )
            print( "Completed" )
        else:
            print( "No path to wind files specified, skipping all wind simulation!" )

    def update( self ):  # Not used
        return

    def preupdate( self ):  # Not used
        return

    def reset( self ):  # Not used
        return


class StateBasedUsepe( ConflictDetection ):
    def __init__( self ):
        super().__init__()

        self.confpairs_default = list()

        # read look-up tables
        lookup_tables_dir = usepeconfig['BlueSky']['lookup_tables_dir']
        self.tables = self.readAllLookUpTables( lookup_tables_dir )

        self.table_grid = 10
        self.time_to_react_min = 5
        self.table_gs_list = [6, 12, 18, 24, 44]

    def readLookUpTable( self, path ):
        columns = ['x', 'y', 'WCV', 'course', 'min_dist', 'v_ow', 'min_dist_v', 'min_dist_h',
                   'alt', 'v_in', 'time', 'prob', 'pitch']
        df = pd.read_csv( path, names=columns, sep=' ' )
        return df

    def readAllLookUpTables( self, lookup_tables_dir ):
        tables = {}
        for file in listdir( lookup_tables_dir ):
            path = join( lookup_tables_dir, file )
            df = self.readLookUpTable( path )
            tables[file[6:-6]] = df
        return tables

    def checkTable( self, table, x, y, course, gs_ow, gs_in ):
        if table.empty:
            return [0]

        df = table[( table['x'] == x ) &
                   ( table['y'] == y ) &
                   ( table['v_ow'] == gs_ow ) &
                   ( table['v_in'] == gs_in ) &
                   ( table['course'] == course ) ].sort_values( by=['time'] )
        time_to_react = list( df['time'] )
        return time_to_react

    def selectTable( self, drone, manoeuvre, gs ):
        if drone == 'AMZN':
            drone_code = 'A'
        elif drone == 'M600':
            drone_code = 'D'
        elif drone == 'W178':
            if gs > 12:
                drone_code = 'F'
            else:
                drone_code = 'M'

        key = '-'.join( [drone_code, manoeuvre] )
        if key in self.tables:
            return self.tables[key]
        else:
            print( 'WARNING: The table {} does not exist'.format( key ) )
            columns = ['x', 'y', 'WCV', 'course', 'min_dist', 'v_ow', 'min_dist_v', 'min_dist_h',
                       'alt', 'v_in', 'time', 'prob', 'pitch']
            return pd.DataFrame( columns=columns )


    def detect( self, ownship, intruder, rpz, hpz, dtlookahead ):
        ''' Conflict detection between ownship (traf) and intruder (traf/adsb).'''
        # Identity matrix of order ntraf: avoid ownship-ownship detected conflicts
        I = np.eye( ownship.ntraf )

        # Horizontal conflict ------------------------------------------------------

        # qdrlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
        qdr, dist = geo.kwikqdrdist_matrix( ownship.lat.view( np.matrix ), ownship.lon.view( np.matrix ),
                                    intruder.lat.view( np.matrix ), intruder.lon.view( np.matrix ) )

        # Convert back to array to allow element-wise array multiplications later on
        # Convert to meters and add large value to own/own pairs
        qdr = np.asarray( qdr )
        dist = np.asarray( dist ) * nm + 1e9 * I

        # Calculate horizontal closest point of approach (CPA)
        qdrrad = np.radians( qdr )
        dx = dist * np.sin( qdrrad )  # is pos j rel to i
        dy = dist * np.cos( qdrrad )  # is pos j rel to i

        # Ownship track angle and speed
        owntrkrad = np.radians( ownship.trk )
        ownu = ownship.gs * np.sin( owntrkrad ).reshape( ( 1, ownship.ntraf ) )  # m/s
        ownv = ownship.gs * np.cos( owntrkrad ).reshape( ( 1, ownship.ntraf ) )  # m/s

        # Intruder track angle and speed
        inttrkrad = np.radians( intruder.trk )
        intu = intruder.gs * np.sin( inttrkrad ).reshape( ( 1, ownship.ntraf ) )  # m/s
        intv = intruder.gs * np.cos( inttrkrad ).reshape( ( 1, ownship.ntraf ) )  # m/s

        du = ownu - intu.T  # Speed du[i,j] is perceived eastern speed of i to j
        dv = ownv - intv.T  # Speed dv[i,j] is perceived northern speed of i to j

        dv2 = du * du + dv * dv
        dv2 = np.where( np.abs( dv2 ) < 1e-6, 1e-6, dv2 )  # limit lower absolute value
        vrel = np.sqrt( dv2 )

        tcpa = -( du * dx + dv * dy ) / dv2 + 1e9 * I

        # Calculate distance^2 at CPA (minimum distance^2)
        dcpa2 = np.abs( dist * dist - tcpa * tcpa * dv2 )

        # Check for horizontal conflict
        # RPZ can differ per aircraft, get the largest value per aircraft pair
        rpz = np.asarray( np.maximum( rpz.view( np.matrix ), rpz.view( np.matrix ).transpose() ) )
        R2 = rpz * rpz
        swhorconf = dcpa2 < R2  # conflict or not

        # Calculate times of entering and leaving horizontal conflict
        dxinhor = np.sqrt( np.maximum( 0., R2 - dcpa2 ) )  # half the distance travelled inzide zone
        dtinhor = dxinhor / vrel

        tinhor = np.where( swhorconf, tcpa - dtinhor, 1e8 )  # Set very large if no conf
        touthor = np.where( swhorconf, tcpa + dtinhor, -1e8 )  # set very large if no conf

        # Vertical conflict --------------------------------------------------------

        # Vertical crossing of disk (-dh,+dh)
        dalt = ownship.alt.reshape( ( 1, ownship.ntraf ) ) - \
            intruder.alt.reshape( ( 1, ownship.ntraf ) ).T + 1e9 * I

        dvs = ownship.vs.reshape( 1, ownship.ntraf ) - \
            intruder.vs.reshape( 1, ownship.ntraf ).T
        dvs = np.where( np.abs( dvs ) < 1e-6, 1e-6, dvs )  # prevent division by zero

        # Check for passing through each others zone
        # hPZ can differ per aircraft, get the largest value per aircraft pair
        hpz = np.asarray( np.maximum( hpz.view( np.matrix ), hpz.view( np.matrix ).transpose() ) )
        tcrosshi = ( dalt + hpz ) / -dvs
        tcrosslo = ( dalt - hpz ) / -dvs
        tinver = np.minimum( tcrosshi, tcrosslo )
        toutver = np.maximum( tcrosshi, tcrosslo )

        # Combine vertical and horizontal conflict----------------------------------
        tinconf = np.maximum( tinver, tinhor )
        toutconf = np.minimum( toutver, touthor )

        swconfl = np.array( swhorconf * ( tinconf <= toutconf ) * ( toutconf > 0.0 ) *
                           np.asarray( tinconf < dtlookahead.view( np.matrix ).T ) * ( 1.0 - I ), dtype=bool )

        # --------------------------------------------------------------------------
        # Update conflict lists
        # --------------------------------------------------------------------------
        # Ownship conflict flag and max tCPA
        inconf = np.any( swconfl, 1 )
        tcpamax = np.max( tcpa * swconfl, 1 )

        # Select conflicting pairs: each a/c gets their own record
        confpairs = [( ownship.id[i], ownship.id[j] ) for i, j in zip( *np.where( swconfl ) )]
        swlos = ( dist < rpz ) * ( np.abs( dalt ) < hpz )
        lospairs = [( ownship.id[i], ownship.id[j] ) for i, j in zip( *np.where( swlos ) )]

        # ------------------------------------------------------------------------------
        # Look-up tables
        # ------------------------------------------------------------------------------
        self.confpairs_default = confpairs

        # print( 'confpairs_default' )
        # print( self.confpairs_default )
        # print( 'default' )
        # print( self.confpairs_default )

        if usepeconfig.getboolean( 'BlueSky', 'D2C2' ):

            for i in range( len( swconfl ) ):
                for j in range( len( swconfl[i] ) ):
                    if swconfl[i][j]:
                        ow_ = ownship.id[i]
                        in_ = intruder.id[j]

                        # Calculate relative position
                        qdr_iter = qdr[i][j] - ownship.trk[i]
                        qdrrad_iter = np.radians( qdr_iter )
                        dx_iter = dist[i][j] * np.sin( qdrrad_iter )  # is pos j rel to i
                        dy_iter = dist[i][j] * np.cos( qdrrad_iter )  # is pos j rel to i

                        # Ownship speed
                        own_gs_iter = ownship.gs[i]  # m/s

                        own_gs_iter_table = min( self.table_gs_list, key=lambda x:abs( x - own_gs_iter ) )

                        # Intruder track angle and speed
                        int_gs_iter = intruder.gs[j]  # m/s

                        int_gs_iter_table = min( self.table_gs_list, key=lambda x:abs( x - int_gs_iter ) )

                        dx_grid = [math.floor( dx_iter / self.table_grid ) * self.table_grid, math.floor( dx_iter / self.table_grid ) * self.table_grid,
                                   math.ceil( dx_iter / self.table_grid ) * self.table_grid, math.ceil( dx_iter / self.table_grid ) * self.table_grid]

                        dy_grid = [math.floor( dy_iter / self.table_grid ) * self.table_grid, math.ceil( dy_iter / self.table_grid ) * self.table_grid,
                                   math.floor( dy_iter / self.table_grid ) * self.table_grid, math.ceil( dy_iter / self.table_grid ) * self.table_grid]

                        time_to_react_all_list = []

                        # intruder relative course
                        ow_course = ownship.aporasas.trk[i]
                        in_course = intruder.aporasas.trk[j]

                        relative_course = round( ( in_course - 180 ) - ow_course )
                        if relative_course >= 360:
                            relative_course -= 360
                        elif relative_course < 0:
                            relative_course += 360

                        for manoeuvre in ['H', 'S', 'V']:
                            # select table
                            table = self.selectTable( ownship.type[i], manoeuvre, own_gs_iter_table )
                            time_to_react_list = []

                            for x, y in zip( dx_grid, dy_grid ):
                                if manoeuvre == 'S' and x == 0 and relative_course == 0:  # check this condition
                                    time_to_react_list.extend( [0] )
                                else:
                                    time_to_react_list.extend( self.checkTable( table, x, y, relative_course, own_gs_iter_table, int_gs_iter_table ) )

                            if time_to_react_list:
                                time_to_react_all_list.extend( [min( time_to_react_list )] )
                        if time_to_react_all_list:
                            if max( time_to_react_all_list ) <= self.time_to_react_min:
                                pass
                            else:
                                swconfl[i][j] = False
                        else:
                            swconfl[i][j] = False

                    else:
                        pass

            # Ownship conflict flag and max tCPA
            inconf = np.any( swconfl, 1 )
            tcpamax = np.max( tcpa * swconfl, 1 )

            # Select conflicting pairs: each a/c gets their own record
            confpairs = [( ownship.id[i], ownship.id[j] ) for i, j in zip( *np.where( swconfl ) )]

            # print( 'confpairs' )
            # print( confpairs )

        # print( 'confpairs' )
        # print( confpairs )

        return confpairs, lospairs, inconf, tcpamax, \
            qdr[swconfl], dist[swconfl], np.sqrt( dcpa2[swconfl] ), \
                tcpa[swconfl], tinconf[swconfl]

