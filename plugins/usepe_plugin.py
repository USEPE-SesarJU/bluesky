""" This plugin load the graph for the USEPE project. """
# Import the global bluesky objects. Uncomment the ones you need
from os import listdir
from os.path import join
import configparser
import copy
import datetime
import math
import os
import pickle
from pathlib import Path

from bluesky import core, traf, stack, sim  # , settings, navdb,  scr, tools
from bluesky.tools import geo
from bluesky.tools.aero import nm
from bluesky.traffic.asas.detection import ConflictDetection
from bluesky.traffic.asas.statebased import StateBased
from usepe.city_model.dynamic_segments import dynamicSegments
from usepe.city_model.scenario_definition import createFlightPlan, createDeliveryFlightPlan, createSurveillanceFlightPlan
from usepe.city_model.strategic_deconfliction import initialPopulation, deconflictedPathPlanning, deconflictedDeliveryPathPlanning, deconflictedSurveillancePathPlanning
from usepe.city_model.utils import read_my_graphml, layersDict
import numpy as np
import pandas as pd


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2022'


usepeconfig = None
usepegraph = None
usepesegments = None
usepestrategic = None
usepeflightplans = None
usepedronecommands = None
updateInterval = 1.0


# ## Initialisation function of your plugin. Do not change the name of this
# ## function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate the UsepeLogger entity
    global usepeconfig

    global usepegraph
    global usepesegments
    global usepeflightplans
    global usepestrategic
    global usepedronecommands

    # ---------------------------------- DEFINED BY USER ------------------------------------
    # config_path = r"C:\workspace3\scenarios-USEPE\scenario\USEPE\exercise_1\settings_exercise_1_reference.cfg"
    config_path = r"/home/ror/ws/scenarios/scenario/USEPE/exercise_3/settings_exe_3_ref.cfg"
    # ------------------------------------------------------------------------------------------

    # graph_path = r"C:\workspace3\scenarios-USEPE\scenario\USEPE\exercise_1\data\testing_graph.graphml"
    # segment_path = r"C:\workspace3\scenarios-USEPE\scenario\USEPE\exercise_1\data\offline_segments.pkl"
    # flight_plan_csv_path = r"C:\workspace3\scenarios-USEPE\scenario\USEPE\exercise_1\data\delivery_testing.csv"
    #
    # initial_time = 0  # seconds
    # final_time = 7200  # seconds

    usepeconfig = configparser.ConfigParser()
    usepeconfig.read( config_path )

    graph_path = usepeconfig['BlueSky']['graph_path']
    segment_path = usepeconfig['BlueSky']['segment_path']
    flight_plan_csv_path = usepeconfig['BlueSky']['flight_plan_csv_path']

    initial_time = int( usepeconfig['BlueSky']['initial_time'] )
    final_time = int( usepeconfig['BlueSky']['final_time'] )

    usepegraph = UsepeGraph( graph_path )
    usepesegments = UsepeSegments( segment_path )
    usepestrategic = UsepeStrategicDeconfliction( initial_time, final_time )
    usepeflightplans = UsepeFlightPlan( flight_plan_csv_path )
    usepedronecommands = UsepeDroneCommands()

    # Activate the detection and resolution method, and logger
    configuration_path = r"{}".format( usepeconfig['BlueSky']['configuration_path'] )
    stack.stack( 'PCALL {} REL'.format( configuration_path ) )
    stack.stack( 'OP' )

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

    # init_plugin() should always return a configuration dict.
    return config


def update():
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
    usepegraph.preupdate()
    usepesegments.preupdate()
    usepestrategic.preupdate()
    usepeflightplans.preupdate()
    usepedronecommands.preupdate()
    return


def reset():
    usepegraph.reset()
    usepesegments.reset()
    usepestrategic.reset()
    usepeflightplans.reset()
    usepedronecommands.reset()
    return


class UsepeGraph( core.Entity ):
    ''' UsepeGraph new entity for BlueSky
    This class reads the graph that represents the city.
    '''

    def __init__( self, graph_path ):
        super().__init__()

        self.graph = graph_path
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

    def __init__( self, segment_path ):
        super().__init__()

        with open( segment_path, 'rb' ) as f:
            self.segments = pickle.load( f )

        #### Remove: This is included for testing. We want to avoid no-fly zones
        # for key in self.segments:
        #     self.segments[key]['speed'] = 20
        #     self.segments[key]['capacity'] = 5
        #     self.segments[key]['updated'] = True
        #     self.segments[key]['new'] = True
        #     if self.segments[key]['speed'] == 0:
        #         self.segments[key]['speed'] = 5
        #         self.segments[key]['capacity'] = 1
        #         self.segments[key]['updated'] = True
        usepegraph.graph, self.segments = dynamicSegments( usepegraph.graph, usepeconfig, self.segments, deleted_segments=None )
        # Initialise class of dynamic segmentation - it provides the initial segments
        # Include the region for input
        #####

        with self.settrafarrays():
            self.recentpath = np.array( [], dtype=np.ndarray )

    def create( self, n=1 ):
        super().create( n )

        positions = math.ceil( 300 / updateInterval )
        self.recentpath[-n:] = [np.empty( positions, dtype=tuple ) for _ in range( n )]

    def dynamicSegments( self ):
        """
        TODO. Here we have to include the function which updates the segments
        """
        updated = False
        if not usepeconfig.getboolean( 'BlueSky', 'D2C2' ):
            return updated, self.segments

        '''if ( sim.simt > 1800 ) & ( sim.simt < 1802 ):
            self.segments['573']['speed'] = 0
            self.segments['573']['capacity'] = 0
            self.segments['573']['updated'] = True
            updated = True
            stack.stack( 'POLY segment573 52.375,9.72,52.375,9.73,52.3875,9.73,52.3875,9.72' )

        if ( sim.simt > 3600 ) & ( sim.simt < 3602 ):
            self.segments['573']['speed'] = 20
            self.segments['573']['capacity'] = 5
            self.segments['573']['updated'] = True
            updated = True
        '''
        segments = self.segments


        '''Inputs provided for the update rules'''
        # waypoints for each drone + graph
        usepeflightplans.route_dict  # dictionary containing the waypoints of each drone flying. Key - drone id, values - list of waypoints id
        # usepegraph.node  # dict containng the features of each waypoint key - waypoint id, value dict with features and its values

        # drone ids
        # traf.id

        # drone path
        for i in range( self.recentpath.size ):
            temparr = np.empty_like( self.recentpath[i] )
            currentpos = ( sim.simt, traf.lat[i], traf.lon[i], traf.alt[i] )
            temparr[-1] = currentpos
            temparr[:-1] = self.recentpath[i][1:]
            self.recentpath[i][:] = temparr

        # active waypoint
        actwp = []
        for acid in traf.id:
            idx = traf.id2idx( acid )
            acrte = traf.ap.route[idx]
            iactwp = acrte.iactwp
            actwp.append( acrte.wpname[iactwp] )

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


        # Save variables
        if ( ( sim.simt > 27 ) & ( sim.simt < 29 ) ) or\
            ( ( sim.simt > 627 ) & ( sim.simt < 629 ) ) or\
            ( ( sim.simt > 727 ) & ( sim.simt < 729 ) ) or\
            ( ( sim.simt > 1727 ) & ( sim.simt < 1729 ) ):
            #
            with open( 'drone_id.list', 'wb' ) as file:
                pickle.dump( traf.id, file )
            #
            with open( 'drone_active_waypoint.list', 'wb' ) as file:
                pickle.dump( actwp, file )
            #
            with open( 'drones_routes.dict', 'wb' ) as file:
                pickle.dump( usepeflightplans.route_dict, file )
            #
            # with open( 'graph_test.graph', 'wb' ) as file:
            #    pickle.dump( usepegraph, filehandler )
            #
            with open( 'drones_actual_paths.list', 'wb' ) as file:
                pickle.dump( self.recentpath, file )
            #
            with open( 'conflict_pairs.list', 'wb' ) as file:
                pickle.dump( currentconf, file )
            #
            with open( 'conflict_pairs_cpa_location.list', 'wb' ) as file:
                pickle.dump( currentconf_loc, file )
            #
            with open( 'conflict_pairs_headings.list', 'wb' ) as file:
                pickle.dump( currentconf_hdg, file )
            #
            # print( 'Saved.' )


        '''Update rules'''
        # Call update method from dynamic segments class - includes all the rules

        return updated, segments

    def update( self ):  # Not used
        # stack.stack( 'ECHO Example update: import segments' )
        return

    def preupdate( self ):
        updated, self.segments = self.dynamicSegments()

        if updated:
            # TODO: Perform all the activities associated to the segmetns update

            # 1st:  to update the graph
            usepegraph.graph, self.segments = dynamicSegments( usepegraph.graph, usepeconfig, self.segments, deleted_segments=None )

            # 2nd: to initialised the population of segments
            usepestrategic.initialisedUsers()

            segments_df = pd.DataFrame.from_dict( self.segments, orient='index' )
            # 3rd. To update the drones that are already flying
            for acid in traf.id:
                print( acid )
                idx = traf.id2idx( acid )

                acrte = traf.ap.route[idx]
                iactwp = acrte.iactwp
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
                    altf = traf.alt[idx]

                orig = [lon0, lat0, alt0 ]
                dest = [lonf, latf, altf ]

                # We check which if the origin is in a no fly zone
                cond = ( segments_df['lon_min'] <= lon0 ) & ( segments_df['lon_max'] > lon0 ) & \
                    ( segments_df['lat_min'] <= lat0 ) & ( segments_df['lat_max'] > lat0 ) & \
                    ( segments_df['z_min'] <= alt0 ) & ( segments_df['z_max'] > alt0 )

                if segments_df[cond].empty:
                    segment_name_0 = 'N/A'
                else:
                    segment_name_0 = segments_df[cond].index[0]

                # We check which is the destination is in a no fly zone
                cond = ( segments_df['lon_min'] <= lonf ) & ( segments_df['lon_max'] > lonf ) & \
                    ( segments_df['lat_min'] <= latf ) & ( segments_df['lat_max'] > latf ) & \
                    ( segments_df['z_min'] <= altf ) & ( segments_df['z_max'] > altf )

                if segments_df[cond].empty:
                    segment_name_f = 'N/A'
                else:
                    segment_name_f = segments_df[cond].index[0]

                if ( self.segments[segment_name_0]['speed'] == 0 ) | ( self.segments[segment_name_f]['speed'] == 0 ):
                    # origin or destination is not allowed, so the drone lands
                    usepedronecommands.droneLanding( acid )
                    continue

                usepestrategic.updateStrategicDeconflictionDrone( acid, orig, dest )

                scn = usepedronecommands.rerouteDrone( acid )

                acrte.wpstack[iactwp] = ['DEL {}'.format( acid ), scn]

            # 4th. To update the flight plans in the queue
            usepeflightplans.reprocessFlightPlans()

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
                name = row['purpose'].upper() + str(self.surveillance_drones)

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
            users, route, delayed_time = deconflictedSurveillancePathPlanning(orig, dest, dest, orig,
                                            departure_time, usepegraph.graph, self.users,
                                            self.initial_time, self.final_time,
                                            copy.deepcopy(usepesegments.segments), usepeconfig,
                                            ac, only_rerouting=False, wait_time=row['operation_duration'])
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
        """

        departure_time = sim.simt

        name = acid

        ac = usepeflightplans.ac_dict[name]

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
        else:
            users, route, delayed_time = deconflictedPathPlanning( orig, dest, departure_time,
                                                                   usepegraph.graph, self.users,
                                                                   self.initial_time, self.final_time,
                                                                   copy.deepcopy( usepesegments.segments ), usepeconfig,
                                                                   ac, only_rerouting=True )

        usepeflightplans.route_dict[name] = route

        self.users = users

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
        scenario_path = Path('USEPE/temp', scenario_name)
        scenario_file = open(Path('scenario', scenario_path), 'w')

        if ac['purpose'] == 'delivery': #TODO Conform to new scenario path
            createDeliveryFlightPlan( route[0], route[1], ac, departure_time, G, layers_dict,
                                      scenario_file, scenario_path, hovering_time=30 )
        elif ac['purpose'] == 'surveillance':
            premade_scenario_path = Path('USEPE', 'exercise_3', 'surveillance_' + ac['op_id'] + '.scn')
            createSurveillanceFlightPlan(route[0], route[1], ac, departure_time, G, layers_dict, scenario_file, scenario_path, premade_scenario_path)
        else:
            createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )

        scenario_file.close()

        stack.stack(f'PCALL {scenario_path} REL')

    def rerouteDrone( self, acid ):
        """
        When the segments are updated, it is used to reroute the flights that have already departed
        """
        route = usepeflightplans.route_dict[acid]
        ac = usepeflightplans.ac_dict[acid]

        departure_time = str( datetime.timedelta( seconds=0 ) )  # Relative time is considered
        G = usepegraph.graph
        layers_dict = usepegraph.layers_dict

        scenario_path = r'.\scenario\usepe\temp\scenario_traffic_drone_{}.scn'.format( ac['id'] )

        scenario_file = open( scenario_path, 'w' )

        if ac['purpose'] == 'delivery':
            if len( route ) == 2:
                createDeliveryFlightPlan( route[0], route[1], ac, departure_time, G, layers_dict,
                                          scenario_file, scenario_path, hovering_time=30 )
            else:
                createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )
        else:
            createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )

        scenario_file.close()

        text = 'PCALL {} REL'.format( '.' + scenario_path[10:] )

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
        segments_df = pd.DataFrame.from_dict( usepesegments.segments, orient='index' )
        while not self.flight_plan_df_buffer[self.flight_plan_df_buffer['planned_time_s'] <= sim.simt].empty:
            df_row = self.flight_plan_df_buffer.iloc[[0]]
            print( df_row )

            row = df_row.iloc[0]
            orig = [row['origin_lon'], row['origin_lat'], row['origin_alt'] ]
            dest = [row['destination_lon'], row['destination_lat'], row['destination_alt'] ]

            # We check if the origin/destination is in a no fly zone
            cond = ( segments_df['lon_min'] <= orig[0] ) & ( segments_df['lon_max'] > orig[0] ) & \
                ( segments_df['lat_min'] <= orig[1] ) & ( segments_df['lat_max'] > orig[1] ) & \
                ( segments_df['z_min'] <= orig[2] ) & ( segments_df['z_max'] > orig[2] )

            if segments_df[cond].empty:
                segment_name_0 = 'N/A'
            else:
                segment_name_0 = segments_df[cond].index[0]

            # We check which is the destination is in a no fly zone
            cond = ( segments_df['lon_min'] <= dest[0] ) & ( segments_df['lon_max'] > dest[0] ) & \
                ( segments_df['lat_min'] <= dest[1] ) & ( segments_df['lat_max'] > dest[1] ) & \
                ( segments_df['z_min'] <= dest[2] ) & ( segments_df['z_max'] > dest[2] )

            if segments_df[cond].empty:
                segment_name_f = 'N/A'
            else:
                segment_name_f = segments_df[cond].index[0]

            if ( usepesegments.segments[segment_name_0]['speed'] == 0 ) | ( usepesegments.segments[segment_name_f]['speed'] == 0 ):
                # origin or destination is not allowed, so the flight plan is rejected
                self.flight_plan_df_buffer = self.flight_plan_df_buffer.drop( self.flight_plan_df_buffer.index[0] )
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

        segments_df = pd.DataFrame.from_dict( usepesegments.segments, orient='index' )
        while not previous_df.empty:
            df_row = previous_df.iloc[[0]]
            print( df_row )

            row = df_row.iloc[0]
            orig = [row['origin_lon'], row['origin_lat'], row['origin_alt'] ]
            dest = [row['destination_lon'], row['destination_lat'], row['destination_alt'] ]

            # We check if the origin/destination is in a no fly zone
            cond = ( segments_df['lon_min'] <= orig[0] ) & ( segments_df['lon_max'] > orig[0] ) & \
                ( segments_df['lat_min'] <= orig[1] ) & ( segments_df['lat_max'] > orig[1] ) & \
                ( segments_df['z_min'] <= orig[2] ) & ( segments_df['z_max'] > orig[2] )

            if segments_df[cond].empty:
                segment_name_0 = 'N/A'
            else:
                segment_name_0 = segments_df[cond].index[0]

            # We check which is the destination is in a no fly zone
            cond = ( segments_df['lon_min'] <= dest[0] ) & ( segments_df['lon_max'] > dest[0] ) & \
                ( segments_df['lat_min'] <= dest[1] ) & ( segments_df['lat_max'] > dest[1] ) & \
                ( segments_df['z_min'] <= dest[2] ) & ( segments_df['z_max'] > dest[2] )

            if segments_df[cond].empty:
                segment_name_f = 'N/A'
            else:
                segment_name_f = segments_df[cond].index[0]

            if ( usepesegments.segments[segment_name_0]['speed'] == 0 ) | ( usepesegments.segments[segment_name_f]['speed'] == 0 ):
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
                drone_code == 'F'
            else:
                drone_code == 'M'

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
        qdr, dist = geo.kwikqdrdist_matrix( np.asmatrix( ownship.lat ), np.asmatrix( ownship.lon ),
                                    np.asmatrix( intruder.lat ), np.asmatrix( intruder.lon ) )

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
        rpz = np.asarray( np.maximum( np.asmatrix( rpz ), np.asmatrix( rpz ).transpose() ) )
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
        hpz = np.asarray( np.maximum( np.asmatrix( hpz ), np.asmatrix( hpz ).transpose() ) )
        tcrosshi = ( dalt + hpz ) / -dvs
        tcrosslo = ( dalt - hpz ) / -dvs
        tinver = np.minimum( tcrosshi, tcrosslo )
        toutver = np.maximum( tcrosshi, tcrosslo )

        # Combine vertical and horizontal conflict----------------------------------
        tinconf = np.maximum( tinver, tinhor )
        toutconf = np.minimum( toutver, touthor )

        swconfl = np.array( swhorconf * ( tinconf <= toutconf ) * ( toutconf > 0.0 ) *
                           np.asarray( tinconf < np.asmatrix( dtlookahead ).T ) * ( 1.0 - I ), dtype=np.bool )

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
        print( 'default' )
        print( self.confpairs_default )

        if usepeconfig.getboolean( 'BlueSky', 'D2C2' ):

            for i in range( len( swconfl ) ):
                for j in range( len( swconfl[i] ) ):
                    if swconfl[i][j]:
                        ow_ = ownship.id[i]
                        in_ = intruder.id[j]

                        # Calculate relative position
                        qdr_iter = qdr[i][j]
                        qdrrad_iter = np.radians( qdr_iter )
                        dx_iter = dist[i][j] * np.sin( qdrrad_iter )  # is pos j rel to i
                        dy_iter = dist[i][j] * np.cos( qdrrad_iter )  # is pos j rel to i

                        # Ownship speed
                        own_gs_iter = ownship.gs[i]  # m/s

                        own_gs_iter_table = min( self.table_gs_list, key=lambda x:abs( x - own_gs_iter ) )

                        # Intruder track angle and speed
                        int_gs_iter = intruder.gs[j]  # m/s

                        int_gs_iter_table = min( self.table_gs_list, key=lambda x:abs( x - int_gs_iter ) )

                        dx_grid = [math.floor( dx_iter / self.table_grid ), math.floor( dx_iter / self.table_grid ),
                                   math.ceil( dx_iter / self.table_grid ), math.ceil( dx_iter / self.table_grid )]

                        dy_grid = [math.floor( dy_iter / self.table_grid ), math.ceil( dy_iter / self.table_grid ),
                                   math.floor( dy_iter / self.table_grid ), math.ceil( dy_iter / self.table_grid )]

                        time_to_react_all_list = []
                        for manoeuvre in ['H', 'S', 'V']:
                            # select table
                            table = self.selectTable( ownship.type[i], manoeuvre, own_gs_iter_table )
                            time_to_react_list = []

                            for x, y in zip( dx_grid, dy_grid ):
                                if manoeuvre == 'S' and x == 0 and qdr_iter == 0:  # check this condition
                                    time_to_react_list.extend( [0] )
                                else:
                                    time_to_react_list.extend( self.checkTable( table, x, y, qdr_iter, own_gs_iter_table, int_gs_iter_table ) )
                            print( time_to_react_list )
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

            print( 'confpairs' )
            print( confpairs )

        # print( 'confpairs' )
        # print( confpairs )

        return confpairs, lospairs, inconf, tcpamax, \
            qdr[swconfl], dist[swconfl], np.sqrt( dcpa2[swconfl] ), \
                tcpa[swconfl], tinconf[swconfl]

