""" This plugin load the graph for the USEPE project. """
# Import the global bluesky objects. Uncomment the ones you need
import configparser
import copy
import datetime
import math
import os
import pickle

from bluesky import core, traf, stack, sim  # , settings, navdb,  scr, tools
from usepe.city_model.dynamic_segments import dynamicSegments
from usepe.city_model.scenario_definition import createFlightPlan, createDeliveryFlightPlan
from usepe.city_model.strategic_deconfliction import initialPopulation, deconflictedPathPlanning, deconflictedDeliveryPathPlanning
from usepe.city_model.utils import read_my_graphml, layersDict
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


# ## Initialisation function of your plugin. Do not change the name of this
# ## function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''

    # Configuration parameters
    config = {
        'plugin_name': 'USEPE',
        'plugin_type': 'sim',
        'update_interval': 1.0,

        # The update function is called after traffic is updated.
        'update': update,

        # The preupdate function is called before traffic is updated.
        'preupdate': preupdate,

        # Reset
        'reset': reset }
    
    stackfunctions = {
        'USEPE': [
            'USEPE CONFIG/ON/OFF, [config_path]',
            'txt, [txt]',
            usepe,
            'Set path to configuration file, or turn on/off the plugin.'
        ]
    }

    # init_plugin() should always return a configuration dict.
    return config, stackfunctions


def update():
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


def usepe(cmd, args=''):
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

    if cmd == 'CONFIG':
        if args == '':
            return False, f'"USEPE CONFIG" needs a valid path to configuration file.'

        config_path = args
        usepeconfig = configparser.ConfigParser()
        usepeconfig.read(config_path)

        graph_path = usepeconfig['BlueSky']['graph_path']
        segment_path = usepeconfig['BlueSky']['segment_path']
        flight_plan_csv_path = usepeconfig['BlueSky']['flight_plan_csv_path']

        initial_time = int(usepeconfig['BlueSky']['initial_time'])
        final_time = int(usepeconfig['BlueSky']['final_time'])

        usepegraph = UsepeGraph(graph_path)
        usepesegments = UsepeSegments(segment_path)
        usepestrategic = UsepeStrategicDeconfliction(initial_time, final_time)
        usepeflightplans = UsepeFlightPlan(flight_plan_csv_path)
        usepedronecommands = UsepeDroneCommands()


        return True, f'The configuration file has been set.'

    elif cmd == 'ON':
        if usepeconfig is not None:
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
        for key in self.segments:
            if self.segments[key]['speed'] == 0:
                self.segments[key]['speed'] = 5
                self.segments[key]['capacity'] = 1
                self.segments[key]['updated'] = True
        usepegraph.graph, self.segments = dynamicSegments( usepegraph.graph, usepeconfig, self.segments, deleted_segments=None )
        #####

    def dynamicSegments( self ):
        """
        TODO. Here we have to include the function which updates the segments
        """
        updated = False
        if not usepeconfig.getboolean( 'BlueSky', 'D2C2' ):
            return updated, self.segments

        if ( sim.simt > 30 ) & ( sim.simt < 32 ):
            updated = True
        segments = self.segments

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

                mask = usepeflightplans.flight_plan_df_back_up['ac'] == acid

                latf = usepeflightplans.flight_plan_df_back_up[mask].iloc[0]['destination_lat']
                lonf = usepeflightplans.flight_plan_df_back_up[mask].iloc[0]['destination_lon']
                altf = usepeflightplans.flight_plan_df_back_up[mask].iloc[0]['destination_alt']

                orig = [lon0, lat0, alt0 ]
                dest = [lonf, latf, altf ]

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

        # TODO: to include more drone purposes (e.g., surveillance, emergency, etc.)
        self.delivery_drones = 0
        self.background_drones = 0

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
                name = row['purpose'].upper() + str( self.delivery_drones )
                self.delivery_drones += 1
            elif row['purpose'] == 'background':
                name = row['purpose'].upper() + str( self.background_drones )
                self.background_drones += 1

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
            v_max = 41
            vs_max = 6
            safety_volume_size = 1

        ac = {'id': name, 'type': row['drone'], 'accel': 3.5, 'v_max': v_max, 'vs_max': vs_max,
              'safety_volume_size': safety_volume_size, 'purpose': row['purpose']}

        if ac['purpose'] == 'delivery':
            users, route, delayed_time = deconflictedDeliveryPathPlanning( orig, dest, dest, orig,
                                                                           departure_time, usepegraph.graph,
                                                                           self.users, self.initial_time,
                                                                           self.final_time,
                                                                           copy.deepcopy( usepesegments.segments ),
                                                                           usepeconfig, ac, hovering_time=30,
                                                                           only_rerouting=False )
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

        scenario_path = r'.\scenario\usepe\temp\scenario_traffic_drone_{}.scn'.format( ac['id'] )
        scenario_file = open( scenario_path, 'w' )

        if ac['purpose'] == 'delivery':
            createDeliveryFlightPlan( route[0], route[1], ac, departure_time, G, layers_dict,
                                      scenario_file, scenario_path, hovering_time=30 )
        else:
            createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )

        scenario_file.close()

        stack.stack( 'PCALL {} REL'.format( '.' + scenario_path[10:] ) )

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
            createDeliveryFlightPlan( route[0], route[1], ac, departure_time, G, layers_dict,
                                      scenario_file, scenario_path, hovering_time=30 )
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
    flight_plan_df_buffer: DataFrame with all the flights that have been processed (simt > planned_time_s)
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
        while not self.flight_plan_df_buffer[self.flight_plan_df_buffer['planned_time_s'] <= sim.simt].empty:
            df_row = self.flight_plan_df_buffer.iloc[[0]]
            print( df_row )
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

        while not previous_df.empty:
            df_row = previous_df.iloc[[0]]
            print( df_row )
            usepestrategic.strategicDeconflictionDrone( df_row, new=False )

            previous_df = previous_df.drop( previous_df.index[0] )


    def update( self ):  # Not used
        return

    def preupdate( self ):
        self.processFlightPlans()
        return

    def reset( self ):  # Not used
        return
