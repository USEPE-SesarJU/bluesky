#!/usr/bin/python

"""
A module responsible for strategic deconfliction
"""
from operator import add
from pickle import FALSE
import datetime
import math

from usepe.city_model.dynamic_segments import dynamicSegments
from usepe.city_model.path_planning import trajectoryCalculation, printRoute
from usepe.city_model.scenario_definition import createFlightPlan, calcDistAccel, routeParameters
from usepe.city_model.utils import checkIfNoFlyZone
import time as tm


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def initialPopulation( segments, t0, tf ):
    """
    Create an initial data structure with the information of how the segments are populated.
    The information is stored as a list for each segment: segment(j) = [x(t1), x(t2), x(t3),..., x(tn)],
    where x(t) represents the number of drones in the segment j during the second t.
    Initially, all values are zeros.

    Args:
            segments (dictionary): dictionary with all the information about segments
            t0 (integer): initial seconds of the flight plan time horizon
            tf (integer): final seconds of the flight plan time horizon

    Returns:
            users (dictionary): information of how the segments are populated from t0 to tf
    """
    users = {}
    empty_list = [0 for i in range( tf - t0 )]
    for idx, row in segments.iterrows():
        users[idx] = empty_list

    return users


def calcTravelTime( route_parameters, ac, step ):
    """
    Calculate the travel time of a segment considering the times needed to accelerate and decelerate

    Args:
            route_parameters (dictionary): dictionary with all the information about the route
            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max}
            step (integer): indicating the the step of the route

    Returns:
            t (float): travel time in seconds
    """
    m2nm = 0.000539957
    m_s2knot = 1.944

    if route_parameters[str( step - 1 )]['turn speed']:
        v0 = route_parameters[str( step - 1 )]['turn speed'] / m_s2knot
    elif step == 1:
        v0 = 0
    else:
        v0 = route_parameters[str( step - 2 )]['speed']

    if route_parameters[str( step )]['turn speed']:
        vn = route_parameters[str( step )]['turn speed'] / m_s2knot
        d3 = min( route_parameters[str( step )]['turn dist'] / m2nm, route_parameters[str( step - 1 )]['dist'] )
    else:
        vn = route_parameters[str( step - 1 )]['speed']
        d3 = 0

    v1 = route_parameters[str( step - 1 )]['speed']

    d1, t1 = calcDistAccel( v0, v1, ac )

    if d1 < route_parameters[str( step - 1 )]['dist'] - d3:
        d2 = route_parameters[str( step - 1 )]['dist'] - d1 - d3
        t2 = d2 / v1

        d3, t3 = calcDistAccel( v1, vn, ac )

    else:
        d1 = max( route_parameters[str( step - 1 )]['dist'] - d3, 0 )

        if v1 >= v0:
            accel = ac['accel']
        else:
            accel = -ac['accel']
        t1 = ( -2 * v0 / accel + math.sqrt( 4 * v0 ** 2 / accel ** 2 + 8 * d1 / accel ) ) / 2

        v1 = v0 + accel * t1

        if d3 == 0:
            t3 = 0
        else:
            d3, t3 = calcDistAccel( v1, vn, ac )

        d2 = max( route_parameters[str( step - 1 )]['dist'] - d1 - d3, 0 )
        t2 = d2 / vn

    t = t1 + t2 + t3

    return t


def droneAirspaceUsage( G, route, time, users_planned, initial_time, final_time,
                        route_parameters, ac ):
    """
    Computes how the new user populates the segments. It returns the information of how the segments
    are populated including the tentative flight plan of the new drone.

    Args:
            G (graph): a graph representing the city
            route (list): list containing the waypoints of the optimal route
            time (int): integer representing the departure time in seconds relative to initial_time
            users (dictionary): information of how the segments are populated from t0 to tf
            initial_time (int): integer representing the initial time in seconds of the period under
                study
            final_time (int): integer representing the final time in seconds of the period under
                study
            route_parameters (dictionary): dictionary with all the information about the route
            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max, safety_volume_size}

    Returns:
            users (dictionary): information of how the segments are populated from t0 to tf
                including the tentative flight plan of the new drone
    """
    users = users_planned.copy()
    actual_segment = None

    segments_updated = []

    actual_time = time
    t0 = time
    tf = actual_time
    step = 0
    for wpt2 in route:
        if not actual_segment:
            actual_segment = G.nodes[wpt2]['segment']

            segments_updated += [actual_segment]

            step += 1
            continue

        if G.nodes[wpt2]['segment'] == actual_segment:
            actual_time += calcTravelTime( route_parameters, ac, step )

            if wpt2 == route[-1]:
                tf = math.floor( actual_time )
                segment_list = [1 * ac['safety_volume_size'] if ( i >= t0 ) & ( i < tf ) else 0 for i in range( int( final_time - initial_time ) )]
                users[actual_segment] = list( map( add, users[actual_segment], segment_list ) )
        else:
            actual_time += calcTravelTime( route_parameters, ac, step )
            tf = math.floor( actual_time )
            segment_list = [1 * ac['safety_volume_size'] if ( i >= t0 ) & ( i < tf ) else 0 for i in range( int( final_time - initial_time ) )]

            users[actual_segment] = list( map( add, users[actual_segment], segment_list ) )

            t0 = math.floor( actual_time )
            actual_segment = G.nodes[wpt2]['segment']

            segments_updated += [actual_segment]

        step += 1

    try:
        if tf > final_time:
            print( 'Warning! Drone ends its trajectory at {0}, but the simulation ends at {1}'.format( tf, final_time ) )
    except:
        pass

    return users, segments_updated

def droneAirspaceUsageDelivery( G, route, time, users_planned, initial_time, final_time,
                                route_parameters, ac, hovering_time ):
    """
    Computes how the new user populates the segments. It returns the information of how the segments
    are populated including the tentative flight plan of the new drone.

    Args:
            G (graph): a graph representing the city
            route (list): list containing the waypoints of the optimal route
            time (int): integer representing the departure time in seconds relative to initial_time
            users (dictionary): information of how the segments are populated from t0 to tf
            initial_time (int): integer representing the initial time in seconds of the period under
                study
            final_time (int): integer representing the final time in seconds of the period under
                study
            route_parameters (dictionary): dictionary with all the information about the route
            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max, safety_volume_size}

    Returns:
            users (dictionary): information of how the segments are populated from t0 to tf
                including the tentative flight plan of the new drone
    """
    users = users_planned.copy()
    actual_segment = None
    actual_time = time
    t0 = time
    tf = actual_time
    step = 0
    for wpt2 in route:
        if not actual_segment:
            actual_segment = G.nodes[wpt2]['segment']
            step += 1
            continue

        if G.nodes[wpt2]['segment'] == actual_segment:
            actual_time += calcTravelTime( route_parameters, ac, step )

            if wpt2 == route[-1]:
                tf = math.floor( actual_time )
                segment_list = [1 * ac['safety_volume_size'] if ( i >= t0 ) & ( i < tf ) else 0 for i in range( int( final_time - initial_time ) )]
                users[actual_segment] = list( map( add, users[actual_segment], segment_list ) )
        else:
            actual_time += calcTravelTime( route_parameters, ac, step )
            tf = math.floor( actual_time )
            segment_list = [1 * ac['safety_volume_size'] if ( i >= t0 ) & ( i < tf ) else 0 for i in range( int( final_time - initial_time ) )]

            users[actual_segment] = list( map( add, users[actual_segment], segment_list ) )

            t0 = math.floor( actual_time )
            actual_segment = G.nodes[wpt2]['segment']

        if wpt2 == route[-1]:
            segment_list = [1 * ac['safety_volume_size'] if ( i >= tf ) & ( i < tf + hovering_time ) else 0 for i in range( int( final_time - initial_time ) )]
            users[actual_segment] = list( map( add, users[actual_segment], segment_list ) )

        step += 1

    if tf > final_time:
        print( 'Warning! Drone ends its trajectory at {0}, but the simulation ends at {1}'.format( tf, final_time ) )

    return users, tf + hovering_time


def checkOverpopulatedSegment( segments, users, initial_time, final_time, segments_updated=None ):
    """
    Check if any segment is overpopulated. It returns the segment name and the time when the segment
    gets overcrowded. If no segment is overpopulated, it retunrs None.

    Args:
            segments (dictionary): dictionary with the segment information
            users (dictionary): information of how the segments are populated from t0 to tf
            initial_time (int): integer representing the initial time in seconds of the period under
                study
            final_time (int): integer representing the final time in seconds of the period under
                study
            segments_updated (list): list indicating the segments used by the new drone

    Returns:
            overpopulated_segment (string): segment name
            overpopulated_time (int): time when the segment gets overcrowded. Value in seconds and
                relative to the initial time.

    """
    overpopulated_segment = None
    overpopulated_time = None
    cond = False

    if segments_updated:
        segments_reduced = segments[segments.index.isin( segments_updated )]
    else:
        segments_reduced = segments

    for i in range( final_time - initial_time ):
        for idx, row in segments_reduced.iterrows():
            capacity = row['capacity']
            if users[idx][i] > capacity:
                overpopulated_segment = str( idx )
                overpopulated_time = i
                cond = True
                print( 'The segment {0} is overpopulated at {1} seconds'.format( 
                    overpopulated_segment, overpopulated_time ) )
                break
        if cond:
            break

    return overpopulated_segment, overpopulated_time


def deconflictedPathPlanning( orig, dest, time, G, users, initial_time, final_time, segments,
                              config, ac, only_rerouting=False, delivery=False, hovering_time=30 ):
    """
    Computes an optimal flight plan without exceeding the segment capacity limit. The procedure
    consist in:
    1. Compute optimal path from origin to destination.
    2. While including the new drone a segment capacity limit is exceeded:
        2.1. A sub-optimal trajectory is computed without considering the overpopulated segment.
        2.2. If the travel time of the sub-optimal trajectory divided by the optimal travel time is
            higher than a configurable threshold:
            2.2.1. The flight is delayed by a configurable value.
            2.2.2. Repeat step 2 with the new departure time.
    3. It returns the flight plan, the departure time and the new information about how the segments
        are populated
    Args:
            orig (list): with the coordinates of the origin point [longitude, latitude]
            dest (list): with the coordinates of the destination point [longitude, latitude]
            time (int): integer representing the departure time in seconds relative to initial_time
            G (graph): a graph representing the city
            users (dictionary): information of how the segments are populated from initial time to
                final time.
            initial_time (int): integer representing the initial time in seconds of the period under
                study
            final_time (int): integer representing the final time in seconds of the period under
                study
            segments (dictionary): dictionary with the segment information

            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max}

    Returns:
            users_step (dictionary): information of how the segments are populated from initial time to
                final time including the deconflcited trajectory of the new dorne.
            route (list): list containing the waypoints of the optimal route
            delayed_time (int): indicating how many seconds the flight is delayed respect to the
                desired departure time

    """
    # avg_flight_time = 20 * 60

    delayed_time = time
    opt_travel_time, route = trajectoryCalculation( G, orig, dest )

    route_parameters = routeParameters( G, route, ac )

    users_step, segments_updated = droneAirspaceUsage( G, route, delayed_time, users, initial_time, final_time,
                                     route_parameters, ac )


    start = tm.time()
    overpopulated_segment, overpopulated_time = checkOverpopulatedSegment( 
        segments, users_step,
        initial_time, final_time,
        segments_updated=segments_updated )

    if overpopulated_segment:
        print( 'Drone {} needs to be strategically deconflicted'.format( ac['id'] ) )
    # print( 'Overpopulated segments time', tm.time() - start, 's' )

    segments_step = segments.copy()
    G_step = G.copy()
    while overpopulated_segment:
        if type( overpopulated_segment ) == str:
            if segments_step['speed_max'][int( overpopulated_segment )] == 0:
                print( 'It is impossible to find a permitted route for the drone {}. It is not included in the simulation'.format( ac['id'] ) )
                return users, route, final_time + 1
            segments_step['speed_max'][int( overpopulated_segment )] = 0
            segments_step['updated'][int( overpopulated_segment )] = True

            G_step, segments_step = dynamicSegments( G_step, None, segments_step )

        # b, route = trajectoryCalculation( G_step, orig, dest )
        # print( b )

        travel_time, route = trajectoryCalculation( G_step, orig, dest )
        route_parameters = routeParameters( G_step, route, ac )

        if ( travel_time / opt_travel_time > config['Strategic_Deconfliction'].getint( 'ratio' ) ) and not only_rerouting:
            delayed_time += config['Strategic_Deconfliction'].getint( 'delay' )
            overpopulated_segment = True
            segments_step = segments.copy()
            G_step = G.copy()
            # G_step, segments = dynamicSegments( G_step, None, segments )
            # G_step, segments_step = dynamicSegments( G_step, None, segments_step )
            # a, route = trajectoryCalculation( G, orig, dest )
            # print( a )
            # b, route = trajectoryCalculation( G_step, orig, dest )
            # print( b )
            # print( a / b )
            print( 'The flight is delayed {0} seconds'.format( delayed_time - time ) )
        else:
            users_step, segments_updated = droneAirspaceUsage( G_step, route, delayed_time, users, initial_time,
                                             final_time, route_parameters, ac )

            overpopulated_segment, overpopulated_time = checkOverpopulatedSegment( 
                segments_step, users_step,
                initial_time, final_time,
                segments_updated=segments_updated )

    if delivery:
        users_step, departure2 = droneAirspaceUsageDelivery( G, route, delayed_time, users,
                                                             initial_time, final_time,
                                                             route_parameters, ac, hovering_time )
        return users_step, route, delayed_time, departure2

    return users_step, route, delayed_time


def deconflictedDeliveryPathPlanning( orig1, dest1, orig2, dest2, time, G, users, initial_time,
                                      final_time, segments, config, ac, hovering_time=30,
                                      only_rerouting=False ):
    """
    Computes an optimal flight plan without exceeding the segment capacity limit. The procedure
    consist in:
    1. Compute optimal path from origin to destination.
    2. While including the new drone a segment capacity limit is exceeded:
        2.1. A sub-optimal trajectory is computed without considering the overpopulated segment.
        2.2. If the travel time of the sub-optimal trajectory divided by the optimal travel time is
            higher than a configurable threshold:
            2.2.1. The flight is delayed by a configurable value.
            2.2.2. Repeat step 2 with the new departure time.
    3. It returns the flight plan, the departure time and the new information about how the segments
        are populated
    Args:
            orig (list): with the coordinates of the origin point [longitude, latitude]
            dest (list): with the coordinates of the destination point [longitude, latitude]
            time (int): integer representing the departure time in seconds relative to initial_time
            G (graph): a graph representing the city
            users (dictionary): information of how the segments are populated from initial time to
                final time.
            initial_time (int): integer representing the initial time in seconds of the period under
                study
            final_time (int): integer representing the final time in seconds of the period under
                study
            segments (dictionary): dictionary with the segment information

            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max}

    Returns:
            users_step (dictionary): information of how the segments are populated from initial time to
                final time including the deconflcited trajectory of the new dorne.
            route (list): list containing the waypoints of the optimal route
            delayed_time (int): indicating how many seconds the flight is delayed respect to the
                desired departure time

    """

    users1, route1, delayed_time1, departure2 = deconflictedPathPlanning( orig1, dest1, time, G, users,
                                                           initial_time, final_time, segments,
                                                           config, ac, only_rerouting=only_rerouting,
                                                           delivery=True,
                                                           hovering_time=hovering_time )

    users2, route2, delayed_time2 = deconflictedPathPlanning( orig2, dest2, departure2, G, users1,
                                                              initial_time, final_time, segments,
                                                              config, ac, only_rerouting=True )

    return users2, [route1, route2], delayed_time1


def deconflictedSurveillancePathPlanning( orig1, dest1, orig2, dest2, departure_time, G, users, initial_time,
                                        final_time, segments, config, ac, only_rerouting=False, wait_time=600 ):
    """
    This function

    Args:
        - orig (list): Coordinates of origin [longitude, latitude]
        - dest (list): Coordinates of destination [longitude, latitude]
        - departure_time (int): Departure time in seconds, relative to initial_time
        - G (graph): Graph representing the area
        - users (dict): Information on how segments are populated for the entire duration
        - initial_time (int): Initial time of simulation in seconds
        - final_time (int): Final time of simulation in seconds
        - segments (dict): Segment information
        - config (Config): Configuration of the simulation
        - ac (dict): Aircraft parameters
        - only_rerouting (Boolean): True if rerouting an existing plan
        - wait_time (int): How long before making the return trip after arrival

    Returns:
        - users2 (dict): Information on how segments are populated for the entire duration, including this latest flight
        - route (list): Waypoints of the optimal route, for both trips
        - delayed_time1 (int): Delay of the flight (first trip) in seconds
    """

    return users, [], departure_time


def deconflcitedScenario( orig, dest, ac, departure_time, G, users, initial_time, final_time,
                          segments, layers_dict, scenario_file, config ):
    """
    A strategic deconflicted trajectory from origin to destination is computed and a BluSky scenario
    is generated.

    Args:
            orig (list): with the coordinates of the origin point [longitude, latitude]
            dest (list): with the coordinates of the destination point [longitude, latitude]
            ac (dictionary): aircraft parameters {id, type, accel, v_max, vs_max}
            departure_time (int): integer representing the departure time in seconds relative to initial_time
            G (graph): a graph representing the city
            users (dictionary): information of how the segments are populated from initial time to
                final time.
            initial_time (int): integer representing the initial time in seconds of the period under
                study
            final_time (int): integer representing the final time in seconds of the period under
                study
            segments (dictionary): dictionary with the segment information
            layers_dict (dictionary): dictionary with the information about layers and altitudes
            scenario_file (object): text file object where the commands are written

    Returns:
            users (dictionary): information of how the segments are populated from initial time to
                final time.
    """

    users, route, delayed_time = deconflictedPathPlanning( orig, dest, departure_time, G, users,
                                                           initial_time, final_time, segments,
                                                           config, ac )

    departure_time = str( datetime.timedelta( seconds=delayed_time ) )

    createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )

    return users, route


if __name__ == '__main__':
    pass
