#!/usr/bin/python

"""
A module responsible for strategic deconfliction
"""
from operator import add
import math

from nommon.city_model.dynamic_segments import dynamicSegments
from nommon.city_model.path_planning import trajectoryCalculation, printRoute
from nommon.city_model.scenario_definition import createFlightPlan


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
    for key in segments:
        users[key] = empty_list

    return users


def droneAirspaceUsage( G, route, time, users_planned, initial_time, final_time ):
    users = users_planned.copy()
    actual_segment = None
    actual_time = time
    t0 = time
    for wpt2 in route:
        if not actual_segment:
            actual_segment = G.nodes[wpt2]['segment']
            wpt1 = wpt2
            continue

        if G.nodes[wpt2]['segment'] == actual_segment:
            actual_time += G.edges[wpt1, wpt2, 0]['travel_time']

            if wpt2 == route[-1]:
                tf = math.floor( actual_time )
                segment_list = [1 if ( i >= t0 ) & ( i < tf ) else 0 for i in range( final_time - initial_time )]
                users[actual_segment] = list( map( add, users[actual_segment], segment_list ) )
        else:
            actual_time += G.edges[wpt1, wpt2, 0]['travel_time']
            tf = math.floor( actual_time )
            segment_list = [1 if ( i >= t0 ) & ( i < tf ) else 0 for i in range( final_time - initial_time )]

            users[actual_segment] = list( map( add, users[actual_segment], segment_list ) )

            t0 = math.floor( actual_time )
            actual_segment = G.nodes[wpt2]['segment']

        wpt1 = wpt2

    if tf > final_time:
        print( 'Warning! Drone ends its trajectory at {0}, but the simulation ends at {1}'.format( tf, final_time ) )

    return users


def checkOverpopulatedSegment( segments, users, initial_time, final_time ):
    overpopulated_segment = None
    overpopulated_time = None
    cond = False
    for i in range( final_time - initial_time ):
        for key in segments:
            capacity = segments[key]['capacity']
            if users[key][i] > capacity:
                overpopulated_segment = key
                overpopulated_time = i
                cond = True
                print( 'The segment {0} is overpopulated at {1} seconds'.format( 
                    overpopulated_segment, overpopulated_time ) )
                break
        if cond:
            break

    return overpopulated_segment, overpopulated_time


def deconflictedPathPlanning( orig, dest, time, G, users, initial_time, final_time, segments ):
    delayed_time = time
    opt_travel_time, route = trajectoryCalculation( G, orig, dest )

    users_step = droneAirspaceUsage( G, route, delayed_time, users, initial_time, final_time )

    overpopulated_segment, overpopulated_time = checkOverpopulatedSegment( 
        segments, users_step, initial_time, final_time )

    segments_step = segments.copy()
    G_step = G.copy()
    while overpopulated_segment:
        if type( overpopulated_segment ) == str:
            print( 'hola' )
            segments_step[overpopulated_segment]['speed'] = 0
            segments_step[overpopulated_segment]['updated'] = True

            G_step, segments_step = dynamicSegments( G_step, None, segments_step )

        # b, route = trajectoryCalculation( G_step, orig, dest )
        # print( b )

        travel_time, route = trajectoryCalculation( G_step, orig, dest )
        # print( travel_time / opt_travel_time )

        users_step = droneAirspaceUsage( G_step, route, delayed_time, users, initial_time, final_time )

        overpopulated_segment, overpopulated_time = checkOverpopulatedSegment( 
            segments_step, users_step, initial_time, final_time )

        if travel_time / opt_travel_time > 3:
            delayed_time += 60
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
            print( 'The fligth is delayed {0} seconds'.format( delayed_time ) )

    return users_step, route, delayed_time


def deconflcitedScenario( orig, dest, ac, departure_time, G, users, initial_time, final_time,
                          segments, layers_dict, scenario_file ):

    users, route, delayed_time = deconflictedPathPlanning( orig, dest, departure_time, G, users,
                                                           initial_time, final_time, segments )

    departure_time = delayed_time
    createFlightPlan( route, ac, departure_time, G, layers_dict, scenario_file )

    return users


if __name__ == '__main__':
    pass