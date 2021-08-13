#!/usr/bin/python

"""

"""

__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'

import random
import time

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


def defineSegment( segments, lon_min, lon_max, lat_min, lat_max, z_min, z_max,
                   speed, capacity, name ):
    segments[name] = {'lon_min': lon_min,
                      'lon_max': lon_max,
                      'lat_min': lat_min,
                      'lat_max': lat_max,
                      'z_min': z_min,
                      'z_max': z_max,
                      'speed': speed,
                      'capacity': capacity,
                      'new': True,
                      'updated': True}
    return segments


def divideAirspaceSegments( lon_min, lon_max, lat_min, lat_max, z_min, z_max, divisions_lon,
                            divisions_lat, divisions_z ):

    print( 'Creating segments...' )
    delta_lon = ( lon_max - lon_min ) / divisions_lon
    delta_lat = ( lat_max - lat_min ) / divisions_lat
    delta_z = ( z_max - z_min ) / divisions_z

    segments = {}
    for i in range( divisions_lon ):
        for j in range( divisions_lat ):
            for k in range( divisions_z ):
                name = 'segment_' + str( i ) + '_' + str( j ) + '_' + str( k )
                lon_min_seg = lon_min + i * delta_lon
                lon_max_seg = lon_min + ( i + 1 ) * delta_lon
                lat_min_seg = lat_min + j * delta_lat
                lat_max_seg = lat_min + ( j + 1 ) * delta_lat
                z_min_seg = z_min + k * delta_z
                z_max_seg = z_min + ( k + 1 ) * delta_z
                speed_seg = float( random.randint( 30, 40 ) )
                capacity_seg = random.randint( 0, 20 )

                segments = defineSegment( segments, lon_min_seg, lon_max_seg, lat_min_seg,
                                          lat_max_seg, z_min_seg, z_max_seg, speed_seg,
                                          capacity_seg, name )

    segments = defineSegment( segments, 0, 0, 0,
                              0, 0, 0, 0.0,
                              0, 'N/A' )

    return segments


def selectNodesWithNewSegments( G, segments ):
    new_segments = list( segments.keys() )

    nodes = ox.graph_to_gdfs( G, edges=False, node_geometry=False )
    cond = nodes['segment'] == 'new'
#     for n in new_segments:
#         cond = cond | ( nodes['segment'] == n )
    cond = cond | ( nodes['segment'].isin( new_segments ) )
    df = nodes[cond]
    return df.index


def assignSegmet2Edge( G, segments_df ):

    if segments_df.empty:
        print( 'No new segments' )
        return G

    print( 'Assigning segments...' )
#     segments_df = pd.DataFrame.from_dict( segments, orient='index' )

    nodes_affected = selectNodesWithNewSegments( G, segments_df )
    for node in nodes_affected:
        if node[0:2] == 'COR':
            continue

        node_lon = G.nodes[node]['x']
        node_lat = G.nodes[node]['y']
        node_z = G.nodes[node]['z']

        # We chech which is the segment associated to the node
        cond = ( segments_df['lon_min'] <= node_lon ) & ( segments_df['lon_max'] > node_lon ) & \
            ( segments_df['lat_min'] <= node_lat ) & ( segments_df['lat_max'] > node_lat ) & \
            ( segments_df['z_min'] <= node_z ) & ( segments_df['z_max'] > node_z )

        if segments_df[cond].empty:
            segment_name = 'N/A'
        else:
            segment_name = segments_df[cond].index[0]
        G.nodes[node]['segment'] = segment_name
        connected_edges = list( G.neighbors( node ) )
        for edge in connected_edges:
            G.edges[node, edge, 0 ]['segment'] = segment_name

    return G


def updateSegmentVelocity( G, segments ):
    if segments.empty:
        print( 'No new velocities' )
        return G
    print( 'Updating segment velocity...' )

    new_segments = list( segments.keys() )

    edges = ox.utils_graph.graph_to_gdfs( G, nodes=False, fill_edge_geometry=False )
    cond = edges['segment'] == 'N/A'
#     for n in new_segments:
#         cond = cond | ( edges['segment'] == n )
    cond = cond | ( edges['segment'].isin( new_segments ) )
    edges[cond]['speed'] = edges[cond]['segment'].apply( lambda segment_name:
                                                         segments[segment_name]['speed'] )


    nx.set_edge_attributes( G, values=edges["speed"], name="speed" )
    return G


def addTravelTimes( G, precision=4 ):
    print( 'Updating travel times...' )
    edges = ox.utils_graph.graph_to_gdfs( G, nodes=False )

    # verify edge length and speed_kph attributes exist and contain no nulls
    if not ( "length" in edges.columns and "speed" in edges.columns ):
        raise KeyError( "all edges must have `length` and `speed` attributes." )
    else:
        if ( pd.isnull( edges["length"] ).any() or pd.isnull( edges["speed"] ).any() ):
            raise ValueError( "edge `length` and `speed_kph` values must be non-null." )

    # convert distance meters to km, and speed km per hour to km per second
    distance_km = edges["length"] / 1000
    speed_km_sec = edges["speed"] / ( 60 * 60 )

    # calculate edge travel time in seconds
    travel_time = distance_km / speed_km_sec

    # add travel time attribute to graph edges
    edges["travel_time"] = travel_time.round( precision ).values
    nx.set_edge_attributes( G, values=edges["travel_time"], name="travel_time" )

    return G


def dynamicSegments( G, config, segments=None ):
    print( 'Updating segments...' )
    if not segments:
        segments = divideAirspaceSegments( config['City'].getfloat( 'hannover_lon_min' ),
                                           config['City'].getfloat( 'hannover_lon_max' ),
                                           config['City'].getfloat( 'hannover_lat_min' ),
                                           config['City'].getfloat( 'hannover_lat_max' ),
                                           0,
                                           config['Layers'].getfloat( 'layer_width' ) *
                                           ( config['Layers'].getfloat( 'number_of_layers' ) + 1 ),
                                           4, 4, 2 )

    segments_df = pd.DataFrame.from_dict( segments, orient='index' )
    new_segments = segments_df[segments_df['new'] == True]
    updated_segments = segments_df[segments_df['updated'] == True]
    start = time.time()
    G = assignSegmet2Edge( G, new_segments )
    end = time.time()
    print( end - start )
    start = time.time()
    G = updateSegmentVelocity( G, updated_segments )
    end = time.time()
    print( end - start )
    G = addTravelTimes( G )

    segments_df['new'] = False
    segments_df['updated'] = False

    segments = segments_df.to_dict( orient='index' )
    print( 'Dynamic segments completed' )
    return G, segments


if __name__ == '__main__':
    filepath = "./data/hannover.graphml"
    from auxiliar import read_my_graphml
    from multi_di_graph_3D import MultiDiGrpah3D
    G = read_my_graphml( filepath )
    G = MultiDiGrpah3D( G )
    segments = divideAirspaceSegments( 0, 20, 50, 54, 0, 250, 4, 4, 2 )
    G = assignSegmet2Edge( G, segments )
    G = updateSegmentVelocity( G, segments )
    edges = ox.graph_to_gdfs( G, nodes=False )
    print( edges['segment'] )
    print( edges.columns )
    print( edges['speed'] )
    print( segments['segment_1_2_0'] )
    G = addTravelTimes( G )

    from path_planning import trajectoryCalculation
    orig = ( 9.74 , 52.36 )
    dest = ( 9.78 , 53.38 )
    print( trajectoryCalculation( G, orig, dest ) )



