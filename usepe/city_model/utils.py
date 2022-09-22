#!/usr/bin/python

"""
Additional functions
"""
import copy
import math
import string

from osmnx.io import _convert_node_attr_types, _convert_bool_string, _convert_edge_attr_types

from usepe.city_model.multi_di_graph_3D import MultiDiGrpah3D
import networkx as nx
import osmnx as ox


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def read_my_graphml( filepath ):
    """
    Read a previously computed graph
    Args:
            filepath (string): string representing the path where the graph is stored
    Returns:
            G (graph): graph stored at the filepath
    """

    default_node_dtypes = {
        "elevation": float,
        "elevation_res": float,
        "lat": float,
        "lon": float,
        "street_count": int,
        "x": float,
        "y": float,
        "z": float,
    }
    default_edge_dtypes = {
        "bearing": float,
        "grade": float,
        "grade_abs": float,
        "length": float,
        "oneway": _convert_bool_string,
        "osmid": int,
        "speed": float,
        "travel_time": float,
        "maxspeed": float,
    }

    # read the graphml file from disk
    print( 'Reading the graph...' )
    G = nx.read_graphml( filepath, force_multigraph=True )

    # convert graph/node/edge attribute data types
    G.graph.pop( "node_default", None )
    G.graph.pop( "edge_default", None )
    G = _convert_node_attr_types( G, default_node_dtypes )
    G = _convert_edge_attr_types( G, default_edge_dtypes )
    G = MultiDiGrpah3D( G )

    return G


def layersDict( config ):
    """
    Create a dictionary with the information about the altitude of each layer
    Args:
            config (configuration file): A configuration file with all the relevant information
    Returns:
            layers_dict (dict): dictionary with keys=layers values=altitude [m]
    """
    letters = list( string.ascii_uppercase )
    total_layers = letters[0:config['Layers'].getint( 'number_of_layers' )]
    layer_width = config['Layers'].getint( 'layer_width' )
    altitude = 0
    layers_dict = {}
    for layer in total_layers:
        altitude += layer_width
        layers_dict[layer] = altitude

    return layers_dict

def nearestNode3d( G, lon, lat, altitude, exclude_corridor=True ):
    '''
    This function gets the closest node of the city graph nodes with respect
    to a given reference point (lat, lon, alt)

    Input:
        G - graph
        lon - longitude of the reference point
        lat - latitude of the reference point
        altitude - altitude of the reference point

    Output:
        nearest_node - closest node of the city graph nodes with respect to the reference point
        distance - distance between the nearest node and the reference point (lat, lon, alt)
    '''
    # The nodes are filtered to exclude corridor nodes
    nodes = list( G.nodes )
    if exclude_corridor:
        filtered_latlon = list( filter( lambda node: str( node )[:3] != 'COR', nodes ) )
    else:
        filtered_latlon = nodes
    
    # Filter out nodes belonging to the 'priority' segment (these are strictly used by premade scenarios)
    nodes = G.nodes
    filter_priority = list(filter(lambda node: nodes[node]['segment'] == 'priority', nodes))
    filtered_latlon = [x for x in filtered_latlon if x not in filter_priority]

    # Iterates to get the closest one
    nearest_node = filtered_latlon[0]
    delta_xyz = ( ( G.nodes[nearest_node]['z'] - altitude ) ** 2 +
                  ( G.nodes[nearest_node]['y'] - lat ) ** 2 +
                  ( G.nodes[nearest_node]['x'] - lon ) ** 2 )

    for node in filtered_latlon[1:]:
        delta_xyz_aux = ( ( G.nodes[node]['z'] - altitude ) ** 2 +
                          ( G.nodes[node]['y'] - lat ) ** 2 +
                          ( G.nodes[node]['x'] - lon ) ** 2 )
        if delta_xyz_aux < delta_xyz:
            delta_xyz = delta_xyz_aux
            nearest_node = node
    return nearest_node

def checkIfNoFlyZone( lat, lon, alt, G, segments ):
    '''
    This function checks if the point or its nearest node is within a no-fly zone
    '''
    if type( segments ) == dict:
        # Get closed segments
        closed_segments = {}
        for segment_id, segment in segments.items():
            if segment['speed'] == 0:
                closed_segments[segment_id] = segment
        # Check if the point is inside a no-fly zone
        for segment_id, segment in closed_segments.items():
            # Origin
            if lat > segment['lat_min'] and  lat < segment['lat_max']:
                if lon > segment['lon_min'] and  lon < segment['lon_max']:
                    print( 'Point in no fly zone: lat {0}, lon {1}'.format( lat, lon ) )
                    return True
        # Check if the closest node of the graph is inside a no-fly zone
        if alt == None:
            nearest_node = ox.distance.nearest_nodes( G, X=lon, Y=lat )
        else:
            nearest_node = nearestNode3d( G, lon, lat, alt )
        speed = segments[G.nodes[nearest_node]['segment']]['speed']
        cap = segments[G.nodes[nearest_node]['segment']]['capacity']
        if speed == 0:
            return True
        return False
    else:
        closed_segments = segments[segments['speed_max'] == 0]
        for idx, row in closed_segments.iterrows():
            if lat > row['lat_min'] and  lat < row['lat_max']:
                if lon > row['lon_min'] and  lon < row['lon_max']:
                    print( 'Point in no fly zone: lat {0}, lon {1}'.format( lat, lon ) )
                    return True
        # Check if the closest node of the graph is inside a no-fly zone
        if alt == None:
            nearest_node = ox.distance.nearest_nodes( G, X=lon, Y=lat )
        else:
            nearest_node = nearestNode3d( G, lon, lat, alt )
        if G.nodes[nearest_node]['segment'].isdigit():
            segmentIndex = int( G.nodes[nearest_node]['segment'] )
            speed = segments.loc[segmentIndex]['speed_max']
            cap = segments.loc[segmentIndex]['capacity']
            if speed == 0:
                return True
        else:
            # print("Skipping node with segment '" +G.nodes[nearest_node]['segment'] + "'!")
            return True
        return False

def shortest_dist_to_point( x1, y1, x2, y2, x, y ):
    '''
    This function gets the shortest distance from a point (x,y) to a line defined by two points:
    (x1,y1) and (x2,y2)

    Input:
        x1 (float): x coordinate
        y1 (float): y coordinate
        x2 (float): x coordinate
        y2 (float): y coordinate
        x (float): x coordinate
        y (float): y coordinate

    Output:
        shortest distance from a point (x,y) to a line
    '''
    dx = x2 - x1
    dy = y2 - y1
    dr2 = float( dx ** 2 + dy ** 2 )

    lerp = ( ( x - x1 ) * dx + ( y - y1 ) * dy ) / dr2
    if lerp < 0:
        lerp = 0
    elif lerp > 1:
        lerp = 1

    x0 = lerp * dx + x1
    y0 = lerp * dy + y1

    _dx = x0 - x
    _dy = y0 - y
    square_dist = _dx ** 2 + _dy ** 2
    return math.sqrt( square_dist )


def wpt_bsc2wpt_graph( wpt_route_bsc, wpt_route_graph ):
    '''
    This function relates the name of the wpts in BlueSky with the names of the
    waypoints in the graph for a given aircraft

    wpt_route_bsc - list with waypoints names as bluesky loads them
    wpt_route_graph - list of waypoints forming the route of the drone in the graph
                        if it is a delivery drone, it will include two lists: go and back

    wpt_bsc2graph_dict - dictionary relating the names of the waypoints in bsk (keys)
                        with the names in the graph (values)
    '''

    if type( wpt_route_graph[0] ) is list:
        # Delivery case
        for route in wpt_route_graph:
            # clean route
            route_clean = cleanRoute( route )
            # select corresponding route
            if route_clean[-1] == wpt_route_bsc[-1]:
                # create dict
                wpt_dict = createDictWpt( wpt_route_bsc, route_clean )
            else:
                continue
    else:
        # clean route
        route_clean = cleanRoute( wpt_route_graph )
        # create dict
        wpt_dict = createDictWpt( wpt_route_bsc, route_clean )

    return wpt_dict

def cleanRoute( wpt_route_graph ):
    wpt_route_graph_clean = copy.deepcopy( wpt_route_graph )

    for wpt_graph in wpt_route_graph[1:]:
        idx = wpt_route_graph_clean.index( wpt_graph )
        if wpt_route_graph_clean[idx][1:] == wpt_route_graph_clean[idx - 1][1:]:
            del wpt_route_graph_clean[idx]
    if len( wpt_route_graph_clean ) > 1:
        del wpt_route_graph_clean[0]
    return wpt_route_graph_clean

def createDictWpt( wpt_route_bsc, wpt_route_graph_clean ):
    wpt_dict = {}
    for wpt_bsc, wpt_graph in zip( wpt_route_bsc, wpt_route_graph_clean ):
        wpt_dict[wpt_bsc] = wpt_graph

    return wpt_dict

if __name__ == '__main__':
    pass
