#!/usr/bin/python

"""
Auxiliar functions for reading graphs
"""
from osmnx.io import _convert_node_attr_types, _convert_bool_string, _convert_edge_attr_types

from multi_di_graph_3D import MultiDiGrpah3D
import networkx as nx


__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


def read_my_graphml( filepath ):
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


if __name__ == '__main__':
    pass
