import math

from networkx.algorithms.components.connected import connected_components
from shapely.geometry import Polygon, box

from usepe.segmentation_service.python.lib import rectangles
import geopandas as gpd
import networkx as nx
import pandas as pd


def simplify( features, rules ):
    def to_edges( l ):
        it = iter( l )
        last = next( it )
        for current in it:
            yield last, current
            last = current

    def close_holes( poly ):
        if poly.interiors:
            return Polygon( list( poly.exterior.coords ) )
        else:
            return poly

    def min_size_rect( bounds, minL ):
        bounds.maxx = math.ceil( bounds.maxx / minL ) * minL
        bounds.maxy = math.ceil( bounds.maxy / minL ) * minL
        bounds.minx = math.floor( bounds.minx / minL ) * minL
        bounds.miny = math.floor( bounds.miny / minL ) * minL
        poly = box( *list( bounds ) )
        return poly

    features = features.copy()
    features = features.sort_index()
    features.geometry = [
        min_size_rect( bounds, rules["min_grid"] )
        for _, bounds in features.geometry.bounds.iterrows()
    ]
    # features.geometry = [box(*list(bounds)) for _, bounds in features.geometry.bounds.iterrows()]

    neighbor_graph = nx.Graph()
    for _, row in features.iterrows():
        classfeat = features[features["class"] == row["class"]]
        intersects = classfeat.geometry.intersects( row.geometry )
        intersections = classfeat.intersection( row.geometry )

        neighbors = classfeat[intersects & ( intersections.type != "Point" )].index.tolist()
        neighbor_graph.add_nodes_from( neighbors )
        neighbor_graph.add_edges_from( to_edges( neighbors ) )

    features["cluster"] = -1
    for cluster_index, neighborhood in enumerate( connected_components( neighbor_graph ) ):
        for idx in neighborhood:
            features["cluster"].at[idx] = cluster_index

    simple = features.dissolve( 
        by="cluster",
        aggfunc={
            "class": "first",
            "type": lambda s: "; ".join( set( s ) ),
            "name": lambda s: "; ".join( set( filter( lambda x: x != "nan", s ) ) ),
            "z_min": "min",
            "z_max": "max",
            "speed_min": "max",
            "speed_max": "min",
        },
    )
    simple = gpd.GeoDataFrame( simple, geometry=[close_holes( poly ) for poly in simple.geometry] )
    simple = simple[simple["class"] != "grey"]  # HACK remove large airspace structures
    return simple


def deoverlap( simple, rules ):
    classes = rules["classes"]

    cutout = None
    stacked = None
    grouped = simple.groupby( "class" )
    available = grouped.groups.keys()
    for class_name in classes:
        if class_name not in available:
            continue
        group = grouped.get_group( class_name )
        if cutout is None:
            cutout = group
            stacked = group
        else:
            group = group.overlay( cutout, "difference" ).explode( index_parts=True )
            cutout = pd.concat( [cutout, group] )
            stacked = pd.concat( [stacked, group], ignore_index=True )
    return stacked


def fillspace( region, stacked, rules ):
    cutout = stacked.dissolve()
    region = region.difference( cutout.at[0, "geometry"] )
    defClass = list( rules["classes"] )[-1]
    spaced = gpd.GeoDataFrame( 
        {
            "class": defClass,
            "type": "space",
            "name": "",
            "z_min": min( rules["classes"][defClass]["altitude"] ),
            "z_max": max( rules["classes"][defClass]["altitude"] ),
            "speed_min": min( rules["classes"][defClass]["velocity"] ),
            "speed_max": max( rules["classes"][defClass]["velocity"] ),
            "geometry": region.geoms if region.geom_type == "MultiPolygon" else [region],
        }
    )
    return pd.concat( [spaced, stacked], ignore_index=True ).set_crs( "EPSG:4326" )


def dissect( spaced ):
    segments = None
    for _, feature in spaced.iterrows():
        rects = rectangles.decompose( feature.geometry )
        decomp = gpd.GeoDataFrame( 
            {
                "class": feature["class"],
                "z_min": feature["z_min"],
                "z_max": feature["z_max"],
                "speed_max": feature["speed_max"],
                "speed_min": feature["speed_min"],
                "geometry": rects.geoms,
            }
        )
        if segments is None:
            segments = decomp
        else:
            segments = pd.concat( [segments, decomp], ignore_index=True )

    return segments
