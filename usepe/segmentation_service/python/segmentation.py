from datetime import datetime
import json

from numpy import NaN
import hvplot
import hvplot.pandas

from usepe.segmentation_service.python.lib import air, autoseg, ground, misc, polygons
import geopandas as gpd
import osmnx as ox
import pandas as pd


def create_cell( rect ):
    pass


def load_rules( rule_file="config/rules.json" ):
    with open( rule_file, "r" ) as tags_file:
        rules = json.load( tags_file )
    return rules


def init_region( region ):
    _SHAPELY_TOLERANCE = 0.01  # [deg] Used to simplify the bounding polygon of the region

    # Make sure the region is in the right format
    if isinstance( region, str ):
        region = ox.geocode_to_gdf( region )
    if isinstance( region, gpd.GeoDataFrame ):
        region = region.geometry[0]
        region = region.buffer( 2 * _SHAPELY_TOLERANCE )  # increases size of all polygons
        region = region.simplify( _SHAPELY_TOLERANCE, preserve_topology=False )
    region, _ = polygons.orthogonal_bounds( region )
    return region


def eval_capacity_km_sq( segments, rules ):
    capDensity = float( rules["capacity_km_sq"] )
    segments["capacity"] = 0
    for name, data in rules["classes"].items():
        segments.loc[segments["class"] == name, "capacity"] = round( 
            segments.loc[segments["class"] == name].geometry.area
            * ( capDensity * ( 111 ** 2 ) )
            * rules["classes"][name]["capacity_factor"]
        )

    return segments


def eval_capacity_m_3( segments, rules ):
    capDensity = float( rules["capacity_m_3"] )
    segments["capacity"] = 0
    for name, data in rules["classes"].items():
        segments.loc[segments["class"] == name, "capacity"] = round( 
            segments.loc[segments["class"] == name].geometry.area
            * ( 111139 ** 2 )
            * ( 
                segments.loc[segments["class"] == name, "z_max"]
                -segments.loc[segments["class"] == name, "z_min"]
            )
            * capDensity
            * rules["classes"][name]["capacity_factor"]
        )

    return segments


def init_cells( region ):
    rules = load_rules()
    region = init_region( region )
    ground_data = ground.get( region, rules )
    air_data = air.get( region, rules )
    features = pd.concat( [ground_data, air_data] )
    simple = autoseg.simplify( features, rules )
    stacked = autoseg.deoverlap( simple, rules )
    spaced = autoseg.fillspace( region, stacked, rules )
    segments = autoseg.dissect( spaced )

    # evaluate segment capacity
    segments = eval_capacity_m_3( segments, rules )
    # Prepare the segments to make the cells divisible
    segments["parent"] = None
    segments["children"] = None

    # add update properties
    segments["new"] = False
    segments["updated"] = False
    segments["wind_avg"] = -1.0
    segments["wind_max"] = -1.0

    return segments


if __name__ == "__main__":
    rules = load_rules()
    region = "Hannover"
    start = datetime.now()
    cells = init_cells( region )
    print( "duration of segmentation took ", datetime.now() - start )

    start = datetime.now()
    cells = misc.split_aspect_ratio( cells, rules["aspect_ratio"] )
    cells = misc.split_build( cells, rules["building_layer"] )
    print( "duration of initial split took ", datetime.now() - start )

    plot = cells.hvplot( 
        c="class",
        geo=True,
        frame_height=1000,
        tiles="CartoDark",
        hover_cols=["z_min", "z_max", "capacity"],
        alpha=0.2,
    )
    cells.to_file( ( "data/examples/" + region + ".geojson" ), driver="GeoJSON" )
    hvplot.show( plot )
