from datetime import datetime
from heapq import nlargest
import json
import math

from scipy.interpolate import griddata, interp2d
from shapely.geometry import LineString, Point
from shapely.ops import split

import geopandas as gpd
import numpy as np
import xarray as xr


def split_cell( segments, idx, splitter, keepParent=True, min_grid=1.0 ):
    """Divide the cell at idx using either a supplied splitter geometry or direction (x|y)"""
    cells = segments.copy()
    selected = cells.loc[idx]

    # Prepare the pre-defined splitter if a direction was given
    if splitter in ["x", "y"]:
        x = selected.geometry.exterior.bounds[0::2]
        y = selected.geometry.exterior.bounds[1::2]
        if splitter == "x":
            splitter = LineString( 
                [
                    Point( min( x ), np.round( ( sum( y ) / 2 ) / min_grid ) * min_grid ),
                    Point( max( x ), np.round( ( sum( y ) / 2 ) / min_grid ) * min_grid ),
                ]
            )
        elif splitter == "y":
            splitter = LineString( 
                [
                    Point( np.round( ( sum( x ) / 2 ) / min_grid ) * min_grid, min( y ) ),
                    Point( np.round( ( sum( x ) / 2 ) / min_grid ) * min_grid, max( y ) ),
                ]
            )

    # Perform the split and update the cell dataframe while creating the necessary links between parents and children
    divided = split( selected.geometry, splitter )
    floor_ceil = 0
    for idx_rect in range( len( divided.geoms ) ):
        new_cell = selected.copy()
        new_idx = len( segments.index ) + idx_rect
        new_cell["parent"] = idx
        new_cell["geometry"] = divided.geoms[idx_rect]
        new_cell["capacity"] = ( 
            new_cell["capacity"] * new_cell["geometry"].area / selected["geometry"].area
        )
        if floor_ceil:
            new_cell["capacity"] = math.ceil( new_cell["capacity"] )
        else:
            new_cell["capacity"] = math.floor( new_cell["capacity"] )

        cells.loc[new_idx] = new_cell
        if cells.loc[idx, "children"] is None:
            cells.loc[idx, "children"] = [[new_idx]]
        else:
            cells.loc[idx, "children"].append( new_idx )
        floor_ceil ^= 1

    if not keepParent:
        cells = cells.drop( idx )
        cells = cells.reset_index( drop=True )
    return cells


def split_alt( segments, idx, splitter, keepParent=True ):
    cells = segments.copy()
    selected = cells.loc[idx]
    if splitter == "z":
        splitter = ( selected.z_max - selected.z_min ) / 2.0

    # Perform the split and update the cell dataframe while creating the necessary links between parents and children
    floor_ceil = 0
    for idx_rect in range( 2 ):
        new_cell = selected.copy()
        new_idx = len( segments.index ) + idx_rect
        new_cell["parent"] = idx
        if floor_ceil:
            new_cell["z_min"] = splitter
            new_cell["capacity"] = math.ceil( 
                new_cell["capacity"]
                * ( new_cell["z_max"] - new_cell["z_min"] )
                / ( selected["z_max"] - selected["z_min"] )
            )
        else:
            new_cell["z_max"] = splitter
            new_cell["capacity"] = math.floor( 
                new_cell["capacity"]
                * ( new_cell["z_max"] - new_cell["z_min"] )
                / ( selected["z_max"] - selected["z_min"] )
            )

        cells.loc[new_idx] = new_cell
        if cells.loc[idx, "children"] is None:
            cells.loc[idx, "children"] = [[new_idx]]
        else:
            cells.loc[idx, "children"].append( new_idx )
        floor_ceil ^= 1

    if not keepParent:
        cells = cells.drop( idx )
        cells = cells.reset_index( drop=True )

    return cells


def split_aspect_ratio( cells, rules ):
    # split cells to achieve minimal aspect ratio
    # it might seem a good idea to split each segment as many times as required
    # but that is approximatelly two times slower than this approach
    maxAspect = rules["aspect_ratio"]
    min_grid = rules["min_grid"]
    if maxAspect < 1:
        maxAspect = 1.0 / maxAspect

    notEmpty = True
    while notEmpty:
        aspect = []
        idx = []
        for ii in range( len( cells ) ):
            x, y = cells.loc[ii].geometry.exterior.coords.xy
            if ( max( y ) - min( y ) ) > 2.0 * min_grid:
                aspect.append( ( max( x ) - min( x ) ) / ( max( y ) - min( y ) ) )
            else:
                aspect.append( 1e10 )

        idx = [iidx for iidx in range( len( aspect ) ) if ( ( 1.0 / aspect[iidx] ) >= maxAspect )]

        if len( idx ) != 0:
            for ii in range( len( idx ) ):
                cells = split_cell( cells, idx[ii] - ii, "x", False, min_grid )

        aspect = []
        idy = []
        for ii in range( len( cells ) ):
            x, y = cells.loc[ii].geometry.exterior.coords.xy
            if ( max( x ) - min( x ) ) > 2.0 * min_grid:
                aspect.append( ( max( x ) - min( x ) ) / ( max( y ) - min( y ) ) )
            else:
                aspect.append( 1e-10 )

        idy = [iidy for iidy in range( len( aspect ) ) if ( ( aspect[iidy] ) > maxAspect )]

        if len( idy ) != 0:
            for ii in range( len( idy ) ):
                cells = split_cell( cells, idy[ii] - ii, "y", False, min_grid )

        if len( idx ) == 0 & len( idy ) == 0:
            notEmpty = False

    return cells


def split_build( cells, building_layer ):
    kk = 0
    for ii in range( len( cells ) ):
        if ( 
            ( cells.at[kk, "z_min"] < building_layer )
            & ( cells.at[kk, "z_max"] > building_layer )
            & ( cells.at[kk, "capacity"] > 0 )
        ):
            cells = split_alt( 
                cells, kk, building_layer, False
            )  # kk -> erasing the parent and addinf chindren at the end
        else:
            kk += 1

    return cells


def cells2file( cells, file ):
    # when saving to GeoJSON cells cannot contain lists
    # only "children" can contain list of cell's children -> children are forgotten before saving
    for jj in range( len( cells ) ):
        if cells.loc[jj, "children"] is not None:
            cells.loc[jj, "children"] = None
    cells.to_file( file, driver="GeoJSON" )


def update_wind( cells, windData, interp_UTM=False ):
    # calculates wind magnitude from staggered wind vector
    # and assigns average and maximum wind magnitute to each airspace cell
    # indexing is done by ignoring the warping of the UTM coordinate system
    # -> maximal error approx 5 meters

    # assumes interpolated wind data to equidistant grid
    windData.load()
    # remove time axix
    windData["u"] = windData["u"][0, ...]
    windData["v"] = windData["v"][0, ...]
    windData["w"] = windData["w"][0, ...]

    nLat = len( windData["lat"][0,:] )
    nLon = len( windData["lon"][:, 0] )
    nAlt = len( windData["zu_3d"] )

    midLat = math.ceil( len( windData["lat"][0,:] ) / 2 )
    midLon = math.ceil( len( windData["lon"][:, 0] ) / 2 )

    # # calculate scalar wind speed from staggered wind velocities
    # # scalar values in grid center -> interpolate into the grid center
    windData["speed"] = np.sqrt( 
        np.power( windData["u"].interp( y=windData.coords["yv"], zu_3d=windData.coords["zw_3d"] ), 2 )
        +np.power( 
            windData["v"].interp( x=windData.coords["xu"], zu_3d=windData.coords["zw_3d"] ), 2
        )
        +np.power( windData["w"].interp( x=windData.coords["xu"], y=windData.coords["yv"] ), 2 )
    )

    if interp_UTM:
        windLat = np.linspace( windData["lat"].min(), windData["lat"].max(), nLat )
        windLon = np.linspace( windData["lon"].min(), windData["lon"].max(), nLon )
        grid_lat, grid_lon = np.meshgrid( windLat, windLon )

        windSpeed_interp = np.empty( ( nAlt, nLon, nLat ) )
        for ii in range( nAlt ):
            print( "interpolating for altitude number:", ii )
            windSpeed_interp[ii, ...] = griddata( 
                ( windData["lat"].to_numpy().ravel(), windData["lon"].to_numpy().ravel() ),
                windData["speed"].to_numpy()[ii,:,:].ravel(),
                ( grid_lat, grid_lon ),
            )
    else:
        windLat = windData["lat"][:, midLat].to_numpy()
        windLon = windData["lon"][midLon,:].to_numpy()

    # # zu_3d and zw_3d are both relative to the height level of origin_z. origin_z is the altitude above sea level.
    # windAlt = (windData["zu_3d"] + windData.origin_z).to_numpy()  # 0 m at sea level
    windAlt = ( windData["zu_3d"] ).to_numpy()  # 0 m at "ground" level

    dLat = np.mean( np.diff( windLat ) )
    dLon = np.mean( np.diff( windLon ) )
    dAlt = np.mean( np.diff( windAlt ) )

    for ii in cells.index:
        lon_min = cells.geometry[ii].bounds[0]
        lat_min = cells.geometry[ii].bounds[1]
        lon_max = cells.geometry[ii].bounds[2]
        lat_max = cells.geometry[ii].bounds[3]
        alt_min = cells.z_min[ii]
        alt_max = cells.z_max[ii]

        if ( 
            ( lon_max > windLon[0] )
            & ( lon_min < windLon[-1] )
            & ( lat_max > windLat[0] )
            & ( lat_min < windLat[-1] )
            & ( alt_max > windAlt[0] )
            & ( alt_min < windAlt[-1] )
        ):
            if lon_min < windLon[0]:
                lon_start = 0
            else:
                lon_start = math.floor( ( lon_min - windLon[0] ) / dLon )
            if lon_max > windLon[-1]:
                lon_end = nLon - 1
            else:
                lon_end = nLon - math.ceil( ( windLon[-1] - lon_max ) / dLon )

            if lat_min < windLat[0]:
                lat_start = 0
            else:
                lat_start = math.floor( ( lat_min - windLat[0] ) / dLat )
            if lat_max > windLat[-1]:
                lat_end = nLat - 1
            else:
                lat_end = nLat - math.ceil( ( windLat[-1] - lat_max ) / dLat )

            if alt_min < windAlt[0]:
                alt_start = 0
            else:
                alt_start = math.floor( ( alt_min - windAlt[0] ) / dAlt )
            if alt_max > windAlt[-1]:
                alt_end = nAlt - 1
            else:
                alt_end = nAlt - math.ceil( ( windAlt[-1] - alt_max ) / dAlt )

            if interp_UTM:
                cellWind = windSpeed_interp[
                    alt_start:alt_end, lat_start:lat_end, lon_start:lon_end
                ]
                cells.at[ii, "wind_avg"] = np.round( 100 * np.nanmean( cellWind ) ) / 100
                cells.at[ii, "wind_max"] = np.round( 100 * np.nanmax( cellWind ) ) / 100
            else:
                cellWind = windData["speed"][
                    alt_start:alt_end, lat_start:lat_end, lon_start:lon_end
                ]
                cells.at[ii, "wind_avg"] = np.round( 100 * cellWind.mean().to_numpy() ) / 100
                cells.at[ii, "wind_max"] = np.round( 100 * cellWind.max().to_numpy() ) / 100

    return True


def process_wind_data( wind_file, interp_UTM=False ):

    # assumes interpolated wind data to equidistant grid
    windData = xr.open_dataset( ( wind_file + ".nc" ) )
    windData.load()
    # remove time axix
    windData["u"] = windData["u"][0, ...]
    windData["v"] = windData["v"][0, ...]
    windData["w"] = windData["w"][0, ...]

    nLat = len( windData["lat"][0,:] )
    nLon = len( windData["lon"][:, 0] )
    nAlt = len( windData["zu_3d"] )

    midLat = math.ceil( len( windData["lat"][0,:] ) / 2 )
    midLon = math.ceil( len( windData["lon"][:, 0] ) / 2 )

    # # calculate scalar wind speed from staggered wind velocities
    # # scalar values in grid center -> interpolate into the grid center
    windData["speed"] = np.sqrt( 
        np.power( windData["u"].interp( y=windData.coords["yv"], zu_3d=windData.coords["zw_3d"] ), 2 )
        +np.power( 
            windData["v"].interp( x=windData.coords["xu"], zu_3d=windData.coords["zw_3d"] ), 2
        )
        +np.power( windData["w"].interp( x=windData.coords["xu"], y=windData.coords["yv"] ), 2 )
    )

    if interp_UTM:
        windLat = np.linspace( windData["lat"].min(), windData["lat"].max(), nLat )
        windLon = np.linspace( windData["lon"].min(), windData["lon"].max(), nLon )
        grid_lat, grid_lon = np.meshgrid( windLat, windLon )
        windSpeed_interp = np.empty( ( nAlt, nLon, nLat ) )
        for ii in range( nAlt ):
            print( "interpolating for altitude number:", ii )
            windSpeed_interp[ii, ...] = griddata( 
                ( windData["lat"].to_numpy().ravel(), windData["lon"].to_numpy().ravel() ),
                windData["speed"].to_numpy()[ii,:,:].ravel(),
                ( grid_lat, grid_lon ),
            )
        windData["speed_interp"] = xr.DataArray( 
            windSpeed_interp,
            coords=[( windData["zu_3d"] ).data, windLon, windLat],
            dims=["alt_proc", "lat_proc", "lon_proc"],
        )
        # aa = xr.DataArray(
        #     windData["speed"].data,
        #     coords=[windData["alt_proc"].data, windData["lon_proc"].data, windData["lat_proc"].data],
        #     dims=["alt_proc", "lat_proc", "lon_proc"],
        # )
    else:
        windLat = windData["lat"][:, midLat].to_numpy()
        windLon = windData["lon"][midLon,:].to_numpy()

    # # zu_3d and zw_3d are both relative to the height level of origin_z. origin_z is the altitude above sea level.
    # windAlt = (windData["zu_3d"] + windData.origin_z).to_numpy()  # 0 m at sea level
    windAlt = ( windData["zu_3d"] ).to_numpy()  # 0 m at "ground" level

    windData["dLat"] = np.mean( np.diff( windLat ) )
    windData["dLon"] = np.mean( np.diff( windLon ) )
    windData["dAlt"] = np.mean( np.diff( windAlt ) )

    windData["nLat"] = nLat
    windData["nLon"] = nLon
    windData["nAlt"] = nAlt

    windData["lat_proc"] = windLat
    windData["lon_proc"] = windLon
    windData["alt_proc"] = windAlt

    # windData["speed"] = windData["speed"].rename_dims(
    #     {"xu": "lon_proc", "yv": "lat_proc", "zw_3d": "alt_proc"}
    # )
    # aa = xr.DataArray(
    #     windData["speed"].data,
    #     coords=[windData["alt_proc"].data, windData["lon_proc"].data, windData["lat_proc"].data],
    #     dims=["alt_proc", "lat_proc", "lon_proc"],
    # )
    # windData = windData.drop_vars("speed")
    # windData["speed"] = aa
    return windData


def load_rules( rule_file="usepe/segmentation_service/config/rules.json" ):
    with open( rule_file, "r" ) as tags_file:
        rules = json.load( tags_file )
    return rules
