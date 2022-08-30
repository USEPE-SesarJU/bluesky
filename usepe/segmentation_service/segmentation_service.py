from threading import Thread
import math
import pickle
import time

from numpy import NaN
from shapely.geometry import LineString, Point
from tqdm import tqdm
import hvplot
import hvplot.pandas

from usepe.segmentation_service.python.utils import air, autoseg, ground, misc, polygons
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import xarray as xr


# from python.utils import air, autoseg, ground, misc, polygons
class segmentationService:
    def __init__( self, region ):
        self.rules = misc.load_rules()
        self.region_name = region
        self.init_cells()
        print( "Region ", region, " initialized." )

        self.cells = misc.split_aspect_ratio( self.cells, self.rules )
        print( "Initial split assuring maximal aspect ratio ", self.rules["aspect_ratio"], " DONE" )

        self.cells = misc.split_build( self.cells, self.rules["building_layer"] )
        print( "Altitude split at building top level ", self.rules["building_layer"], " m DONE." )

        # # reset history of splitting during the initialization stage
        self.cells = self.cells.reset_index( drop=True )

        self.cells["parent"] = None
        self.cells["children"] = None
        self.cells["new"] = False
        self.cells["updated"] = False

        self.event_update = False
        return

    def init_cells( self ):
        self.init_region()
        ground_data = ground.get( self.region, self.rules )
        air_data = air.get( self.region, self.rules )
        features = pd.concat( [ground_data, air_data] )
        simple = autoseg.simplify( features, self.rules )
        stacked = autoseg.deoverlap( simple, self.rules )
        spaced = autoseg.fillspace( self.region, stacked, self.rules )
        self.cells = autoseg.dissect( spaced )

        # adding segment relevant data
        self.cells["capacity"] = 0
        self.cells["occupancy"] = 0
        self.cells["conflicts"] = 0
        self.cells["aoc"] = 0
        self.cells[
            "geovect"
        ] = "NSEW"  # allowed directions - N -> allowed from South to North, W -> allowed from East to West, etc.
        # Prepare the cells to make the cells divisible
        self.cells["parent"] = None
        self.cells["children"] = None
        # add update properties
        self.cells["class_init"] = self.cells["class"].copy()
        self.cells["new"] = False
        self.cells["updated"] = False
        self.cells["wind_avg"] = -1.0
        self.cells["wind_max"] = -1.0
        self.cells["turbulence"] = -1.0  # turbulence intensity

        # evaluate segment capacity
        self.eval_capacity_m_3()

        return

    def init_region( self ):
        _SHAPELY_TOLERANCE = 0.01  # [deg] Used to simplify the bounding polygon of the region
        # Make sure the region is in the right format
        region = self.region_name
        if isinstance( self.region_name, str ):
            region = ox.geocode_to_gdf( region )
        if isinstance( region, gpd.GeoDataFrame ):
            region = region.geometry[0]
            region = region.buffer( 2 * _SHAPELY_TOLERANCE )  # increases size of all polygons
            region = region.simplify( _SHAPELY_TOLERANCE, preserve_topology=False )
        self.region, _ = polygons.orthogonal_bounds( region, self.rules["min_grid"] )
        return

    def eval_capacity_km_sq( self ):
        capDensity = float( self.rules["capacity_km_sq"] )
        for name, data in self.rules["classes"].items():
            self.cells.loc[self.cells["class"] == name, "capacity"] = round( 
                self.cells.loc[self.cells["class"] == name].geometry.area
                * ( capDensity * ( 111 ** 2 ) )
                * self.rules["classes"][name]["capacity_factor"]
            )

        return

    def eval_capacity_m_3( self ):
        capDensity = float( self.rules["capacity_m_3"] )
        for name, _ in self.rules["classes"].items():
            self.cells.loc[self.cells["class"] == name, "capacity"] = round( 
                self.cells.loc[self.cells["class"] == name].geometry.area
                * ( 111139 ** 2 )
                * ( 
                    self.cells.loc[self.cells["class"] == name, "z_max"]
                    -self.cells.loc[self.cells["class"] == name, "z_min"]
                )
                * capDensity
                * self.rules["classes"][name]["capacity_factor"]
            )
        return

    def eval_capacity_m_3_id( self, id ):
        capDensity = float( self.rules["capacity_m_3"] )
        self.cells.at[id, "capacity"] = round( 
            self.cells.loc[id].geometry.area
            * ( 111139 ** 2 )
            * ( self.cells.loc[id]["z_max"] - self.cells.loc[id]["z_min"] )
            * capDensity
            * self.rules["classes"][self.cells.loc[id]["class"]]["capacity_factor"]
        )
        return

    def close_cell( self, id ):
        # name, data in rules["classes"].items():
        n = ( name for name, data in self.rules["classes"].items() )
        # self.cells.iloc[idx,"class"]= self.rules[]
        # first class in rules is forbiden even for emenrgency
        # closing cell switches to the second class that is allowed for emergency
        if self.cells.at[id, "class"] != next( n ):
            close_class = next( n )
            if self.cells.at[id, "class"] != close_class:
                self.cells.at[id, "class"] = close_class
                self.cells.at[id, "speed_min"] = min( 
                    self.rules["classes"][close_class]["velocity"]
                )
                self.cells.at[id, "speed_max"] = max( 
                    self.rules["classes"][close_class]["velocity"]
                )
                self.cells.at[id, "updated"] = True

                self.eval_capacity_m_3_id( id )

        return

    def restore_cell( self, id ):
        self.cells.at[id, "class"] = self.cells.at[id, "class_init"]
        self.cells.at[id, "speed_min"] = min( 
            self.rules["classes"][self.cells.at[id, "class"]]["velocity"]
        )
        self.cells.at[id, "speed_max"] = max( 
            self.rules["classes"][self.cells.at[id, "class"]]["velocity"]
        )
        self.cells.at[id, "updated"] = True

        self.eval_capacity_m_3_id( id )

        return

    def decrease_speed( self, id ):
        sp = [
            self.rules["allowed_uas_speed"][isp]
            for isp in range( len( self.rules["allowed_uas_speed"] ) )
            if ( self.rules["allowed_uas_speed"][isp] < self.cells.at[id, "speed_max"] )
        ]
        if sp:
            self.cells.at[id, "speed_max"] = max( sp )
        else:
            self.cells.at[id, "speed_max"] = 0.0

        self.cells.at[id, "updated"] = True
        return

    def update_wind_strat( self, wind_file, interp_UTM=False ):
        # strategic wind data-based update method
        # wind file is assumed in ".data/wind/" + windFile + ".nc" (suffix added automatically)
        # interp_UTM chooses whether the wind data are approximated or interpolated into WGS84 Lat, Lon, Alt coordinates
        # wind data is assigned to airspace cells in:
        # self.cells["wind_avg"] - average wind speed (magnitude of wind vector) inside each cell
        # self.cells["wind_max"] - maximum wind speed (magnitude of wind vector) inside each cell
        # self.cells["turbulence"] - wind turbulence intensity in each cell obtained as standard deviation of wind speed inside cell / average wind speed

        # rules are applied based on "wind_avg" and "turbulence" parameter of each cell using update_wind_rules() method
        # self.rules["wind_rules"]["wind_speed_th"] - wind speed threshold parameter in rules.json file
        # self.rules["wind_rules"]["turbulence_intensity_th"] - wind turbulence intensity threshold parameter in rules.json file

        print( "Staring to process wind data." )
        wind_data = misc.process_wind_data( 
            ( self.rules["wind_rules"]["wind_data_folder"] + wind_file ), interp_UTM
        )
        print( "Processing wind data is DONE." )
        nAlt = int( wind_data["nAlt"] )
        nLon = int( wind_data["nLon"] )
        nLat = int( wind_data["nLat"] )

        print( "Assigning wind data to cells:" )
        for id in tqdm( self.cells.index ):
            lon_min = self.cells.geometry[id].bounds[0]
            lat_min = self.cells.geometry[id].bounds[1]
            lon_max = self.cells.geometry[id].bounds[2]
            lat_max = self.cells.geometry[id].bounds[3]
            alt_min = self.cells.z_min[id]
            alt_max = self.cells.z_max[id]

            if ( 
                ( lon_max > wind_data["lon_proc"][0] )
                & ( lon_min < wind_data["lon_proc"][-1] )
                & ( lat_max > wind_data["lat_proc"][0] )
                & ( lat_min < wind_data["lat_proc"][-1] )
                & ( alt_max > wind_data["alt_proc"][0] )
                & ( alt_min < wind_data["alt_proc"][-1] )
            ):
                if lon_min < wind_data["lon_proc"][0]:
                    lon_start = 0
                else:
                    lon_start = math.floor( 
                        ( lon_min - wind_data["lon_proc"][0] ) / wind_data["dLon"]
                    )
                if lon_max > wind_data["lon_proc"][-1]:
                    lon_end = nLon - 1
                else:
                    lon_end = nLon - math.ceil( 
                        ( wind_data["lon_proc"][-1] - lon_max ) / wind_data["dLon"]
                    )

                if lat_min < wind_data["lat_proc"][0]:
                    lat_start = 0
                else:
                    lat_start = math.floor( 
                        ( lat_min - wind_data["lat_proc"][0] ) / wind_data["dLat"]
                    )
                if lat_max > wind_data["lat_proc"][-1]:
                    lat_end = nLat - 1
                else:
                    lat_end = nLat - math.ceil( 
                        ( wind_data["lat_proc"][-1] - lat_max ) / wind_data["dLat"]
                    )

                if alt_min < wind_data["alt_proc"][0]:
                    alt_start = 0
                else:
                    alt_start = math.floor( 
                        ( alt_min - wind_data["alt_proc"][0] ) / wind_data["dAlt"]
                    )
                if alt_max > wind_data["alt_proc"][-1]:
                    alt_end = nAlt - 1
                else:
                    alt_end = nAlt - math.ceil( 
                        ( wind_data["alt_proc"][-1] - alt_max ) / wind_data["dAlt"]
                    )

                if interp_UTM:
                    windSpeed = xr.DataArray( 
                        wind_data["speed_interp"].data,
                        coords=[
                            wind_data["alt_proc"].data,
                            wind_data["lat_proc"].data,
                            wind_data["lon_proc"].data,
                        ],
                        dims=["alt_proc", "lat_proc", "lon_proc"],
                    )
                    cellWind = windSpeed[alt_start:alt_end, lat_start:lat_end, lon_start:lon_end]
                    wind_mean = np.nanmean( cellWind )

                    self.cells.at[id, "wind_avg"] = np.round( 100 * wind_mean ) / 100
                    self.cells.at[id, "wind_max"] = np.round( 100 * np.nanmax( cellWind ) ) / 100
                    self.cells.at[id, "turbulence"] = ( 
                        np.round( 100 * np.nanstd( cellWind ) / wind_mean ) / 100
                    )
                else:
                    windSpeed = xr.DataArray( 
                        wind_data["speed"].data,
                        coords=[
                            wind_data["alt_proc"].data,
                            wind_data["lat_proc"].data,
                            wind_data["lon_proc"].data,
                        ],
                        dims=["alt_proc", "lat_proc", "lon_proc"],
                    )
                    cellWind = windSpeed[alt_start:alt_end, lat_start:lat_end, lon_start:lon_end]
                    wind_mean = cellWind.mean().to_numpy()
                    wind_max = cellWind.max().to_numpy()
                    self.cells.at[id, "wind_avg"] = np.round( 100 * wind_mean ) / 100
                    self.cells.at[id, "wind_max"] = np.round( 100 * wind_max ) / 100
                    self.cells.at[id, "turbulence"] = ( 
                        np.round( 100 * cellWind.std().to_numpy() / wind_mean ) / 100
                    )
                    # speed_argmax = cellWind.where(cellWind == wind_max, drop=True).squeeze()
        for id in self.cells.index:
            self.update_wind_rules( id )

        return

    def update_wind_rules( self, id ):
        if self.cells.at[id, "wind_avg"] > 0:
            # # strategic wind speed-dependent cell update
            if self.cells.at[id, "wind_avg"] > self.rules["wind_rules"]["wind_speed_th"]:
                self.close_cell( id )
            elif self.cells.at[id, "wind_avg"] > 0.75 * self.rules["wind_rules"]["wind_speed_th"]:
                self.cells.at[id, "speed_max"] /= 2.0
                self.cells.at[id, "updated"] = True
                # self.decrease_speed(id)

            # # strategic wind turbulence-dependent cell update
            if ( 
                self.cells.at[id, "turbulence"]
                > self.rules["wind_rules"]["turbulence_intensity_th"]
            ):
                self.close_cell( id )
            elif ( 
                self.cells.at[id, "turbulence"]
                > 0.75 * self.rules["wind_rules"]["turbulence_intensity_th"]
            ):
                self.cells.at[id, "speed_max"] /= 2.0
                self.cells.at[id, "updated"] = True
                # self.decrease_speed(id)
        return

    def update_wind_tact( self, plan, log_pos, log_w, id, city ):
        print( "Starting the tactical wind field dependency update." )
        # log_w ... log of active waypoints, actual drone goal
        # city.nodes["A21105471"]
        cods = []
        for ii in range( len( log_pos ) ):
            for jj in range( len( log_pos[ii] ) ):
                if log_pos[ii][jj] != None:
                    goal = city.nodes[log_pos[ii][jj][-1]]
                    prev_goal = city.nodes[
                        [
                            plan[id[ii]][0][kk - 1]
                            for kk in range( len( plan[id[ii]][0] ) )
                            if ( plan[id[ii]][0][kk] == log_pos[ii][jj][-1] )
                        ][0]
                    ]
                    traj = LineString( 
                        [
                            Point( np.asarray( ( goal["x"], goal["y"], goal["z"] ) ).astype( float ) ),
                            Point( 
                                np.asarray( 
                                    ( prev_goal["x"], prev_goal["y"], prev_goal["z"] )
                                ).astype( float )
                            ),
                        ]
                    )
                    # check if deviation larger than threshold
                    if ( 
                        traj.distance( 
                            Point( log_pos[ii][jj][2], log_pos[ii][jj][1], log_pos[ii][jj][3] )
                        )
                        > self.rules["wind_rules"]["path_dev_th"]
                    ):
                        # dev_loc=(log_pos[ii][jj][2],log_pos[ii][jj][1],log_pos[ii][jj][3])
                        cod = self.cells.sindex.query( 
                            Point( log_pos[ii][jj][2], log_pos[ii][jj][1] ), predicate="within"
                        )
                        if len( cod ) > 0:
                            cod = [
                                cod[ind]
                                for ind in range( len( cod ) )
                                if ( 
                                    ( log_pos[ii][jj][3] > self.cells.iloc[cod[ind]]["z_min"] )
                                    and ( log_pos[ii][jj][3] < self.cells.iloc[cod[ind]]["z_max"] )
                                )
                            ][0]
                            cods.append( cod )

        cods = list( set( cods ) )  # get unique
        for ii in cods:
            self.decrease_speed( self.cells.index[ii] )
        print( "The tactical wind field dependency update DONE." )
        return

    def update_conflict( self, conflict_pairs, conflict_loc, conflict_head, dur_sec ):
        print( "Starting the concentration of conflict update." )
        self.cells["conflicts"] = 0
        self.cells["aoc"] = 0
        anglAvg = [misc.AnglRunAvg() for ii in range( len( self.cells ) )]
        for ii in range( len( conflict_pairs ) ):
            # coc ... cell of conflict
            coc = self.cells.sindex.query( 
                Point( conflict_loc[ii][1], conflict_loc[ii][0] ), predicate="within"
            )
            # [what[index] for index in range(len(what)) if () ]
            if len( coc ) > 0:
                coc = [
                    coc[ind]
                    for ind in range( len( coc ) )
                    if ( 
                        ( conflict_loc[ii][2] > self.cells.iloc[coc[ind]]["z_min"] )
                        and ( conflict_loc[ii][2] < self.cells.iloc[coc[ind]]["z_max"] )
                    )
                ][0]

            if coc >= 0:
                self.cells.at[self.cells.index[coc], "aoc"] = anglAvg[coc].update_angl_deg( 
                    conflict_head[ii][0]
                )
                self.cells.at[self.cells.index[coc], "conflicts"] += 1

        for id in self.cells.index:
            c_km3h = self.cells.at[id, "conflicts"] / ( 
                self.cells.loc[id].geometry.area
                * ( 111139 ** 2 * 1e-9 )
                * ( self.cells.at[id, "z_max"] - self.cells.at[id, "z_min"] )
                * ( dur_sec / 3600 )
            )

            if c_km3h > self.rules["concentration_rules"]["conflict_th_km_3_h"]:
                self.decrease_speed( id )
                # allow only the dominant ownship direction
                angl = self.cells.at[id, "aoc"]
                if angl >= -45 and angl <= 45:
                    self.cells.at[id, "geovect"] = "N"
                elif angl > 45 and angl <= 135:
                    self.cells.at[id, "geovect"] = "E"
                elif angl > 135 and angl < -135:
                    self.cells.at[id, "geovect"] = "S"
                elif angl >= -135 and angl < -45:
                    self.cells.at[id, "geovect"] = "W"

            elif self.cells.at[id, "geovect"] != "NSEW":
                # sufficiently low conflict frequency -> open sector directions
                self.cells.at[id, "geovect"] = "NSEW"
        anglAvg = {}
        print( "The concentration of conflict update finished." )
        return

    def update_traffic_strat( self, flight_log_pos ):
        print( "Starting the strategic traffic dependency update." )
        self.eval_occupancy( self.cells, flight_log_pos )

        self.traffic_split( flight_log_pos )
        self.traffic_merge()
        print( "The strategic traffic dependency update DONE." )
        return

    def eval_occupancy( self, cells, flight_log_pos ):
        cells["occupancy"] = 0
        # eval cell occupancy
        for i_uav in range( len( flight_log_pos ) ):
            for t in range( len( flight_log_pos[i_uav] ) ):
                if flight_log_pos[i_uav][t] != None:
                    cid = cells.sindex.query( 
                        Point( flight_log_pos[i_uav][t][2], flight_log_pos[i_uav][t][1] ),
                        predicate="within",
                    )

                    if len( cid ) > 0:
                        cid = [
                            cid[ind]
                            for ind in range( len( cid ) )
                            if ( 
                                ( flight_log_pos[i_uav][t][3] > cells.iloc[cid[ind]]["z_min"] )
                                and ( flight_log_pos[i_uav][t][3] < cells.iloc[cid[ind]]["z_max"] )
                            )
                        ][0]

                    if cid >= 0:
                        cells.at[cells.index[cid], "occupancy"] += 1
        return cells

    def traffic_split( self, flight_log_pos ):
        dur_sec = len( flight_log_pos[0] ) * np.mean( 
            np.diff( 
                [
                    flight_log_pos[0][ii][0]
                    for ii in range( len( flight_log_pos[0] ) )
                    if flight_log_pos[0][ii] != None
                ]
            )
        )
        # split cells with high occupancy to 8 sub-cells to test split rule
        for id in self.cells.index:
            if self.cells.at[id, "occupancy"] > 0:
                self.cells.at[id, "occupancy"] /= dur_sec
                if ( self.cells.at[id, "capacity"] > 0 ) and ( 
                    self.cells.at[id, "occupancy"] > ( self.cells.at[id, "capacity"] / 8 )
                ):
                    # split to 8 sub cells
                    split = gpd.GeoDataFrame( self.cells.loc[id:id] ).reset_index( drop=True )
                    split = misc.split_cell( split, 0, "x", False )
                    for id_split in split.index:
                        split = misc.split_cell( split, id_split, "y", False )
                    for id_split in split.index:
                        split = misc.split_alt( split, id_split, "z", False )
                    split = split.reset_index( drop=True )
                    # evaluate occupancy of each sub cell
                    split = self.eval_occupancy( split, flight_log_pos )
                    occupancy = split["occupancy"].to_numpy() / dur_sec
                    # check if any sub cell is over occupied
                    if any( occupancy > ( self.cells.at[id, "capacity"] / 8 ) ):
                        # the split assures that two most occupied sub cells remain in the same cell after split
                        # sum of two most occupied cell indices suffice to decide split mode
                        split_mode = np.sum( np.argsort( occupancy )[-2:] )
                        # 7 ... two most occupied segments in oposite corners -> no split required
                        if split_mode != 7:
                            if any( split_mode == [2, 3, 11, 12] ):
                                misc.split_cell( self.cells, id, "x", False )
                            if any( split_mode == [1, 5, 9, 13] ):
                                misc.split_cell( self.cells, id, "y", False )
                            if any( split_mode == [4, 6, 8, 10] ):
                                misc.split_alt( self.cells, id, "z", False )
                            # update occupancy after split
                            self.cells.iloc[-2:] = self.eval_occupancy( 
                                self.cells.iloc[-2:], flight_log_pos
                            )

        return

    def traffic_merge( self ):
        segments = self.cells[["parent", "occupancy", "capacity"]]
        segments["dissolve"] = -segments.index
        for iid in segments.index:
            if segments.at[iid, "parent"] != None:
                segments.at[iid, "dissolve"] = segments.at[iid, "parent"][-1]
        segments = segments.groupby( by=["dissolve"], sort=False, dropna=False )
        segments = segments.aggregate( func={"capacity": "sum", "occupancy": "sum"} )
        id_parent = [
            iid
            for iid in segments.index
            if ( 
                ( iid >= 0 ) and ( segments.at[iid, "occupancy"] < 0.2 * segments.at[iid, "capacity"] )
            )
        ]

        self.cells = misc.merge_cells( self.cells, id_parent )

        return

    def update_traffic_tact( self, flight_log_pos ):
        print( "Starting the tactical traffic dependency update." )
        self.cells["occupancy"] = 0
        for ii in range( len( flight_log_pos ) ):
            # cid ... airplane location cell id
            # expecting the actual plane position as the last entry of flight_log_pos
            if flight_log_pos[ii][-1] != None:
                cid = self.cells.sindex.query( 
                    Point( flight_log_pos[ii][-1][2], flight_log_pos[ii][-1][1] ), predicate="within"
                )

                if len( cid ) > 0:
                    cid = [
                        cid[ind]
                        for ind in range( len( cid ) )
                        if ( 
                            ( flight_log_pos[ii][-1][3] > self.cells.iloc[cid[ind]]["z_min"] )
                            and ( flight_log_pos[ii][-1][3] < self.cells.iloc[cid[ind]]["z_max"] )
                        )
                    ][0]

                if cid >= 0:
                    self.cells.at[self.cells.index[cid], "occupancy"] += 1

        for id in self.cells.index:
            if self.cells.at[id, "occupancy"] > self.cells.at[id, "capacity"]:
                self.decrease_speed( id )

        print( "The tactical traffic dependency update finished." )
        return

    def update_event( self, event, now ):
        # event e.g.,
        #
        # { "type": "Feature", "properties": { "begin": 0, "end": 28 }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 9.7355, 52.366 ], [ 9.748, 52.366 ], [ 9.748, 52.374 ], [ 9.7355, 52.374 ], [ 9.7355, 52.366 ] ] ] } }
        #
        # - parallel thread "p" finds all airspace segments {within, overlapps, contains} the event polygon
        # - thread p sleeps until "begin"
        # - corresponding airspace segments -> ‘red’ class (including the changing the capacity and speed properties)
        # - thread p sleeps until "end"
        # - effected airspace cells are restored to initial class including its properties "speed_max", "speed_min", "capacity"
        # - p is closed

        if type( event ) == str:
            event = gpd.read_file( 
                ( self.rules["event_rules"]["event_data_folder"] + event + ".geojson" ),
                driver="GeoJSON",
            )
        if now < event.at[0, "end"]:
            # self.event_instance(event, now)

            # p = Process(target=self.event_instance, args=(event, now))
            # p.start()
            p = Thread( target=self.event_instance, args=( event, now ) )
            p.start()

        return

    def event_instance( self, event, now ):
        close = ( self.cells.sindex.query( event.at[0, "geometry"], predicate="overlaps" ), )
        close = np.append( 
            close, self.cells.sindex.query( event.at[0, "geometry"], predicate="contains" )
        )
        close = np.append( 
            close, self.cells.sindex.query( event.at[0, "geometry"], predicate="within" )
        )

        if event.at[0, "begin"] == now:
            # time.sleep( event.at[0, "begin"] - now )
            print( "Event started." )
            for idx in close:
                self.close_cell( idx )
            self.event_update = True

        # time.sleep( event.at[0, "end"] - now )
        if event.at[0, "end"] == now:
            print( "Event ended." )
            for idx in close:
                self.restore_cell( idx )
            self.event_update = True
        return

    def plot_cells( self ):
        plot = self.cells.hvplot( 
            c="class",
            geo=True,
            frame_height=1000,
            tiles="CartoDark",
            hover_cols=["z_min", "z_max", "capacity", "turbulence"],
            alpha=0.2,
        )
        hvplot.show( plot )
        return

    def export_cells( self ):

        latlon = self.cells.bounds.copy()
        latlon.rename( 
            columns={"minx": "lon_min", "miny": "lat_min", "maxx": "lon_max", "maxy": "lat_max"},
            inplace=True,
        )
        export = pd.concat( 
            [
                self.cells[["class"]].copy(),
                latlon[["lat_min", "lat_max", "lon_min", "lon_max"]].copy(),
                self.cells[
                    [
                        "z_min",
                        "z_max",
                        "speed_min",
                        "speed_max",
                        "capacity",
                        "occupancy",
                        "geovect",
                        "parent",
                        "new",
                        "updated",
                    ]
                ].copy(),
            ],
            axis=1,
            join="inner",
        )
        export.to_json( 
            ( "usepe/segmentation_service/data/examples/" + self.region_name + ".json" ), orient="records", lines=True
        )
        return


if __name__ == "__main__":
    # rules are loaded from ".config/rules.json"

    # wind file is assumed in rules.json file at rules["wind_rules"]["wind_data_folder"] + windFile + ".nc" (suffix added automatically)
    # therefore wind data are ".data/wind/test_hannover_1m/test_hannover_1m_3d.nc" in this example

    # event is either GeoDataFrame or name of the .geojson file located in rules.json file at rules["event_rules"]["event_data_folder"] + event + ".geojson" (suffix added automatically)
    # therefore the event file in this example is ".data/event/event.geojson"

    region = "Hannover"
    windFile = "test_hannover_1m/test_hannover_1m_3d"
    event = "event"

    with open( "usepe/segmentation_service/data/flight_log/drones_routes.dict", "rb" ) as f:
        flight_plan = pickle.load( f )

    with open( "usepe/segmentation_service/data/flight_log/drones_actual_paths.list", "rb" ) as f:
        flight_log_pos = pickle.load( f )

    with open( "usepe/segmentation_service/data/flight_log/drone_active_waypoint.list", "rb" ) as f:
        flight_log_w = pickle.load( f )

    with open( "usepe/segmentation_service/data/flight_log/drone_id.list", "rb" ) as f:
        drone_id = pickle.load( f )

    city_graph = nx.read_graphml( "usepe/segmentation_service/data/flight_log/exercise_1_2_layers_reduced.graphml" )

    with open( "usepe/segmentation_service/data/conflict/conflict_pairs.list", "rb" ) as f:
        conflict_pairs = pickle.load( f )

    with open( "usepe/segmentation_service/data/conflict/conflict_pairs_cpa_location.list", "rb" ) as f:
        conflict_loc = pickle.load( f )

    with open( "usepe/segmentation_service/data/conflict/conflict_pairs_headings.list", "rb" ) as f:
        conflict_head = pickle.load( f )

    conflict_log_dur_sec = len( flight_log_pos[0] ) * np.mean( 
        np.diff( 
            [
                flight_log_pos[0][ii][0]
                for ii in range( len( flight_log_pos[0] ) )
                if flight_log_pos[0][ii] != None
            ]
        )
    )
    # initialize airspace cells of the region specified in "region"
    segments = segmentationService( region )

    segments.update_wind_strat( windFile, False )  # strategic update rules based on wind data
    segments.update_wind_tact( flight_plan, flight_log_pos, flight_log_w, drone_id, city_graph )
    segments.update_conflict( conflict_pairs, conflict_loc, conflict_head, conflict_log_dur_sec )
    segments.update_traffic_strat( flight_log_pos )
    segments.update_traffic_tact( flight_log_pos )
    segments.update_event( event, 1.0 )
    segments.export_cells()  # export .json file to "./data/examples"
    # exported properties: "z_min","z_max","speed_min","speed_max","capacity","occupancy","geovect","new","updated"
    segments.plot_cells()
