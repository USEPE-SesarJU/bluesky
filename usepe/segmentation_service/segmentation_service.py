from datetime import datetime
from multiprocessing import Process
import math
import pickle
import time

from numpy import NaN
from shapely.strtree import STRtree
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


class segmentationService:
    def __init__( self, region ):
        self.rules = misc.load_rules()
        self.region_name = region
        self.init_cells()
        print( "Region ", region, " initialized." )

        self.cells = misc.split_aspect_ratio( self.cells, self.rules )
        print( "Initial split assuring maximal aspect ratio ", self.rules["aspect_ratio"], " done" )

        self.cells = misc.split_build( self.cells, self.rules["building_layer"] )
        print( "Altitude split at building top level ", self.rules["building_layer"], " m done." )

        self.event_update = False

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
            * ( self.cells.loc[id, "z_max"] - self.cells.loc[id, "z_min"] )
            * capDensity
            * self.rules["classes"][self.cells.loc[id, "class"]]["capacity_factor"]
        )
        return

    def close_cell( self, id ):
        # name, data in rules["classes"].items():
        n = ( name for name, data in self.rules["classes"].items() )
        # self.cells.loc[idx,"class"]= self.rules[]
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
        print( "Processing wind data is done." )
        nAlt = int( wind_data["nAlt"] )
        nLon = int( wind_data["nLon"] )
        nLat = int( wind_data["nLat"] )

        print( "Assigning wind data to cells:" )
        for ii in tqdm( self.cells.index ):
            lon_min = self.cells.geometry[ii].bounds[0]
            lat_min = self.cells.geometry[ii].bounds[1]
            lon_max = self.cells.geometry[ii].bounds[2]
            lat_max = self.cells.geometry[ii].bounds[3]
            alt_min = self.cells.z_min[ii]
            alt_max = self.cells.z_max[ii]

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

                    self.cells.at[ii, "wind_avg"] = np.round( 100 * wind_mean ) / 100
                    self.cells.at[ii, "wind_max"] = np.round( 100 * np.nanmax( cellWind ) ) / 100
                    self.cells.at[ii, "turbulence"] = ( 
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
                    self.cells.at[ii, "wind_avg"] = np.round( 100 * wind_mean ) / 100
                    self.cells.at[ii, "wind_max"] = np.round( 100 * wind_max ) / 100
                    self.cells.at[ii, "turbulence"] = ( 
                        np.round( 100 * cellWind.std().to_numpy() / wind_mean ) / 100
                    )
                    # speed_argmax = cellWind.where(cellWind == wind_max, drop=True).squeeze()
        for id in range( len( self.cells ) ):
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

    def update_wind_tact( self, plan, log, city ):

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
        if now < event.at[0, "end"] + 1:
            p = Process( target=self.event_instance, args=( event, now ) )
            # self.event_instance(event, now)
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

    # with open("./data/flight_log/drones_routes.dict", "rb") as f:
    #     flight_plan = pickle.load(f)

    # with open("./data/flight_log/drones_actual_paths.list", "rb") as f:
    #     flight_log = pickle.load(f)

    # city_graph = nx.read_graphml("./data/flight_log/exercise_1_2_layers_reduced.graphml")

    # initialize airspace cells of the region specified in "region"
    segments = segmentationService( region )

    segments.update_wind_strat( windFile, False )  # strategic update rules based on wind data
    # segments.update_wind_tact(flight_plan, flight_log, city_graph) # not yet implemented
    segments.update_event( event, 1.0 )  # event update rule
    segments.export_cells()  # export .json file to "./data/examples"
    # exported properties: "z_min","z_max","speed_min","speed_max","capacity","occupancy","geovect","parent","new","updated"
    segments.plot_cells()
