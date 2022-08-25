from collections import defaultdict

import geopandas as gpd
import osmnx as ox
import pandas as pd


def parse_rules(rules):
    osm_tags = defaultdict(list)
    for _, data in rules["ground"].items():
        for key, value in data["osm"].items():
            osm_tags[key].extend(value)
    return dict(osm_tags)


def filter_osm(osm_data, rules):
    ground_data = []
    for name, data in rules["ground"].items():
        tags = data["osm"]
        filter = osm_data.isin(tags).any(axis=1)
        if not filter.any():
            continue
        filtered = osm_data.loc[filter, ["geometry", "name"]]
        filtered["type"] = name
        filtered["class"] = data["class"]
        filtered["buffer"] = data["buffer"] if "buffer" in data and data["buffer"] > 1 else 1
        ground_data.append(filtered)
    ground_data = gpd.GeoDataFrame(pd.concat(ground_data), geometry="geometry")
    ground_data["name"] = ground_data["name"].astype(str)
    ground_data["type"] = ground_data["type"].astype(str)
    return ground_data


def buffer_regions(ground_data):
    buffered = ground_data.to_crs("EPSG:3035").buffer(ground_data["buffer"])
    ground_data["geometry"] = buffered.to_crs("EPSG:4326")
    ground_data["buffer"] = ground_data["buffer"].astype(int) * -1

    return ground_data


def add_restrictions(ground_data, rules):
    ground_data["z_min"] = False
    ground_data["z_max"] = False
    ground_data["speed_min"] = False
    ground_data["speed_max"] = False
    for name, data in rules["classes"].items():
        ground_data.loc[ground_data["class"] == name, "z_min"] = min(data["altitude"])
        ground_data.loc[ground_data["class"] == name, "z_max"] = max(data["altitude"])
        ground_data.loc[ground_data["class"] == name, "speed_min"] = min(data["velocity"])
        ground_data.loc[ground_data["class"] == name, "speed_max"] = max(data["velocity"])
    return ground_data


def get(region, rules):
    osm_tags = parse_rules(rules)
    osm_data = ox.geometries_from_polygon(region, osm_tags)
    gnd_data = filter_osm(osm_data, rules)
    gnd_data = buffer_regions(gnd_data)
    gnd_data = gnd_data.rename_axis(index=["element", "id"])
    gnd_data = add_restrictions(gnd_data, rules)

    return gnd_data[
        ["class", "type", "name", "geometry", "buffer", "z_min", "z_max", "speed_min", "speed_max"]
    ]
