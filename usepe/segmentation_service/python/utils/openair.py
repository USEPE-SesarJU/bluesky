from math import asin, cos, degrees, pi, radians, sin, sqrt

import numpy as np
import shapely.affinity
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon

_RADIUS = 6372.8 * 1e3  # [m]  Reference radius of the earth


def _haversine(p1, p2, radius=_RADIUS):
    if type(p2) == str and p2 == "lon":
        p2 = [p1[0] - 1, p1[1]]
    if type(p2) == str and p2 == "lat":
        p2 = [p1[0], p1[1] - 1]

    lon = [p1[0], p2[0]]
    lat = [p1[1], p2[1]]

    dlon = radians(lon[1] - lon[0])
    dlat = radians(lat[1] - lat[0])
    a = sin(dlat / 2) ** 2 + cos(radians(lat[0])) * cos(radians(lat[1])) * sin(dlon / 2) ** 2
    d = 2 * asin(sqrt(a)) * radius
    return d


def _inverse(d, lat, radius=_RADIUS):
    dlat = d / radius
    dlon = asin(sin(dlat)) / cos(radians(lat))
    return degrees(dlon), degrees(dlat)


def to_gdf(filename):
    def parse_class(val):
        ac = {
            "r": "Restricted",
            "q": "Danger",
            "p": "Prohibited",
            "a": "Class A",
            "b": "Class B",
            "c": "Class C",
            "d": "Class D",
            "gp": "Glider prohibited",
            "ctr": "CTR",
            "w": "Wave Window",
            "tmz": "TMZ",
            "rmz": "RMZ",
            "g": "Gliding",
        }
        if val.lower() in ac:
            return ac[val.lower()]
        else:
            return val

    def parse_name(val):
        return val

    def parse_altitude(val):
        val = val.strip()
        if val.lower() == "gnd":
            return 0
        if val.lower().endswith("agl") or val.lower().endswith("gnd"):
            val = val[:-3].strip()
        if val.startswith("FL"):
            val = f"{int(val[2:]) * 1000}ft msl"
        if val.lower().endswith("msl"):
            # TODO MSL to AGL, this however doesn't really matter if we are only looking at VLL anyway
            val = val[:-3].strip()
        if val.endswith("ft"):
            val = val[:-2].strip()
        if val.lower().endswith("f"):
            val = val[:-1].strip()
        if val.lower().endswith("m"):
            val = int(val[:-1]) * 3.281
        return int(val)

    def parse_variable(val):
        x, n = val.split("=", 1)
        types = {"d": "direction", "x": "coordinate", "w": "width"}
        return (types[x.lower()], n)

    def parse_coordinate(val):
        def dms2decimal(val):
            deg, minutes, seconds, dir = val.split(":")
            deg = float(deg) + float(minutes) / 60 + float(seconds) / (60 * 60)
            return deg * (-1 if dir in ["W", "S"] else 1)

        val = val.split()
        lat = dms2decimal(":".join((val[0], val[1])))
        lon = dms2decimal(":".join((val[2], val[3])))
        return (lon, lat)

    def parse_point(val):
        center = parse_coordinate(val)
        return Point(*center)

    def parse_radius(val, center):
        radius = float(val) * 1852  # nm to m for radius
        radius = _inverse(radius, center[1])
        return radius

    def draw_arc(center, radius, a0, a1, segments=50, direction="+"):
        # TODO fix the circle projection issues
        if direction == "-":
            a1 = -1 * a1
        theta = np.radians(np.linspace(a0, a1, segments))
        x = center[0] + radius[0] * np.cos(theta)
        y = center[1] + radius[1] * np.sin(theta)
        p = [Point(x0, y0) for x0, y0 in zip(x, y)]

        return p

    def parse_arc_radius(val):
        center = parse_coordinate(variables["coordinate"])
        radius, a0, a1 = val.split(",")
        radius = parse_radius(radius, center)
        return draw_arc(center, radius, a0, a1, direction=variables["direction"])

    def parse_arc_points(val):
        center = parse_coordinate(variables["coordinate"])
        p1, p2 = val.split(",")
        p1 = parse_coordinate(p1)
        p2 = parse_coordinate(p2)
        radius = _haversine(center, p1)
        radius = _inverse(radius, center[0])
        a0 = np.degrees(np.arctan2(p1[1] - center[1], p1[0] - center[0]))
        a1 = np.degrees(np.arctan2(p2[1] - center[1], p2[0] - center[0]))
        return draw_arc(center, radius, a0, a1, direction=variables["direction"])

    def parse_circle(val):
        center = parse_coordinate(variables["coordinate"])
        radius = parse_radius(val, center)
        return draw_arc(center, radius, 0, 360, 100)

    tokens = {
        "ac": {"key": "class", "value": parse_class},
        "an": {"key": "name", "value": parse_name},
        "ah": {"key": "z_max", "value": parse_altitude},
        "al": {"key": "z_min", "value": parse_altitude},
        "v": {"key": "variable", "value": parse_variable},
        "dp": {"key": "point", "value": parse_point},
        "da": {"key": "arc-radius", "value": parse_arc_radius},
        "db": {"key": "arc-points", "value": parse_arc_points},
        "dc": {"key": "circle", "value": parse_circle},
    }
    skip = "*"

    with open(filename, "r") as f:
        current_airspace = {}
        all_airspaces = []
        variables = {"direction": "+"}
        for line in f:
            line = line.strip()
            if line.startswith(skip):
                continue

            if not line:
                if current_airspace:
                    all_airspaces.append(current_airspace)
                    current_airspace = {}
                    variables = {"direction": "+"}
                continue

            tok, val = line.split(" ", 1)
            tok = tok.lower()
            if tok not in tokens:
                continue

            cmd = tokens[tok]["key"]
            val = tokens[tok]["value"](val)

            if tok.startswith("a"):  # add attribute
                current_airspace[cmd] = val
            if tok.startswith("v"):  # set variable
                variables[val[0]] = val[1]
            if tok.startswith("d"):  # add data (i.e. geometry)
                if not "geometry" in current_airspace:
                    current_airspace["geometry"] = []
                if isinstance(val, list):
                    current_airspace["geometry"].extend(val)
                else:
                    current_airspace["geometry"].append(val)

    for airspace in all_airspaces:
        airspace["geometry"] = Polygon([[p.x, p.y] for p in airspace["geometry"]])

    gdf = GeoDataFrame.from_dict(all_airspaces, "geometry")
    return gdf
