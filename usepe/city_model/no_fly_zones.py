#!/usr/bin/python

"""Define no-fly zones and apply them to the segments."""


__author__ = 'mbaena'
__copyright__ = '(c) Nommon 2021'


from usepe.city_model.dynamic_segments import dynamicSegments

from shapely.geometry import Polygon


def intersection( restricted_area, segment, segments ):
    """
    Return True if the given segment intersects with the restricted area.

    Args:
        restricted_area (list): shape of the restricted area,
                                as a Polygon with each vertex defined as a tuple (lon, lat)
        segment (string): id of segment
        segments (dictionary): segment information

    Returns:
        boolean: True if restricted area intersects with the segment area
    """
    restricted_area_shape = Polygon( restricted_area )
    segment_box = [( segments[segment]['lon_min'], segments[segment]['lat_min'] ),
                   ( segments[segment]['lon_max'], segments[segment]['lat_min'] ),
                   ( segments[segment]['lon_max'], segments[segment]['lat_max'] ),
                   ( segments[segment]['lon_min'], segments[segment]['lat_max'] ),
                   ( segments[segment]['lon_min'], segments[segment]['lat_min'] )]

    segmet_shape = Polygon( segment_box )
    return restricted_area_shape.intersects( segmet_shape )


def restrictedSegments( G, segments, restricted_area, config ):
    """
    Impose zero speed limitations on the certain segments, creating a restricted area for flying.

    Args:
        G (graph): graph of the city
        segments (dictionary): segment information
        restricted_area (list): shape of the restricted area,
                                as a Polygon with each vertex defined as a tuple (lon, lat)
        config (ConfigParser): configuration file with all the relevant information

    Returns:
        G (graph): updated graph with restricted zones
        segments (dictionary): updated segments with restricted zones
    """
    for segment in segments:
        if intersection( restricted_area, segment, segments ):
            segments[segment]['speed'] = 0
            segments[segment]['updated'] = True

    G, segments = dynamicSegments( G, config, segments )
    return G, segments


if __name__ == '__main__':
    pass
