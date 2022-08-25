import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import triangulate, unary_union


def orthogonal_bounds(region, min_grid=1e-6):
    rects = []
    region = Polygon(region)
    objects = [tri for tri in triangulate(region) if tri.within(region)]
    for poly in objects:
        xmin = np.round(poly.bounds[0] / min_grid) * min_grid
        ymin = np.round(poly.bounds[1] / min_grid) * min_grid
        xmax = np.round(poly.bounds[2] / min_grid) * min_grid
        ymax = np.round(poly.bounds[3] / min_grid) * min_grid
        rects.append(box(xmin, ymin, xmax, ymax))

    return unary_union(rects), objects


def generate_random(n, irregularity, spikeyness, X=0, Y=0, r=10):
    """Create Polygon by sampling points on a circle around the centre.
    Random noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    irregularity - [0,1] variance in angular spacing of vertices. [0,1] will map to [0, 2pi/n]
    spikeyness - [0,1] variance in radius. [0,1] will map to [0, r]

    Returns a list of vertices, in CCW order.

    Source: https://stackoverflow.com/a/25276331
    """
    import math
    import random

    def clip(x, min, max):
        if min > max:
            return x
        elif x < min:
            return min
        elif x > max:
            return max
        else:
            return x

    irregularity = clip(irregularity, 0, 1) * 2 * math.pi / n
    spikeyness = clip(spikeyness, 0, 1) * r

    # generate n angle steps
    angleSteps = []
    lower = (2 * math.pi / n) - irregularity
    upper = (2 * math.pi / n) + irregularity
    sum = 0
    for i in range(n):
        tmp = random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2 * math.pi)
    for i in range(n):
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(n):
        r_i = clip(random.gauss(r, spikeyness), 0, 2 * r)
        x = X + r_i * math.cos(angle)
        y = Y + r_i * math.sin(angle)
        points.append((int(x), int(y)))

        angle = angle + angleSteps[i]

    return Polygon(points)
