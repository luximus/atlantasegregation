import math

from shapely import Point


def approx_square_distance_on_earth(point1: Point, point2: Point) -> float:
    """
    Quickly approximate the square of the distance between two shapely Points on Earth's surface in miles  by assuming
    an equirectangular model of the earth (where latitude and longitude lines form a flat square grid).

    Method taken from https://www.movable-type.co.uk/scripts/latlong.html.
    """
    x1, y1 = [a[0] for a in point1.xy]
    x2, y2 = [a[0] for a in point2.xy]
    dx = x2 - x1
    dy = y2 - y1
    my = (y1 + y2)/2
    x, y = dx * math.cos(math.radians(my)), dy
    earth_radius = 3958.8  # In miles
    return earth_radius*earth_radius * x*x + y*y
