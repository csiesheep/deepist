#!/usr/bin/python
# -*- encoding: utf8 -*-

import math
from math import sin, cos, sqrt, atan2, radians


__author__ = 'sheep'


def get_distance(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = (sin(dlat/2))**2 + cos(radians(lat1)) * cos(radians(lat2)) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

def get_euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def avg_angle(lons, lats, dist_func=get_distance):
    total_d = 0.0
    total_angle = 0.0
    for i in range(1, len(lons)):
        d = dist_func(lons[i-1], lats[i-1], lons[i], lats[i])
        total_d += d
        angle = get_angle(lons[i-1], lats[i-1], lons[i], lats[i])
        total_angle += d*angle
    return total_angle/total_d

def get_angle(lon1, lat1, lon2, lat2):
    dx = lon2-lon1
    dy = lat2-lat1
    return atan2(dx, dy)*180/math.pi
