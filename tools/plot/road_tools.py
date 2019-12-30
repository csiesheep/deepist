#!/usr/bin/python
# -*- encoding: utf8 -*-

from imposm.parser import OSMParser
import optparse
import sys
from tools import misc


__author__ = 'sheep'


class Intersections():

    def __init__(self, roads):
        self.loc2degree = {}
        for lon1, lat1, lon2, lat2, type_ in roads:
            s = (lon1, lat1)
            d = (lon2, lat2)
            if s not in self.loc2degree:
                self.loc2degree[s] = 0
            self.loc2degree[s] += 1
            if d not in self.loc2degree:
                self.loc2degree[d] = 0
            self.loc2degree[d] += 1

        print len(self.loc2degree)
        to_remove = []
        for n, count in self.loc2degree.items():
            if count <= 2:
                to_remove.append(n)
        for n in to_remove:
            self.loc2degree.pop(n)
        print len(self.loc2degree)

    @staticmethod
    def load_from_file(fname):
        ways = Ways()
        p = OSMParser(concurrency=4, ways_callback=ways.get_ways)
        p.parse(fname)

        coords = Coords()
        p = OSMParser(concurrency=4,
                      coords_callback=coords.get_coords)
        p.parse(fname)
        id2coords = coords.coords

        roads = []
        for tags, refs, weight in ways.ways.values():
            for i in range(1, len(refs)):
                road = (id2coords[refs[i-1]][0],
                        id2coords[refs[i-1]][1],
                        id2coords[refs[i]][0],
                        id2coords[refs[i]][1],
                        weight)
                roads.append(road)
        print len(roads)
        return Intersections(roads)


class RoadNetwork2():

    def __init__(self, roads):
        self.graph = {}
        for lon1, lat1, lon2, lat2, type_ in roads:
            s = (lon1, lat1)
            d = (lon2, lat2)
            if s not in self.graph:
                self.graph[s] = []
            self.graph[s].append((d, type_))
            if d not in self.graph:
                self.graph[d] = []
            self.graph[d].append((s, type_))

    @staticmethod
    def load_from_file(fname):
        ways = Ways()
        p = OSMParser(concurrency=4, ways_callback=ways.get_ways)
        p.parse(fname)

        coords = Coords()
        p = OSMParser(concurrency=4,
                      coords_callback=coords.get_coords)
        p.parse(fname)
        id2coords = coords.coords

        roads = []
        for tags, refs, weight in ways.ways.values():
            for i in range(1, len(refs)):
                road = (id2coords[refs[i-1]][0],
                        id2coords[refs[i-1]][1],
                        id2coords[refs[i]][0],
                        id2coords[refs[i]][1],
                        weight)
                roads.append(road)
        print len(roads)
        return RoadNetwork2(roads)


class RoadNetwork():

    def __init__(self, roads):
        self.id2roads = {}
        self.lons = []
        self.lats = []
        for ith, road in enumerate(roads):
            self.id2roads[ith] = road
            min_lon, min_lat, max_lon, max_lat, type_ = road
            self.lons.append((min_lon, ith))
            self.lons.append((max_lon, ith))
            self.lats.append((min_lat, ith))
            self.lats.append((max_lat, ith))
        self.lons = sorted(self.lons)
        self.lats = sorted(self.lats)

    def query(self, min_lon, min_lat, max_lon, max_lat):
        min_lon_ith = None
        for ith, (lon, id_) in enumerate(self.lons):
            if lon >= min_lon:
                min_lon_ith = ith
                break
        if min_lon_ith is None:
            return []

        min_lat_ith = None
        for ith, (lat, id_) in enumerate(self.lats):
            if lat >= min_lat:
                min_lat_ith = ith
                break
        if min_lat_ith is None:
            return []

        max_lon_ith = None
        for ith in range(len(self.lons)-1, 0, -1):
            if self.lons[ith][0] <= max_lon:
                max_lon_ith = ith
                break
        if max_lon_ith is None:
            return []

        max_lat_ith = None
        for ith in range(len(self.lats)-1, 0, -1):
            if self.lats[ith][0] <= max_lat:
                max_lat_ith = ith
                break
        if max_lat_ith is None:
            return []

        roads = []
        ids = set([id_ for _, id_
                   in self.lons[min_lon_ith:max_lon_ith+1]])
        ids2 = set([id_ for _, id_
                   in self.lats[min_lat_ith:max_lat_ith+1]])
        for id_ in ids:
            if id_ not in ids2:
                continue
            roads.append(self.id2roads[id_])
        return roads

    @staticmethod
    def load_from_file(fname):
        ways = Ways()
        p = OSMParser(concurrency=4, ways_callback=ways.get_ways)
        p.parse(fname)

        coords = Coords()
        p = OSMParser(concurrency=4,
                      coords_callback=coords.get_coords)
        p.parse(fname)
        id2coords = coords.coords

        roads = []
        for tags, refs, weight in ways.ways.values():
            for i in range(1, len(refs)):
                road = (id2coords[refs[i-1]][0],
                        id2coords[refs[i-1]][1],
                        id2coords[refs[i]][0],
                        id2coords[refs[i]][1],
                        weight)
                roads.append(road)
        print len(roads)
        return RoadNetwork(roads)


class Ways(object):

    def __init__(self):
        self.ways = {}

        self.highway_values = set([
            "motorway", "trunk", "primary", "secondary",
            "tertiary", "motorway_link", "trunk_link",
            "primary_link", "secondary_link", "tertiary_link",
            "road", "residential", "living_street",
        ])
        self.railway_values = set(["light_rail"])

        self.weights = {
            "motorway": 3,
            "trunk": 2,
            "primary": 2,
            "secondary": 2,
            "motorway_link": 3,
            "trunk_link": 2,
            "primary_link": 2,
            "secondary_link": 2,
        }

    def get_ways(self, ways):
        for osmid, tags, refs in ways:

            to_draw = False
            weight = 1
            for key, value in tags.items():
                if key not in set(["highway", "railway"]):
                    continue
                if value in self.highway_values:
                    to_draw = True
                if value in self.railway_values:
                    to_draw = True
                if weight < self.weights.get(value, 1):
                    weight = self.weights.get(value, 1)
            if not to_draw:
                continue

            self.ways[osmid] = (tags, refs, weight)


class Coords(object):

    def __init__(self):
        self.coords = {}

    def get_coords(self, coords):
        for osmid, lon, lat in coords:
            lon = float("%.7f" % lon)
            lat = float("%.7f" % lat)
            self.coords[osmid] = (lon, lat)


class Signals():

    def __init__(self, signals):
        self.id2signal = {}
        self.lons = []
        self.lats = []
        for ith, (lon, lat, type_) in enumerate(signals):
            self.id2signal[ith] = (lon, lat, type_)
            self.lons.append((lon, ith))
            self.lats.append((lat, ith))
        self.lons = sorted(self.lons)
        self.lats = sorted(self.lats)

    def query(self, min_lon, min_lat, max_lon, max_lat):
        min_lon_ith = None
        for ith, (lon, id_) in enumerate(self.lons):
            if lon >= min_lon:
                min_lon_ith = ith
                break
        if min_lon_ith is None:
            return []

        min_lat_ith = None
        for ith, (lat, id_) in enumerate(self.lats):
            if lat >= min_lat:
                min_lat_ith = ith
                break
        if min_lat_ith is None:
            return []

        max_lon_ith = None
        for ith in range(len(self.lons)-1, 0, -1):
            if self.lons[ith][0] <= max_lon:
                max_lon_ith = ith
                break
        if max_lon_ith is None:
            return []

        max_lat_ith = None
        for ith in range(len(self.lats)-1, 0, -1):
            if self.lats[ith][0] <= max_lat:
                max_lat_ith = ith
                break
        if max_lat_ith is None:
            return []

        signals = []
        ids = set([id_ for _, id_
                   in self.lons[min_lon_ith:max_lon_ith+1]])
        ids2 = set([id_ for _, id_
                   in self.lats[min_lat_ith:max_lat_ith+1]])
        for id_ in ids:
            if id_ not in ids2:
                continue
            signals.append(self.id2signal[id_])
        return signals

    @staticmethod
    def load_from_file(fname):
        signals = []
        with open(fname) as f:
            for line in f:
                _, type_, lon_lat = line.strip().split('\t')
                lon, lat = eval(lon_lat)
                signals.append((lon, lat, type_))
        return Signals(signals)
