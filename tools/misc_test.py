#!/usr/bin/python
# -*- encoding: utf8 -*-

import unittest

from tools import misc


__author__ = "sheep"


class AvgAngleTest(unittest.TestCase):

    def testLine(self):
        lons, lats = [0, 1, 3], [0, 1, 3]
        expected = 45.0
        actual = misc.avg_angle(lons, lats, dist_func=misc.get_euclidean_distance)
        self.assertAlmostEquals(expected, actual)

        lons, lats = [0, 0, 0], [0, 1, 3]
        expected = 0.0
        actual = misc.avg_angle(lons, lats, dist_func=misc.get_euclidean_distance)
        self.assertAlmostEquals(expected, actual)

    def testTurn(self):
        lons, lats = [0, 1, 0], [0, 1, 2]
        expected = 0.0
        actual = misc.avg_angle(lons, lats, dist_func=misc.get_euclidean_distance)
        self.assertAlmostEquals(expected, actual)

    def testTurn2(self):
        lons, lats = [0, 1, -1], [0, 1, 3]
        expected = -15.0
        actual = misc.avg_angle(lons, lats, dist_func=misc.get_euclidean_distance)
        self.assertAlmostEquals(expected, actual)

    def testTurn3(self):
        lons, lats = [0, 1, 3], [0, 1, 1]
        expected = 71.35900358223131
        actual = misc.avg_angle(lons, lats)
        self.assertEquals(expected, actual)


if __name__ == '__main__':
    unittest.main()

