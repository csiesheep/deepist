#!/usr/bin/python
# -*- encoding: utf8 -*-

import unittest

from tools import plot


__author__ = "sheep"


class EstimateMissingSpeedTest(unittest.TestCase):

    def testSimple(self):
        speeds = [1.0, None, 3.0, 4.0]
        expected = [1.0, 2.0, 3.0, 4.0]
        self.assertEquals(expected, plot.estimate_missing_speeds(speeds))

    def testMultiNone(self):
        speeds = [1.0, None, None, 4.0]
        expected = [1.0, 2.5, 2.5, 4.0]
        self.assertEquals(expected, plot.estimate_missing_speeds(speeds))

    def testNoNone(self):
        speeds = [1.0, 2.0, 3.0, 4.0]
        expected = [1.0, 2.0, 3.0, 4.0]
        self.assertEquals(expected, plot.estimate_missing_speeds(speeds))

    def testNoHead(self):
        speeds = [None, None, 3.0, 4.0]
        expected = [3.0, 3.0, 3.0, 4.0]
        self.assertEquals(expected, plot.estimate_missing_speeds(speeds))

    def testNoTail(self):
        speeds = [1.0, 2.0, None, None]
        expected = [1.0, 2.0, 2.0, 2.0]
        self.assertEquals(expected, plot.estimate_missing_speeds(speeds))

    def testAllNone(self):
        speeds = [None, None, None, None]
        expected = None
        self.assertEquals(expected, plot.estimate_missing_speeds(speeds))


if __name__ == '__main__':
    unittest.main()

