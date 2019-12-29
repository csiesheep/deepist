#!/usr/bin/python
# -*- encoding: utf8 -*-

import optparse
import sys

from tools import misc


__author__ = 'sheep'


def main(traj_fname, path_fname, output_fname):
    '''\
    Filter incorrect map-matched paths
    by comparing the length of a path andd its orignal trajectory

    %prog [options] <traj_fname> <path_fname> <output_fname>
    '''
    traj_distance = []
    traj_times = []
    with open(traj_fname) as f:
        for line in f:
            _, txys, t, ts = line.strip().split('\t')
            traj = eval(txys)
            d = get_distance(traj)
            traj_distance.append(d)
            traj_times.append((t, ts))

    ith = 0
    count = 0
    with open(path_fname) as f:
        with open(output_fname, 'w') as fo:
            for line in f:
                _, txys, t, ts = line.strip().split('\t')
                while (t, ts) != traj_times[ith]:
                    ith += 1
                    continue

                traj = eval(txys)
                d = get_distance(traj)
                diff_ratio = abs(d-traj_distance[ith])/(traj_distance[ith]+0.0001)
                if diff_ratio > 0.1:
                    count += 1
                else:
                    fo.write(line)

                ith += 1
                if ith % 1000 == 0:
                    print ith, 'incorrect ratio:', float(count)/ith
                if ith == len(traj_distance):
                    break
    print 'incorrect ratio:', float(count)/ith

    return 0

def get_distance(traj):
    d = 0.0
    for i in range(1, len(traj)):
        d += misc.get_distance(traj[i-1][1], traj[i-1][2],
                               traj[i][1], traj[i][2])
    return d


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))

