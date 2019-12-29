#!/usr/bin/python
# -*- encoding: utf8 -*-

import optparse
import os
import sys

from tools import misc


__author__ = 'sheep'


def main(fname, matched_folder, output_fname):
    '''\
    %prog [options] <traj_fname> <matched_folder> <output_fname>
    '''
    no_count = 0
    count = 0
    ith = -1
    with open(fname) as f:
        with open(output_fname, 'w') as fo:
            for line in f:
                ith += 1
                id_, txys, time, timestamp = line.strip().split('\t')
                matched = load_matched(matched_folder, id_)
#               matched = load_matched(matched_folder, ith)

                if matched is None:
                    no_count += 1
                    continue
                else:
                    txys = eval(txys)

                    count += 1

                    new_txys = []
                    try:
                        for txy, m in zip(txys, matched):

                            if len(new_txys) == 0:
                                new_x, new_y = map(float, m['geom'][7:-1].split(' '))
                                new_txys.append((txy[0], new_x, new_y))
                            else:
                                diff_t = txy[0] - new_txys[-1][0]
                                new_xys = []
                                for s in m['geom'][12:-1].split(', '):
                                    new_x, new_y = map(float, s.split(' '))
                                    new_xys.append((new_x, new_y))
                                times = split_time(new_txys[-1][0], diff_t, new_xys)

                                for t, (x, y) in zip(times, new_xys[1:]):
                                    new_txys.append((t, x, y))

                        fo.write('%s\t%s\t%s\t%s\n' % (id_, new_txys, time, timestamp))
                    except:
                        continue

                if ith % 100 == 0:
                    print ith, count, no_count

    print count, no_count

    return 0

def split_time(base_t, diff_t, new_xys):
    total_distance = 0
    for i in range(1, len(new_xys)):
        total_distance += misc.get_distance(new_xys[i-1][0], new_xys[i-1][1],
                                            new_xys[i][0], new_xys[i][1])+0.00001
    times = []
    current_distance = 0.0
    for i in range(1, len(new_xys)):
        current_distance += misc.get_distance(new_xys[i-1][0], new_xys[i-1][1],
                                              new_xys[i][0], new_xys[i][1])+0.00001
        times.append(base_t + diff_t * current_distance / total_distance)
    return times

def load_matched(matched_folder, id_):
    fname = os.path.join(matched_folder, '%s.json' % id_)
    if not os.path.exists(fname):
        print fname
        return None
    try:
        with open(fname) as f:
            ith = 0
            for line in f:
                if ith != 2:
                    ith += 1
                    continue
                return eval(line)
        return None
    except IOError:
        return None
    except SyntaxError:
        return None

def load_trajs(fname):
    trajs = {}
    ith = 0
    with open(fname) as f:
        for line in f:
            _, txys, _ = line.split('\t')
            traj = eval(txys)
            times, lats, lons = [], [], []
            for (t, lon, lat) in traj:
                lats.append(lat)
                lons.append(lon)
                times.append(t)
            trajs[ith] = (lons, lats, times)
            ith += 1
    return trajs


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))
