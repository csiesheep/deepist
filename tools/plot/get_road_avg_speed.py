#!/usr/bin/python
# -*- encoding: utf8 -*-

import datetime
import optparse
import sys

from tools import misc


__author__ = 'sheep'


def main(options, fname, output_fname):
    '''\
    %prog [options] <path_fname> <output_fname>
    '''
    ith = 0
    road_count = {}
    road_id_count = {}
    with open(fname) as f:
        for line in f:
            _, txys, _, ts = line.strip().split('\t')
            txys = eval(txys)
            for i in range(1, len(txys)):
                pre_lon, pre_lat = txys[i-1][1], txys[i-1][2]
                lon, lat = txys[i][1], txys[i][2]
                key = (pre_lon, pre_lat, lon, lat)
                if key not in road_count:
                    road_count[key] = 1
                else:
                    road_count[key] += 1

                road_id = txys[i-1][3]
                if road_id is None:
                    continue
                if road_id not in road_id_count:
                    road_id_count[road_id] = 1
                else:
                    road_id_count[road_id] += 1
            ith += 1
            if ith % 1000 == 0:
                print ith, len(road_count)

    ith = 0
    road_times = {}
    road_id_times = {}
    for road in road_count:
        if road_count[road] >= options.min_count:
            road_times[road] = [[], [], [], [], [], [],
                                [], [], [], [], [], [],
                                [], [], [], [], [], [],
                                [], [], [], [], [], []]
    for road_id in road_id_count:
        if road_id_count[road_id] >= options.min_count:
            road_id_times[road_id] = [[], [], [], [], [], [],
                                      [], [], [], [], [], [],
                                      [], [], [], [], [], [],
                                      [], [], [], [], [], []]

    with open(fname) as f:
        for line in f:
            _, txys, _, ts = line.strip().split('\t')
            hour = datetime.datetime.fromtimestamp(int(ts)).hour
            txys = eval(txys)

            # get speeds of roads (represented by lons and lats)
            for i in range(1, len(txys)):
                pre_lon, pre_lat = txys[i-1][1], txys[i-1][2]
                lon, lat = txys[i][1], txys[i][2]
                diff_time = txys[i][0]-txys[i-1][0]

                key = (pre_lon, pre_lat, lon, lat)
                if key not in road_times:
                    continue
                road_times[key][hour].append(diff_time)

            # get speeds of roads (represented by road ids)
            for i in range(1, len(txys)):
                road_id = txys[i-1][3]
                if road_id is None:
                    continue

                pre_lon, pre_lat = txys[i-1][1], txys[i-1][2]
                lon, lat = txys[i][1], txys[i][2]
                diff_time = txys[i][0]-txys[i-1][0]
                d = misc.get_distance(pre_lon, pre_lat, lon, lat)

                if road_id not in road_id_times:
                    continue
                road_id_times[road_id][hour].append((d, diff_time))

            ith += 1
            if ith % 1000 == 0:
                print ith, len(road_times)

    with open(output_fname, 'w') as fo:
        # output road (represented by lons and lats) speeds
        for r, hour_ts in road_times.items():
            d = misc.get_distance(*r)

            count = sum([len(ts) for ts in hour_ts])
            t = sum([sum(ts) for ts in hour_ts])
            if count < options.min_count:
                continue
            t = float(t)/count
            avg_speed = d/t

            hour_avg_speeds = []
            for ts in hour_ts:
                if len(ts) >= options.min_count:
                    t = sum(ts)/len(ts)
                    speed = d/t
                    hour_avg_speeds.append(speed)
                else:
                    hour_avg_speeds.append(None)

            fo.write('%s\t%s\t%f\n' % (r, str(hour_avg_speeds), avg_speed))

        # output road (represented by road_ids) speeds
        for r, hour_dts in road_id_times.items():
            count = sum([len(dts) for dts in hour_dts])
            if count < options.min_count:
                continue
            sum_d = 0.0
            sum_t = 0.0
            for dts in hour_dts:
                for d, t in dts:
                    sum_d += d
                    sum_t += t
            avg_speed = sum_d/sum_t

            hour_avg_speeds = []
            for dts in hour_dts:
                if len(dts) >= options.min_count:
                    sum_d = 0.0
                    sum_t = 0.0
                    for d, t in dts:
                        sum_d += d
                        sum_t += t
                    speed = sum_d/sum_t
                    hour_avg_speeds.append(speed)
                else:
                    hour_avg_speeds.append(None)

            fo.write('%s\t%s\t%f\n' % (r, str(hour_avg_speeds), avg_speed))


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    parser.add_option("-m", "--min_count", dest="min_count",
                      action="store", type="int", default=5)
    options, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
        sys.exit()

    sys.exit(main(options, *args))

