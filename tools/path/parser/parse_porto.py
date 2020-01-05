#!/usr/bin/python
# -*- encoding: utf8 -*-

import csv
import optparse
import sys


__author__ = 'sheep'


def main(fname, output_fname):
    '''\
    Parse trajectories from raw data.
    Each line is a trajectory with the following format:

    uid [(time, lon., lat.,)] travel_time departure_timestamp  #seperated by <TAB>

    %prog [options] <fname> <output_fname>
    '''
    with open(fname, 'r') as f:
        with open(output_fname, 'w') as fo:
            first = True
            spamreader = csv.reader(f, delimiter=',', quotechar='"')
            for line in spamreader:
                if first:
                    first = False
                    continue

                xys = eval(line[-1])
                txys = [(i*0.25, x, y) for i, (x, y)
                        in enumerate(xys)]
                id_ = "%s-%s" % (line[0], line[4])
                time = (len(txys)-1)*0.25
                # id, [(time, lon, lat)], travel_time, departure timestamp
                fo.write("%s\t%s\t%.3f\t%s\n" % (id_, str(txys), time, line[5]))
    return 0


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))

