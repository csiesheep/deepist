#!/usr/bin/python
# -*- encoding: utf8 -*-

import sys
import optparse


__author__ = 'sheep'


def main(train_fname):
    '''\
    %prog [options] <train_fname>
    '''
    lengths = []
    with open(train_fname) as f:
        for line in f:
            line = line.strip()
            lengths.append(len(line.split(' ')[1].split(',')))
    print len(lengths)

    print_distribution(lengths)
    return 0

def print_distribution(lengths):
    length2count = {}
    for value in lengths:
        if value not in length2count:
            length2count[value] = 1
            continue
        length2count[value] += 1
    total = sum(length2count.values())
    acc = 0.0

    for key, value in sorted(length2count.items()):
        acc += value
        print key, value, acc/total


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))

