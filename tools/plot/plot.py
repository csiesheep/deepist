#!/usr/bin/python
# -*- encoding: utf8 -*-

import datetime
import optparse
import os
from PIL import Image, ImageDraw
import Queue
import sys
import tempfile

from tools import misc
from tools.plot import road_tools


__author__ = 'sheep'


base_geo_x_range = 0.0117398 #1km
base_geo_y_range = 0.0089932 #1km
geo_x_range = None
geo_y_range = None

max_speed = 150


def main(options, path_fname, speed_fname, pbf_fname, output_folder, output_fname):
    '''\
    %prog [options] <path_fname> <speed_fname> <pbf_fname> <output_folder> <output_training_fname>
    '''
    roads = None
    if options.plot_road:
        roads = road_tools.RoadNetwork2.load_from_file(pbf_fname)

    global geo_x_range
    geo_x_range = base_geo_x_range * options.window_size
    global geo_y_range
    geo_y_range = base_geo_y_range * options.window_size

    no_speed_count = 0
    road2speed = load_speed(speed_fname, options.plot_speed)
    ith = -1
    with open(output_fname, 'w') as fo:
        with open(path_fname) as f:
            for line in f:
                ith += 1
#               if ith < 500000:
#                   continue
                if ith % 1000 == 0:
                    print datetime.datetime.now(), ith

                traj, time, timestamp = load_trajs(line)
                speeds = get_speeds(traj, timestamp, road2speed,
                                    options.plot_speed, options.is_hourly)
                if speeds is None:
                    no_speed_count += 1
                    continue
                sub_trajs, sub_intervals, sub_speeds = split(traj, speeds, options.window_size)
                assert len(sub_trajs) == len(sub_intervals)
                assert len(sub_trajs) == len(sub_speeds)
                fnames = plot_seq(ith, sub_trajs, sub_speeds, roads, output_folder,
                                  options.image_size, options.window_size)
                fo.write("%s %s %s\n" % (','.join(fnames),
                                         ','.join(map(str, sub_intervals)),
                                         time))

    print "no speeds path count:", no_speed_count
    return 0

def get_speeds(traj, timestamp, road2speed, plot_speed, is_hourly):
    if not plot_speed:
        speeds = [max_speed] * (len(traj[0])-1)
        return speeds
    speeds = []
    lons, lats, times, roads = traj
    hour = datetime.datetime.fromtimestamp(int(timestamp)).hour
    real_speeds = []
    for i in range(1, len(lons)):
        if roads[i-1] is not None:
            road = roads[i-1]
        else:
            road = (lons[i-1], lats[i-1], lons[i], lats[i])

        if road not in road2speed:
            speeds.append(None)
        else:
            if is_hourly:
                speeds.append(road2speed[road][0][hour])
            else:
                speeds.append(road2speed[road][1])

    speeds = estimate_missing_speeds(speeds)
    return speeds

def estimate_missing_speeds(speeds):
    pre_speed = None
    i = 0
    while i < len(speeds):
        if speeds[i] is not None:
            pre_speed = speeds[i]
            i += 1
            continue
        j = i+1
        while j < len(speeds):
            if speeds[j] is not None:
                break
            j += 1

        if pre_speed is None and j == len(speeds):
            return None

        if j == len(speeds):
            for k in range(i, j):
                speeds[k] = pre_speed
        elif pre_speed is None:
            for k in range(i, j):
                speeds[k] = speeds[j]
        else:
            for k in range(i, j):
                speeds[k] = (pre_speed+speeds[j])/2
        i = j
    return speeds

def load_speed(speed_fname, plot_speed):
    print 'loading speeds..'
    road2speed = {}
    if not plot_speed:
        return road2speed
    with open(speed_fname) as f:
        for line in f:
            road, hour_avg_speeds, avg_speed = line.strip().split('\t')
            hour_avg_speeds = eval(hour_avg_speeds)
            for i in range(len(hour_avg_speeds)):
                if hour_avg_speeds[i] is not None:
                    hour_avg_speeds[i] = hour_avg_speeds[i]*60
            avg_speed = float(avg_speed)*60
            try:
                road = eval(road)
            except:
                pass
            road2speed[road] = (hour_avg_speeds, avg_speed)
    print len(road2speed)
    return road2speed

def split(traj, speeds, window_size):
    if len(traj[0]) == 0:
        return [], [], []
    sub_trajs = []
    sub_speeds = []
    sub_sub_speeds = []
    sub_intervals = [0.0]
    sub_traj = []
    lons, lats, times, _ = traj
    lon, lat, time = lons[0], lats[0], times[0]
    sub_lons, sub_lats = [lon], [lat]
    current = 0
    i = 0
    while i+1 != len(traj[0]):
        next_lon, next_lat, next_time = lons[i+1], lats[i+1], times[i+1]
        distance = misc.get_distance(lon, lat, next_lon, next_lat)
        sub_sub_speeds.append(speeds[i])
        if distance + current <= window_size:
            sub_lons.append(next_lon)
            sub_lats.append(next_lat)
            sub_intervals[-1] += next_time-time
            lon, lat, time = next_lon, next_lat, next_time
            current += distance
            i += 1
        else:
            ratio = (window_size-current)/distance
            new_lon = lon+(next_lon-lon)*ratio
            new_lat = lat+(next_lat-lat)*ratio
            new_time = time + (next_time-time)*ratio
            sub_intervals[-1] += new_time-time
            sub_intervals.append(0.0)
            sub_lons.append(new_lon)
            sub_lats.append(new_lat)
            sub_trajs.append((sub_lons, sub_lats))
            sub_lons, sub_lats = [new_lon], [new_lat]
            sub_speeds.append(sub_sub_speeds)
            sub_sub_speeds = []
            lon, lat, time = new_lon, new_lat, new_time
            current = 0
    if len(sub_lons) != 0:
        sub_trajs.append((sub_lons, sub_lats))
        sub_speeds.append(sub_sub_speeds)
    return sub_trajs, sub_intervals, sub_speeds

def plot_seq(id_, sub_trajs, sub_speeds, roads, output_folder, size, window_size):
    fnames = []
    for ith, (sub_traj, sub_sub_speeds) in enumerate(zip(sub_trajs, sub_speeds)):
        base_fname = '%d_%d.bmp' % (id_, ith)
        fname = os.path.join(output_folder, base_fname)
        plot_a_traj(sub_traj, sub_sub_speeds, roads, size, window_size, fname)
        ith += 1
        fnames.append(base_fname)
    return fnames

def plot_a_traj(traj, speeds, graph, size, window_size, fname):

    def get_boundary(traj):
        lons = traj[0]
        lats = traj[1]
        max_lat = max(lats)
        min_lat = min(lats)
        max_lon = max(lons)
        min_lon = min(lons)
        center_lat, center_lon = (max_lat+min_lat)/2, (max_lon+min_lon)/2
        return ((center_lon+geo_x_range/2),
                (center_lon-geo_x_range/2),
                (center_lat+geo_y_range/2),
                (center_lat-geo_y_range/2))

    def to_pixels(lons, lats, size, max_lon, min_lon, max_lat, min_lat):
        pixels = []
        for lon, lat in zip(lons, lats):
            x = int((lon-min_lon)*size/geo_x_range)
            y = size - int((lat-min_lat)*size/geo_y_range)
            pixels.append((x, y))
        return pixels

    def get_speed_value(speed):
        value = int((speed/max_speed)*255)
        if value > 255:
            value = 255
        return value

    def get_pixel_count(x1, y1, x2, y2):
        if abs(x1-x2) > abs(y1-y2):
            return abs(x1-x2)+1
        return abs(y1-y2)+1

    def get_dist_value(d):
        value = int((d/(2*window_size/size))*255)
        if value > 255:
            value = 255
        return value

    image = Image.new('RGB', (size, size), (0, 0, 0, 0))
#   image = Image.new('RGB', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    max_lon, min_lon, max_lat, min_lat = get_boundary(traj)
    path_pixels = to_pixels(traj[0], traj[1], size,
                       max_lon, min_lon, max_lat, min_lat)

    # draw path
    width = 1
    for i in range(1, len(path_pixels)):
        r_value = 255
        g_value = get_speed_value(speeds[i-1])
        color = (r_value, g_value, 0)
        line = [path_pixels[i-1][0],
                path_pixels[i-1][1],
                path_pixels[i][0],
                path_pixels[i][1]]
        draw.line(line, fill=color, width=width)

    # draw intersections of path
    for i in range(1, len(path_pixels)-1):
        r_value = 255
        g_value = get_speed_value((speeds[i-1]+speeds[i])/2)
        color = (r_value, g_value, 0)
        draw.point((path_pixels[i][0], path_pixels[i][1]), color)

    # draw nearby road network 
    if graph is not None:
        image2 = Image.new('RGB', (size, size), (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(image2)
        color = (0, 0, 255)
        max_depth = 5
        q = Queue.Queue()
        for lon, lat in zip(traj[0], traj[1]):
            q.put((lon, lat, 0))

        visited = set()
        while not q.empty():
            lon, lat, depth = q.get()
            visited.add((lon, lat))
            for ((to_lon, to_lat), weight) in graph.graph.get((lon, lat), []):
                if (to_lon, to_lat) in visited:
                    continue

                pixels = tuple(to_pixels([lon, to_lon], [lat, to_lat], size,
                                         max_lon, min_lon, max_lat, min_lat))
                if pixels[0] == pixels[1]:
                    continue
                line = [pixels[0][0], pixels[0][1],
                        pixels[1][0], pixels[1][1]]
                draw2.line(line, fill=color, width=1)

                if depth < max_depth:
                    q.put((to_lon, to_lat, depth+1))
        _, _, b = image2.split()
        b = b.load()
        pixels = image.load()
        for i in range(size):
            for j in range(size):
                try:
                    b_value = b[i, j]
                except IndexError:
                    b_value = 0
                pixels[i, j] = (pixels[i, j][0], pixels[i, j][1], b_value)

    image.save(fname)


def load_trajs(line):
    _, txys, time, timestamp = line.split('\t')
    traj = eval(txys)
    times, lons, lats, roads = [], [], [], []
    for (t, lon, lat, road) in traj:
        times.append(t)
        lons.append(lon)
        lats.append(lat)
        roads.append(road)
    return (lons, lats, times, roads), time, timestamp


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    parser.add_option("-w", "--window_size", dest="window_size",
                      action="store", type="float", default=0.5)
    parser.add_option("-p", "--image_size", dest="image_size",
                      action="store", type="int", default=100)
    parser.add_option("-o", "--hourly_speed", dest="is_hourly",
                      action="store_true", default=False)
    parser.add_option("-r", "--not_plot_road", dest="plot_road",
                      action="store_false", default=True)
    parser.add_option("-s", "--not_plot_speed", dest="plot_speed",
                      action="store_false", default=True)
    options, args = parser.parse_args()

    if len(args) != 5:
        parser.print_help()
        sys.exit()

    sys.exit(main(options, *args))

