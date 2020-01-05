#!/usr/bin/python
# -*- encoding: utf8 -*-

import os
from PIL import Image
import random
import tempfile
import tensorflow as tf


__author__ = 'sheep'



MEAN = tf.constant([127, 127, 0], dtype=tf.float32)
RANGE = tf.constant([0, 1, 0], dtype=tf.float32)

class DataGenerator(object):

    def __init__(self, fname, image_folder, batch_size, image_width, image_height,
                 ratio=0.1, max_len=10, channel=1,
                 shuffle=True, epoch=None, one_image=False):
        self.fname = fname
        self.image_folder = image_folder
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.ratio = ratio
        self.max_len = max_len
        self.channel = channel
        self.null_fname = self.generate_null_image()
        self.shuffle = shuffle
        self.epoch = epoch
        self.one_image = one_image
        train, test = self.load_data()
        self.train_image_batch, self.train_lenght_batch, self.train_sub_value_batch, self.train_value_batch = train
        self.test_image_batch, self.test_length_batch, self.test_sub_value_batch, self.test_value_batch = test

    def generate_null_image(self):
        image = Image.new('RGB', (self.image_height, self.image_width))
        null_fname = tempfile.mktemp(suffix='.bmp')
        image.save(null_fname)
        return null_fname

    def load_data(self):
        def load_images(fnames):
            images = tf.read_file(fnames)
            images = tf.image.decode_bmp(images, channels=self.channel)
            if self.image_width is not None and self.image_height is not None:
                images = tf.image.resize_images(images,
                    [self.image_width, self.image_height],
                    align_corners=True)
#               images = tf.subtract(images, MEAN)
#               images = tf.multiply(images, RANGE)
            return images

        def batch(fname_lists, lengths, sub_values, values):
            queue = tf.train.slice_input_producer([fname_lists, lengths, sub_values, values],
                                                  shuffle=self.shuffle, num_epochs=self.epoch)
            if self.one_image:
                image_lists = load_images(queue[0])
            else:
                image_lists = tf.map_fn(load_images, queue[0], dtype=tf.float32)
            a_batch = tf.train.batch([image_lists, queue[1], queue[2], queue[3]],
                                     batch_size=self.batch_size)
            # (batch, max_len, h, w, c)
            return a_batch

        train_fname_lists, train_lengths, train_sub_values, train_values = [], [], [], []
        test_fname_lists, test_lengths, test_sub_values, test_values = [], [], [], []
        with open(self.fname) as f:
            for line in f:
                image_fnames, sub_values, value = line.strip().split()
                image_fnames = [os.path.join(self.image_folder, f)
                                for f in image_fnames.split(',')]
                sub_values = [[v] for v in map(float, sub_values.split(','))]
                value = [float(value)]
                length = len(image_fnames)

                if len(image_fnames) > self.max_len:
                    image_fnames = image_fnames[:self.max_len]
                    sub_values = sub_values[:self.max_len]
                elif len(image_fnames) < self.max_len:
                    image_fnames.extend([self.null_fname] * (self.max_len-len(image_fnames)))
                    sub_values.extend([[0.0]] * (self.max_len-len(sub_values)))
                if self.one_image:
                    image_fnames = image_fnames[0]

                if random.random() > self.ratio:
                    train_fname_lists.append(image_fnames)
                    train_lengths.append(length)
                    train_sub_values.append(sub_values)
                    train_values.append(value)
                else:
                    test_fname_lists.append(image_fnames)
                    test_lengths.append(length)
                    test_sub_values.append(sub_values)
                    test_values.append(value)

        train_image_batch, train_lenght_batch, train_sub_value_batch, train_value_batch = batch(
                train_fname_lists,
                train_lengths,
                train_sub_values,
                train_values)
        if self.ratio == 0:
            test_image_batch = None
            test_length_batch = None
            test_sub_value_batch = None
            test_value_batch = None
        else:
            test_image_batch, test_length_batch, test_sub_value_batch, test_value_batch = batch(
                    test_fname_lists,
                    test_lengths,
                    test_sub_values,
                    test_values)
        return ((train_image_batch, train_lenght_batch, train_sub_value_batch, train_value_batch),
                (test_image_batch, test_length_batch, test_sub_value_batch, test_value_batch))
