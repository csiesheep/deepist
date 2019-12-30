#!/usr/bin/python
# -*- encoding: utf8 -*-


from functools import reduce
import numpy as np
import tensorflow as tf


class DeepIST(object):

    def __init__(self, image_input, length_input, max_len, trainable=True, dropout=0.5, size=200, channel=1, cnn2d_dim=1024, cnn1d_dim=1024, cnn2d_convs=[64, 128, 256, 512]):
        self.data_dict = None
        self.image_input= image_input   # (batch, max_len, size, size, channel)
        self.length_input = length_input # (batch)
        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.max_len = max_len
        self.size = size
        self.channel = channel
        self.cnn1d_dim = cnn1d_dim
        self.cnn2d_dim = cnn2d_dim
        self.cnn2d_convs = cnn2d_convs
        self.build()

    def build(self, train_mode=None):
        mask = tf.sequence_mask(self.length_input, self.max_len)
        # (sum of lengths, size, size, channel)
        cnn2d_input = tf.boolean_mask(self.image_input, mask)
        self.cnn2d = PathCNN(image_input=cnn2d_input,
                             channel=self.channel,
                             embedding_dim=self.cnn2d_dim,
                             convs=self.cnn2d_convs)
        mask = tf.reshape(mask, [-1])
        indexes = tf.boolean_mask([[i] for i in range(mask.shape[0])], mask)
        # self.cnn2d.final (sum of lengths, cnn2d_dim)
        # (batch * self.max_len, self.cnn2d_dim)
        cnn2d_embeddings = tf.scatter_nd(indexes, self.cnn2d.final, [self.length_input.shape[0]*self.max_len, self.cnn2d_dim])
        # (batch, self.max_len, self.cnn2d_dim)

        cnn2d_embeddings = tf.reshape(cnn2d_embeddings,
                                      [-1, self.max_len, self.cnn2d_dim])

        self.cnn1d = Simple1dCNN(image_input=cnn2d_embeddings,
                                 input_dim=self.cnn2d_dim,
                                 output_dim=self.cnn1d_dim)
        cnn1d_embeddings = self.cnn1d.final # (batch, cnn1d_dim)

        self.sub_fc1 = self.fc_layer(self.cnn2d.final, self.cnn2d_dim, self.cnn2d_dim, "sub_fc1")
        self.sub_relu1 = tf.nn.relu(self.sub_fc1)
        if train_mode is not None:
            self.sub_relu1 = tf.cond(train_mode,
                                     lambda: tf.nn.dropout(self.sub_relu1, self.dropout),
                                     lambda: self.sub_relu1)
        elif train_mode:
            self.sub_relu1 = tf.nn.dropout(self.sub_relu1, self.dropout)
#       self.sub_fc2 = self.fc_layer(self.cnn2d.final, self.cnn2d_dim, 1, "sub_fc2")
        self.sub_fc2 = self.fc_layer(self.sub_relu1, self.cnn2d_dim, 1, "sub_fc2")
        # self.sub_final (sum of lengths, 1)
        self.sub_final = self.sub_fc2


        self.fc1 = self.fc_layer(cnn1d_embeddings, self.cnn1d_dim, self.cnn1d_dim, "2cnn_fc1")
        self.relu1 = tf.nn.relu(self.fc1)
        if train_mode is not None:
            self.relu1 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu1, self.dropout), lambda: self.relu1)
        elif train_mode:
            self.relu1 = tf.nn.dropout(self.relu1, self.dropout)

        self.fc2 = self.fc_layer(self.relu1, self.cnn1d_dim, 1, "2cnn_fc2")
        self.final = self.fc2

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None:
            if name == 'fc8' and var_name == 'fc8_weights':
                value = self.data_dict[name][idx][:, 0:1]
            elif name == 'fc8' and var_name == 'fc8_biases':
                value = self.data_dict[name][idx][:1]
            else:
                value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var


class Simple1dCNN(object):

    def __init__(self, image_input, trainable=True, dropout=0.5, input_dim=1024, output_dim=1024):
        self.data_dict = None
        self.image_input= image_input # (batch, max_len, input_dim)
        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build()

    def build(self, train_mode=None):
        self.conv1_1 = self.conv_layer(self.image_input, self.input_dim, self.input_dim, "conv1_1")
        self.pool1 = self.max_pool(self.conv1_1, 'pool1')
#       self.fc6 = self.fc_layer(self.pool1, int(self.pool1.shape[1]*self.pool1.shape[2]), self.output_dim, "fc6") #200, 1conv

        self.conv2_1 = self.conv_layer(self.pool1, self.input_dim, self.input_dim, "conv2_1")
        self.pool2 = self.max_pool(self.conv2_1, 'pool2')
        self.fc6 = self.fc_layer(self.pool2, int(self.pool2.shape[1]*self.pool2.shape[2]), self.output_dim, "fc6") #200

#       self.conv3_1 = self.conv_layer(self.pool2, self.input_dim, self.input_dim, "conv3_1")
#       self.pool3 = self.max_pool(self.conv3_1, 'pool3')
#       self.fc6 = self.fc_layer(self.pool3, int(self.pool3.shape[1]*self.pool3.shape[2]), self.output_dim, "fc6") #200, 3conv

        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif train_mode:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.final = self.relu6 # (batch, self.embedding_dim)

        self.data_dict = None

    def max_pool(self, bottom, name):
        return tf.layers.max_pooling1d(bottom, 2, 2, padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv1d(bottom, filt, 1, padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None:
            if name == 'fc8' and var_name == 'fc8_weights':
                value = self.data_dict[name][idx][:, 0:1]
            elif name == 'fc8' and var_name == 'fc8_biases':
                value = self.data_dict[name][idx][:1]
            else:
                value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count


class PathCNN(object):

    def __init__(self, image_input, trainable=True, dropout=0.5, channel=1, embedding_dim=2048, convs=[64, 128, 256, 512]):
        self.data_dict = None
        self.image_input= image_input # (sum of lengths, size, size, channel)
        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.channel = channel
        self.embedding_dim = embedding_dim
        self.convs = convs
        self.build()

    def build(self, train_mode=None):

        # conv->max + conv->avg
        self.conv1_avg = self.conv_layer(self.image_input, self.channel, self.convs[0], "conv1_avg")
        self.conv1_max = self.conv_layer(self.image_input, self.channel, self.convs[0], "conv1_max")
        self.pool1_avg = self.avg_pool(self.conv1_avg, 'pool1_avg')
        self.pool1_max = self.max_pool(self.conv1_max, 'pool1_max')
        self.pool1 = tf.concat([self.pool1_avg, self.pool1_max], 3)

        self.conv2_avg = self.conv_layer(self.pool1, self.convs[0]*2, self.convs[1], "conv2_avg")
        self.conv2_max = self.conv_layer(self.pool1, self.convs[0]*2, self.convs[1], "conv2_avg")
        self.pool2_avg = self.avg_pool(self.conv2_avg, 'pool2_avg')
        self.pool2_max = self.max_pool(self.conv2_max, 'pool2_max')
        self.pool2 = tf.concat([self.pool2_avg, self.pool2_max], 3)

        self.conv3_avg = self.conv_layer(self.pool2, self.convs[1]*2, self.convs[2], "conv3_avg")
        self.conv3_max = self.conv_layer(self.pool2, self.convs[1]*2, self.convs[2], "conv3_avg")
        self.pool3_avg = self.avg_pool(self.conv3_avg, 'pool3_avg')
        self.pool3_max = self.max_pool(self.conv3_avg, 'pool3_max')
        self.pool3 = tf.concat([self.pool3_avg, self.pool3_max], 3)

        self.conv4_avg = self.conv_layer(self.pool3, self.convs[2]*2, self.convs[3], "conv4_avg")
        self.conv4_max = self.conv_layer(self.pool3, self.convs[2]*2, self.convs[3], "conv4_avg")
        self.pool4_avg = self.avg_pool(self.conv4_avg, 'pool4_avg')
        self.pool4_max = self.max_pool(self.conv4_max, 'pool4_max')
        self.pool4 = tf.concat([self.pool4_avg, self.pool4_max], 3)

#       # conv --> max+avg
#       self.conv1_1 = self.conv_layer(self.image_input, self.channel, self.convs[0], "conv1_1")
#       self.pool1_avg = self.avg_pool(self.conv1_1, 'pool1_avg')
#       self.pool1_max = self.max_pool(self.conv1_1, 'pool1_max')
#       self.pool1 = tf.concat([self.pool1_avg, self.pool1_max], 3)

#       self.conv2_1 = self.conv_layer(self.pool1, self.convs[0]*2, self.convs[1], "conv2_1")
#       self.pool2_avg = self.avg_pool(self.conv2_1, 'pool2_avg')
#       self.pool2_max = self.max_pool(self.conv2_1, 'pool2_max')
#       self.pool2 = tf.concat([self.pool2_avg, self.pool2_max], 3)

#       self.conv3_1 = self.conv_layer(self.pool2, self.convs[1]*2, self.convs[2], "conv3_1")
#       self.pool3_avg = self.avg_pool(self.conv3_1, 'pool3_avg')
#       self.pool3_max = self.max_pool(self.conv3_1, 'pool3_max')
#       self.pool3 = tf.concat([self.pool3_avg, self.pool3_max], 3)

#       self.conv4_1 = self.conv_layer(self.pool3, self.convs[2]*2, self.convs[3], "conv4_1")
#       self.pool4_avg = self.avg_pool(self.conv4_1, 'pool4_avg')
#       self.pool4_max = self.max_pool(self.conv4_1, 'pool4_max')
#       self.pool4 = tf.concat([self.pool4_avg, self.pool4_max], 3)

#       self.conv1_1 = self.conv_layer(self.image_input, self.channel, self.convs[0], "conv1_1")
#       self.pool1 = self.avg_pool(self.conv1_1, 'pool1')
##      self.pool1 = self.max_pool(self.conv1_1, 'pool1')

#       self.conv2_1 = self.conv_layer(self.pool1, self.convs[0], self.convs[1], "conv2_1")
#       self.pool2 = self.avg_pool(self.conv2_1, 'pool2')
##      self.pool2 = self.max_pool(self.conv2_1, 'pool2')

#       self.conv3_1 = self.conv_layer(self.pool2, self.convs[1], self.convs[2], "conv3_1")
#       self.pool3 = self.avg_pool(self.conv3_1, 'pool3')
##      self.pool3 = self.max_pool(self.conv3_1, 'pool3')
##      self.fc6 = self.fc_layer(self.pool3, 21632, self.embedding_dim, "fc6") #100, 3conv
##      self.fc6 = self.fc_layer(self.pool3, 80000, self.embedding_dim, "fc6") #200, 3conv

#       self.conv4_1 = self.conv_layer(self.pool3, self.convs[2], self.convs[3], "conv4_1")
#       self.pool4 = self.avg_pool(self.conv4_1, 'pool4')
##      self.pool4 = self.max_pool(self.conv4_1, 'pool4')

 #      self.fc6 = self.fc_layer(self.pool4, 8192, self.embedding_dim, "fc6") #50
 #      self.fc6 = self.fc_layer(self.pool3, 13*13*self.convs[-2]*2, self.embedding_dim, "fc6") #100
        self.fc6 = self.fc_layer(self.pool4, 7*7*self.convs[-1]*2, self.embedding_dim, "fc6") #100
 #      self.fc6 = self.fc_layer(self.pool5, 4*4*self.convs[-1]*2, self.embedding_dim, "fc6") #100
 #      self.fc6 = self.fc_layer(self.pool4, 7*7*self.convs[-1], self.embedding_dim, "fc6") #100

        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif train_mode:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.final = self.relu6 # (sum of lengths, self.embedding_dim)

        self.data_dict = None

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filters, biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filters, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None:
            if name == 'fc8' and var_name == 'fc8_weights':
                value = self.data_dict[name][idx][:, 0:1]
            elif name == 'fc8' and var_name == 'fc8_biases':
                value = self.data_dict[name][idx][:1]
            else:
                value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
