#!/usr/bin/python
# -*- encoding: utf8 -*-

# experiment configurations
batch_size = 20
num_iter = 1000000
grad_iter = 100000
learning_rate = 1e-4
filewriter_path = "./tmp/tensorboard"
checkpoint_path = "./tmp/checkpoints"
validation_log = "./all.log"
ratio=0.1

# model configurations
image_width = image_height = 100
max_len = None
cnn1d_dim = 1024
cnn2d_dim = 1024
lstm_dim = 1024
cnn2d_convs = [16, 32, 64, 128]
filter_size = 3
channel = 3

# data configurations
image_folder = None

