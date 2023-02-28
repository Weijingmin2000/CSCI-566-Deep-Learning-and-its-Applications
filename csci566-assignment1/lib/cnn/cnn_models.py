from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=3, stride=1, padding=0, name='conv1'),
            MaxPoolingLayer(pool_size=2, stride=2, name='maxpool1'),
            flatten(name='flatten'),
            fc(input_dim=27, output_dim=5, init_scale=0.02, name='fc1'),
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.seed=seed
        self.dropout = dropout
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 3, 32, 1, 0, name = "conv1"),
            gelu(name="lr1"),
            MaxPoolingLayer(3, 2, name = "pool1"),
            ConvLayer2D(32, 3, 32, 1, 0, name = "conv2"),
            gelu(name="lr2"),
            MaxPoolingLayer(3, 1, name = "pool2"),
            flatten(name = "flatten1"),
            fc(3200, 200, 0.02, name="fc1"),
            gelu(name="lr3"),
            dropout(keep_prob=0.6, seed=seed, name="dropout1"),
            fc(200, 20, 0.02, name="fc2"),
            ########### END ###########
        )