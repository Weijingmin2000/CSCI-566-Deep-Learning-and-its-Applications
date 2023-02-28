from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.layer_utils import *


""" Super Class """
class Module(object):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, feat, is_training=True, seed=None):
        output = feat
        for layer in self.net.layers:
            if isinstance(layer, dropout):
                output = layer.forward(output, is_training, seed)
            else:
                output = layer.forward(output)
        self.net.gather_params()
        return output

    def backward(self, dprev, regularization="none", reg_lambda=0.0):
        for layer in self.net.layers[::-1]:
            dprev = layer.backward(dprev)
        self.net.gather_grads()
        if regularization == "l1":
            self.net.apply_l1_regularization(reg_lambda)
        elif regularization == "l2":
            self.net.apply_l2_regularization(reg_lambda)
        elif regularization != "none":
            raise NotImplementedError
        return dprev


""" Classes """
class TestFCGeLU(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            flatten(name="flatten_tiny"),
            fc(input_dim=15, output_dim=5, name='fc_tiny'),
            gelu(name='gelu_tiny')
            ########### END ###########
        )


class SmallFullyConnectedNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            fc(input_dim=4, output_dim=30, init_scale=0.02, name='fc1'),
            gelu(name='gelu1'),
            fc(input_dim=30, output_dim=7, init_scale=0.02, name='fc2')
            ########### END ###########
        )


class DropoutNet(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.dropout = dropout
        self.seed = seed
        self.net = sequential(
            flatten(name="flat"),
            fc(15, 20, 5e-2, name="fc1"),
            gelu(name="gelu1"),
            fc(20, 30, 5e-2, name="fc2"),
            gelu(name="gelu2"),
            fc(30, 10, 5e-2, name="fc3"),
            gelu(name="gelu3"),
            dropout(keep_prob, seed=seed)
        )


class TinyNet(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        """ Some comments """
        self.net = sequential(
            ########## TODO: ##########
            flatten(name='flat'),
            fc(input_dim=3072, output_dim=1024, init_scale=5e-2, name='fc1'),
            gelu(name="gelu1"),
            fc(1024, 20, init_scale=5e-2, name='fc2')
            ########### END ###########
        )

class DropoutNetTest(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        """ Some comments """
        self.dropout = dropout
        self.seed = seed
        self.net = sequential(
            flatten(name="flat"),
            fc(3072, 500, 1e-2, name="fc1"),
            dropout(keep_prob, seed=seed),
            gelu(name="gelu1"),
            fc(500, 500, 1e-2, name="fc2"),
            gelu(name="gelu2"),
            fc(500, 20, 1e-2, name="fc3"),
        )


class FullyConnectedNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        """ Some comments """
        self.net = sequential(
            flatten(name="flat"),
            fc(3072, 100, 0.1, name="fc1"),
            gelu(name="gelu1"),
            fc(100, 100, 0.1, name="fc2"),
            gelu(name="gelu2"),
            fc(100, 100, 0.1, name="fc3"),
            gelu(name="gelu3"),
            fc(100, 100, 0.1, name="fc4"),
            gelu(name="gelu4"),
            fc(100, 100, 0.1, name="fc5"),
            gelu(name="gelu5"),
            fc(100, 20, 0.1, name="fc6")
        )

