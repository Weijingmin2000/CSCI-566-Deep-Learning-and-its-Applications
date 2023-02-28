from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.special import erf

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v # name : value
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v
    
    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ######## TODO ########
                v += lam * np.sign(layer.params[n])
                layer.grads[n] = v
                ######## END  ########
    
    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ######## TODO ########
                v += 2 * lam * layer.params[n]
                layer.grads[n] = v
                ######## END  ########


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))


class flatten(object):
    def __init__(self, name="flatten"):
        """
        - name: the name of current layer
        - meta: to store the forward pass activations for computing backpropagation
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, feat):
        output = None
        #############################################################################
        # TODO: Implement the forward pass of a flatten layer.                      #
        # You need to reshape (flatten) the input features.                         #
        # Store the results in the variable self.meta provided above.               #
        #############################################################################
        batch_size = feat.shape[0] # batch size
        # flaten: (batch, a, b, c, ...) -> (batch, a*b*c*...)
        output = feat.reshape(batch_size, np.prod(feat.shape[1:])) 
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat = None
        #############################################################################
        # TODO: Implement the backward pass of a flatten layer.                     #
        # You need to reshape (flatten) the input gradients and return.             #
        # Store the results in the variable dfeat provided above.                   #
        #############################################################################
        dfeat = dprev.reshape(feat.shape) # unflaten: dprev -> feat
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat


class fc(object):
    def __init__(self, input_dim, output_dim, init_scale=0.002, name="fc"):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - output_dim: output dimension
        - meta: to store the forward pass activations for computing backpropagation
        """
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None

    def forward(self, feat):
        output = None
        assert len(feat.shape) == 2 and feat.shape[-1] == self.input_dim, \
            "But got {} and {}".format(feat.shape, self.input_dim)
        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        output = np.dot(feat, self.params[self.w_name]) + self.params[self.b_name] # Xw+b
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        assert len(feat.shape) == 2 and feat.shape[-1] == self.input_dim, \
            "But got {} and {}".format(feat.shape, self.input_dim)
        assert len(dprev.shape) == 2 and dprev.shape[-1] == self.output_dim, \
            "But got {} and {}".format(dprev.shape, self.output_dim)
        #############################################################################
        # TODO: Implement the backward pass of a single fully connected layer.      #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        # dloss/dfeat = dloss/douput * doutput/dfeat
        # output = wx + b, dloss/doutput = dprev 
        self.grads[self.w_name] = np.dot(feat.T, dprev) # grad(w): dloss/dw = x.T * dprev, (12, 15) * (15, 15) = (12, 15)
        self.grads[self.b_name] = np.sum(dprev, axis=0) # grad(b): (15, )
        dfeat = np.dot(dprev, self.params[self.w_name].T) # grad(x): dloss/dx = dprev * w.T, (15, 15) * (15, 12) = (15, 12)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat

class gelu(object):
    def __init__(self, name="gelu"):
        """
        - name: the name of current layer
        - meta:  to store the forward pass activations for computing backpropagation
        Notes: params and grads should be just empty dicts here, do not update them
        """
        self.name = name 
        self.params = {}
        self.grads = {}
        self.meta = None 
    
    def forward(self, feat):
        output = None
        #############################################################################
        # TODO: Implement the forward pass of GeLU                                  #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        # 0.5(1 + tanh( √(2/π)(x + 0.044715x^3)) )
        output = 0.5 * feat * (1 + np.tanh(np.sqrt(2 / np.pi) * (feat + 0.044715 * pow(feat, 3))))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output
    
    def backward(self, dprev):
        """ You can use the approximate gradient for GeLU activations """
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat = None
        #############################################################################
        # TODO: Implement the backward pass of GeLU                                 #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        # dloss/dfeat = dloss/douput * doutput/dfeat
        # 0.5(1 + tanh( √(2/π)(x + 0.044715x^3)) ) + 0.5x√(2/π)(1 - tanh^2( √(2/π) * (x + 0.044715x^3)) ) * (1 + 0.134145x^2)
        tanh_part = np.tanh(np.sqrt(2 / np.pi) * (feat + 0.044715 * np.power(feat, 3)))
        doutput_dfeat = 0.5 * (1 + tanh_part) + \
                0.5 * feat * (1 - tanh_part ** 2) * np.sqrt(2 / np.pi) * (1 + 0.134145 * np.power(feat, 2))
        dfeat = dprev * doutput_dfeat
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat



class dropout(object):
    def __init__(self, keep_prob, seed=None, name="dropout"):
        """
        - name: the name of current layer
        - keep_prob: probability that each element is kept.
        - meta: to store the forward pass activations for computing backpropagation
        - kept: the mask for dropping out the neurons
        - is_training: dropout behaves differently during training and testing, use
                       this to indicate which phase is the current one
        - rng: numpy random number generator using the given seed
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.keep_prob = keep_prob
        self.meta = None
        self.kept = None
        self.is_training = False
        self.rng = np.random.RandomState(seed)
        assert keep_prob >= 0 and keep_prob <= 1, "Keep Prob = {} is not within [0, 1]".format(keep_prob)

    def forward(self, feat, is_training=True, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        kept = None
        output = None
        #############################################################################
        # TODO: Implement the forward pass of Dropout.                              #
        # Remember if the keep_prob = 0, there is no dropout.                       #
        # Use self.rng to generate random numbers.                                  #
        # During training, need to scale values with (1 / keep_prob).               #
        # Store the mask in the variable kept provided above.                       #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        if is_training and self.keep_prob:
            # if training and keep_prob > 0, note: when keep_prob = 1, there's identical mapping (no dropout)
            kept = self.rng.binomial(size=feat.shape, n=1, p=self.keep_prob) * (1.0 / self.keep_prob) # scale values
        else:
            # if testing or keep_prob == 0, note: keep_prob = 0 is the same as not use dropout
            kept = np.ones(feat.shape) # when self.keep_prob = 0 or during testing, keep the same
        output = feat * kept
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.kept = kept
        self.is_training = is_training
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        dfeat = None
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        #############################################################################
        # TODO: Implement the backward pass of Dropout                              #
        # Select gradients only from selected activations.                          #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        dfeat = self.kept * dprev
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.is_training = False
        self.meta = None
        return dfeat


class cross_entropy(object):
    def __init__(self, size_average=True):
        """
        - size_average: if dividing by the batch size or not
        - logit: intermediate variables to store the scores
        - label: Ground truth label for classification task
        """
        self.size_average = size_average
        self.logit = None
        self.label = None

    def forward(self, feat, label):
        logit = softmax(feat)
        loss = None
        #############################################################################
        # TODO: Implement the forward pass of an CE Loss                            #
        # Store the loss in the variable loss provided above.                       #
        #############################################################################
        batch_size = feat.shape[0]
        # extract the predicted probabilities for the ground truth labels
        one_hot = np.zeros_like(logit)
        one_hot[np.arange(batch_size), label] = 1.0 # wrong
        # compute the negative log-likelihood
        log_prob = -np.log(logit + 1e-20) * one_hot
        loss = np.sum(log_prob) / batch_size if self.size_average else np.sum(log_prob)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.logit = logit
        self.label = label
        return loss

    def backward(self):
        logit = self.logit
        label = self.label
        if logit is None:
            raise ValueError("No forward function called before for this module!")
        dlogit = None
        #############################################################################
        # TODO: Implement the backward pass of an CE Loss                           #
        # Store the output gradients in the variable dlogit provided above.         #
        #############################################################################
        batch_size = logit.shape[0]
        # one-hot encoding for ground truth labels
        one_hot = np.zeros_like(logit)
        one_hot[np.arange(batch_size), label] = 1 
        # gradient of the loss w.r.t. the logit (softmax input): logit - one_hot
        dlogit = (logit - one_hot) / batch_size if self.size_average else (logit - one_hot)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.logit = None
        self.label = None
        return dlogit


def softmax(feat):
    scores = None
    #############################################################################
    # TODO: Implement the forward pass of a softmax function                    #
    # Return softmax values over the last dimension of feat.                    #
    #############################################################################
    z = feat 
    num = np.exp(z - np.max(feat, axis=-1, keepdims=True)) # subtract maximum value from each element to avoid numerical instability
    den = np.sum(num, axis=-1, keepdims=True) # sum(e^z)
    scores = num / den
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return scores

def reset_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
