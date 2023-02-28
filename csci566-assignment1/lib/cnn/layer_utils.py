from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
                self.params[n] = v
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
                # Update the gradients dictionary with the regularized gradients
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

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        batch_size, input_height, input_width, input_channels = input_size
        output_shape[0] = batch_size # batch
        output_shape[1] = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1 # output_hight
        output_shape[2] = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1 # output_width
        output_shape[3] = self.number_filters # depth(output_channels)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single convolutional layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        # zero padding
        img_padded = img if self.padding == 0 else \
                    np.pad(img, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                           'constant', constant_values=(0, 0),) 
        # initial output
        output = np.zeros(output_shape)
 
        for h in range(output_height):
            top, bottom = h * self.stride, h * self.stride + self.kernel_size
            for w in range(output_width):
                left, right = w * self.stride, w * self.stride + self.kernel_size
                # get the img crop window:
                img_crop = img_padded[:, top:bottom, left:right, :]
                img_crop = np.expand_dims(img_crop, -1) # (1, 5, 5, 1) -> (1, 5, 5, 1, 1)
                # convolution:
                # w: (5, 5, 1, 2), b: (2, ), img: (1, 5, 5, 1, 1)
                # output: (1, 4, 4, 2)
                output[:, h, w, :] = np.sum(self.params[self.w_name] * img_crop, axis=(1, 2, 3)) + self.params[self.b_name]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        # zero padding
        img_padded = img if self.padding == 0 else \
                    np.pad(img, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                           'constant', constant_values=(0, 0))
        # initial grad
        _, output_height, output_width, _ = self.get_output_size(img.shape)
        dw = np.zeros_like(self.params[self.w_name]) # (4, 4, 3, 12)
        db = np.zeros_like(self.params[self.b_name]) # (12,)
        dimg = np.zeros_like(img_padded) #  (15, 10, 10, 3)

        for h in range(output_height):
            top, bottom = h * self.stride, h * self.stride + self.kernel_size
            for w in range(output_width):
                left, right = w * self.stride, w * self.stride + self.kernel_size
                # get the img crop window:
                img_crop = img_padded[:, top:bottom, left:right, :]
                img_crop = np.expand_dims(img_crop, -1)  # (15, 4, 4, 3, 1)
                # output = w * input + b
                # dL/doutput = dL/dconvout * dconvout/doutput, here * means conv. dprev: (15, 4, 4, 12)
                # dL/dw = dL/doutput * doutput/dw = sum conv(input, dL/doutput) = = sum conv(input, dprev)
                dw += np.sum(img_crop * np.expand_dims(dprev[:, h, w, :], axis=(1, 2, 3)), axis=0)
                # dL/db = dL/doutput = sum(dprev)
                db += np.sum(dprev[:, h, w, :], axis=0)
                # dL/dinput = dL/doutput * doutput/dinput = sum conv(dL/doutput, w) = sum conv(dprev, w)
                dimg[:, top:bottom, left:right, :] += np.sum(self.params[self.w_name] * np.expand_dims(dprev[:, h, w, :], axis=(1, 2, 3)),
                                                             axis=-1)
        self.grads[self.w_name] = dw
        self.grads[self.b_name] = db
        if self.padding != 0:
            # unpadding
            dimg = dimg[:, self.padding:-self.padding, self.padding:-self.padding, :]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
        
        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        batch_size, input_height, input_width, input_channels = img.shape
        # calculate output shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        output_channels = input_channels
        output_shape = (batch_size, output_height, output_width, output_channels)
        # initial output
        output = np.zeros(output_shape)

        for h in range(output_height):
            top, bottom = h * self.stride, h * self.stride + self.pool_size
            for w in range(output_width):
                left, right = w * self.stride, w * self.stride + self.pool_size
                # get the img crop window:
                img_crop = img[:, top:bottom, left:right, :]
                # maxpool:
                output[:, h, w, :] = np.max(img_crop, axis=(1, 2)) # to (w, h)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img

        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        batch_size, h_out, w_out, output_channels = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        # initial dimg
        dimg = np.zeros_like(img)

        for h in range(h_out):
            top, bottom = h * self.stride, h * self.stride + h_pool
            for w in range(w_out):
                left, right = w * self.stride, w * self.stride + w_pool
                # get the img crop window:
                img_crop = img[:, top:bottom, left:right, :]
                # get the corresponding mask
                mask = (img_crop == np.max(img_crop, axis=(1, 2)).reshape((batch_size, 1, 1, output_channels)))
                dimg[:, top:bottom, left:right, :] += mask * dprev[:, h, w, :].reshape((batch_size, 1, 1, output_channels))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
    