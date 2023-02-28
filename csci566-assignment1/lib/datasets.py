from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from scipy.io import loadmat

def CIFAR100(data_dir):
    """ Load CIFAR100 data """
    train_data_dict = loadmat(os.path.join(data_dir, 'train.mat'))
    test_data_dict = loadmat(os.path.join(data_dir, 'test.mat'))
    meta_dict = loadmat(os.path.join(data_dir, 'meta.mat'))
    
    data_train = train_data_dict['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels_train = train_data_dict['coarse_labels'].flatten()
    data_test = test_data_dict['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels_test = test_data_dict['coarse_labels'].flatten()
    label_names = [meta_dict['coarse_label_names'][i][0][0] for i in range(20)]

    return data_train, labels_train, data_test, labels_test, label_names

def CIFAR100_data(data_dir, num_training=40000, num_validation=10000):
    # Load the raw SVHN data
    data_train, labels_train, data_test, labels_test, label_names = CIFAR100(data_dir)

    # convert to float and rescale
    data_train = data_train.astype(np.float32) / 255
    data_test = data_test.astype(np.float32) / 255
    # convert labels to zero-indexed
    #labels_train -= 1
    #labels_test -= 1

    # Subsample the data
    data_val = data_train[range(num_training, num_training+num_validation)]
    labels_val = labels_train[range(num_training, num_training+num_validation)]
    data_train = data_train[range(num_training)]
    labels_train = labels_train[range(num_training)]

    # Normalize the data: subtract the images mean
    mean_image = np.mean(data_train, axis=(0,1,2), keepdims=True)
    std_image = np.std(data_train, axis=(0,1,2), keepdims=True)
    data_train = (data_train-mean_image)/std_image
    data_val = (data_val-mean_image)/std_image
    data_test = (data_test-mean_image)/std_image

    # return a data dict
    return {
      'data_train': data_train, 'labels_train': labels_train,
      'data_val': data_val, 'labels_val': labels_val,
      'data_test': data_test, 'labels_test': labels_test,
      'label_names': label_names,
      'mean_image': mean_image, 'std_image': std_image
    }
