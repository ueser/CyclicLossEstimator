from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

###running on a shared cluster ###
import sys
sys.path.append('/home/ue4/tfvenv/lib/python2.7/site-packages/')
##################################


import pdb, traceback, sys # EDIT
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Lambda, Conv2D, concatenate, Reshape, AveragePooling2D, Flatten, BatchNormalization, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras.objectives import kullback_leibler_divergence
import json, six, copy, os


#######################
# Auxiliary functions #
#######################


def multi_softmax(target, axis=1, name=None):
    """Takes a tensor and returns the softmax applied to a particular axis
    """
    with tf.name_scope(name):
        mx = tf.reduce_max(target, axis, keep_dims=True)
        Q = tf.exp(target-mx)
        Z = tf.reduce_sum(Q, axis, keep_dims=True)
    return Q/Z


def KL_divergence(P_, Q_):
    """Takes Tensorflow tensors and returns the KL divergence for each row i.e. D(P_, Q_)
    """
    return tf.reduce_sum(tf.multiply(K.clip(P_, K.epsilon(), 1),
                                     tf.subtract(tf.log(K.clip(P_, K.epsilon(), 1)),
                                                 tf.log(K.clip(Q_, K.epsilon(), 1)))), 1)


def transform_track(track_data_placeholder, option='pdf'):
    """Converts input placeholder tensor to probability distribution function
        :param track_data_placeholder:
        :param option: pdf: converts every entry to a pdf
        :              categorical: discretisizes the continuous input (To be implemented)
        :              standardize: zero mean, unit variance
        :return:
    """
    if option == 'pdf':
        output_tensor = tf.reshape(track_data_placeholder,
                                   [-1, (track_data_placeholder.get_shape()[1] * track_data_placeholder.get_shape()[
                                       2]).value]) + K.epsilon()
        output_tensor = tf.divide(output_tensor, tf.reduce_sum(output_tensor, 1, keep_dims=True))
    # NOT completed yet
    elif option == 'standardize':
        raise NotImplementedError
        from scipy import stats
        output_tensor = stats.zscore(output_tensor, axis=1)
    return output_tensor

def byteify(json_out):
    '''
    Recursively reads in .json to string conversion into python dictionary format
    '''
    if isinstance(json_out, dict):
        return {byteify(key): byteify(value)
                for key, value in six.iteritems(json_out)}
    elif isinstance(json_out, list):
        return [byteify(element) for element in json_out]
    elif isinstance(json_out, unicode):
        return json_out.encode('utf-8')
    else:
        return json_out

class ArchitectureParsingError(Exception):
    pass

class ConfigurationParsingError(Exception):
    pass

#################
# Model Classes #
#################


class BaseTrackContainer(object):
    def __init__(self, track_name):
        pass
    def initialize(self):
        """Initialize the model
        """
        if self.sess is None:
            self.sess = tf.Session()  # Launch the session
        # Initializing the tensorflow variables
        init = tf.global_variables_initializer()  # std out recommended this instead
        self.sess.run(init)
        print('Session initialized.')

    def load(self):
        pass

    def forward(self):
        pass

    def freeze(self):
        pass

    def save(self):
        pass

class ConvolutionalContainer(BaseTrackContainer):

    def __init__(self, track_name, architecture, batch_norm=False, input=None):
        BaseTrackContainer.__init__(self, track_name)
        self.track_name = track_name
        self.architecture = architecture
        self.batch_norm=batch_norm

        if input is None:
            self.input = tf.placeholder(tf.float32, [None, self.architecture['Modules'][self.track_name]["input_height"],
                                                       self.architecture['Modules'][self.track_name]["input_width"], 1],
                                          name=self.track_name+'_input')
        else:
            self.input = input

        self._build()

    def _build(self):
        # scope reusing is for cost estimation  for future implementations
        with tf.variable_scope(self.track_name, reuse=True):
            net = Conv2D(self.architecture['Modules'][self.track_name]['Layer1']['number_of_filters'],
                         [self.architecture['Modules'][self.track_name]['Layer1']['filter_height'],
                         self.architecture['Modules'][self.track_name]['Layer1']['filter_width']],
                         activation=self.architecture['Modules'][self.track_name]['Layer1']['activation'],
                         kernel_regularizer='l2',
                         padding='valid',
                         name='conv_1')(self.input)
            net = AveragePooling2D((1, self.architecture['Modules'][self.track_name]['Layer1']['pool_size']),
                                    strides=(1, self.architecture['Modules'][self.track_name]['Layer1']['pool_stride']))(net)
            if self.batch_norm:
                net = BatchNormalization()(net)
            net = Conv2D(self.architecture['Modules'][self.track_name]['Layer2']['number_of_filters'],
                                  [self.architecture['Modules'][self.track_name]['Layer2']['filter_height'],
                                   self.architecture['Modules'][self.track_name]['Layer2']['filter_width']],
                                  activation=self.architecture['Modules'][self.track_name]['Layer2']['activation'],
                                  kernel_regularizer='l2',
                                  padding='valid',
                                  name='conv_2')(net)

            net = AveragePooling2D([1, self.architecture['Modules'][self.track_name]['Layer2']['pool_size']],
                                    strides=[1, self.architecture['Modules'][self.track_name]['Layer2']['pool_stride']],
                                    padding='valid',
                                    name='AvgPool_2')(net)
            if self.batch_norm:
                net = BatchNormalization()(net)
            net = Flatten()(net)
            self.representation = Dense(self.architecture['Modules'][self.track_name]['representation_width'],
                        name='representation')(net)


def kl_loss(y_true, y_pred):
    return tf.reduce_mean(kullback_leibler_divergence(y_true, y_pred))

def per_bp_accuracy(y_true, y_pred):
    pass

def peak_detection_accuracy(y_true, y_pred, bin_size=50):
    pass


def average_peak_distance(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.argmax(y_true, dimension=1)-tf.argmax(y_pred, dimension=1)))
