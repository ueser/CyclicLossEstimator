from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pdb, traceback, sys #
import tensorflow as tf
import numpy as np
from tqdm import tqdm as tq
import six
from collections import Counter
import os
import h5py
import json
import cPickle as pickle

### FIDDLE specific tools ###
from models import *
from io_tools import *
# import visualization as viz
#############################


flags = tf.app.flags
flags.DEFINE_string('runName', 'experiment', 'Running name.')
flags.DEFINE_string('dataDir', '../data/hdf5datasets', 'Default data directory')
flags.DEFINE_string('configuration', 'configurations.json', 'configuration file [json file]')
flags.DEFINE_string('architecture', 'architecture.json', 'configuration file [json file]')
flags.DEFINE_string('restorePath', '../results/test', 'Regions to validate [bed or gff files]')
flags.DEFINE_string('visualizePrediction', 'offline', 'Prediction profiles to be plotted [online or offline] ')
flags.DEFINE_integer('maxEpoch', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batchSize', 100, 'Batch size.')
flags.DEFINE_integer('strategy', 1, 'Strategy [1: standard loss, 2: adversarial loss, 3: cyclic loss]')
flags.DEFINE_float('learningRate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')
FLAGS = flags.FLAGS

################################################################################
# main
################################################################################

def main(_):

    # read in configurations
    with open(FLAGS.configuration) as fp:
        config = byteify(json.load(fp))

    # create or recognize results directory
    FLAGS.savePath = FLAGS.resultsDir + '/' + FLAGS.runName
    print('Results will be saved in ' + str(FLAGS.savePath))
    if not tf.gfile.Exists(FLAGS.savePath):
        tf.gfile.MakeDirs(FLAGS.savePath)

    architecture = parse_parameters(config, FLAGS.architecture)
    # # save resulting modified architecture and configuration
    # json.dump(model.architecture, open(FLAGS.savePath + "/architecture.json", 'w'))
    # json.dump(model.config, open(FLAGS.savePath + "/configuration.json", 'w'))

    # input training and validation data
    train_h5_handle = h5py.File(os.path.join(FLAGS.dataDir, config['Options']['DataName'], 'train.h5'), 'r')
    validation_h5_handle = h5py.File(os.path.join(FLAGS.dataDir, config['Options']['DataName'], 'validation.h5'),
                                     'r')

    # create iterator over training data
    data = MultiModalData(train_h5_handle, batch_size=FLAGS.batchSize)
    batcher = data.batcher()
    print('Storing validation data to the memory')
    try:
        all_keys = list(set(architecture['Inputs'] + architecture['Outputs']))
        validation_data = {key: validation_h5_handle[key][:] for key in all_keys}
    except KeyError:
        print('Make sure that the configuration file contains the correct track names (keys), '
              'which should match the hdf5 keys')

    global_step = tf.Variable(0, name='globalStep', trainable=False)
    train_estimator = True
    # define neural network
    if train_estimator:
        chipseq_ph = tf.placeholder(tf.float32, [None, architecture['Modules']['chipseq']["input_height"],
                                                       architecture['Modules']['chipseq']["input_width"], 1],
                                          name='chipseq_input')
        d2c = ConvolutionalContainer('dnaseq',
                           architecture=architecture)

        with tf.variable_scope('dnaseq'):
            chipseq_before_softmax = Dense(2*architecture['Modules']['chipseq']['input_width'], name='FC_before_softmax')(d2c.representation)
            chipseq_after_softmax = tf.nn.softmax(chipseq_before_softmax, name='softmax')


        chipseq_pdf = transform_track(chipseq_ph)
        d2c_cost = kl_loss(chipseq_pdf, chipseq_after_softmax)

        d2c_trainables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dnaseq')
        d2c_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learningRate).minimize(d2c_cost,
                                                                                  global_step=global_step,
                                                                                  var_list=d2c_trainables)
        
        globalMinLoss = np.Inf
        d2c_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dnaseq'))
        sess = tf.Session()
        init = tf.global_variables_initializer()  # std out recommended this instead
        sess.run(init)
        print('Session initialized.')
        print('Training the estimator network')

        for it in range(100):
            cost = 0
            for sub_iter in tq(range(10)):
                train_batch = batcher.next()
                # pdb.set_trace()
                _, sub_cost = sess.run([d2c_optimizer, d2c_cost], feed_dict={d2c.input:train_batch['dnaseq'],
                                                                             chipseq_ph:train_batch['chipseq']})
                cost+=sub_cost
            cost/=10.

            val_cost = sess.run(d2c_cost, feed_dict={d2c.input: validation_data['dnaseq'],
                                                                         chipseq_ph:validation_data['chipseq']})
            print('Train cost: {}'.format(cost))
            print('Validation cost: {}'.format(val_cost))
            if val_cost < globalMinLoss:
                globalMinLoss = cost.copy()
                save_path = d2c_saver.save(sess, os.path.join(FLAGS.savePath, 'd2c_model.ckpt'))
                print('Model saved in file: %s' % FLAGS.savePath)

    else:
        c2d = ConvolutionalContainer('chipseq',
                                     architecture=architecture)

        with tf.variable_scope('chipseq'):
            dna_before_softmax = Dense(4*architecture['Modules']['dnaseq']['input_width'], name='FC_before_softmax')(c2d.representation)
            dna_before_softmax = tf.reshape(dna_before_softmax,
                                                     [-1, 4, architecture['Modules']['dnaseq']['input_width'], 1])
            dna_after_softmax = multi_softmax(dna_before_softmax, axis=1, name='multiSoftmax')


        d2c_est = ConvolutionalContainer('dnaseq',
                                         architecture=architecture, input=dna_after_softmax)
        with tf.variable_scope('dnaseq'):
            chipseq_before_softmax = Dense(2*architecture['Modules']['chipseq']['input_width'], name='FC_before_softmax')(d2c_est.representation)
            chipseq_after_softmax = tf.nn.softmax(chipseq_before_softmax, name='softmax')

        c2d_cost = kl_loss(c2d.input, chipseq_after_softmax)

        c2d_trainables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='chipseq')
        c2d_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learningRate).minimize(c2d_cost,
                                                                                          global_step=global_step,
                                                                                          var_list=c2d_trainables)
        sess = tf.Session()
        init = tf.global_variables_initializer()  # std out recommended this instead
        sess.run(init)
        d2c_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dnaseq'))
        d2c_saver.restore(sess, os.path.join(model_path, 'd2c_model.ckpt'))

        c2d_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='chipseq'))
        print('Session initialized.')
        print('Training the predictor network')
        globalMinLoss=np.Inf
        for it in range(100):
            for sub_iter in tq(range(10)):
                train_batch = batcher.next()
                _, sub_cost = sess.run([c2d_optimizer, c2d_cost], feed_dict={c2d.input:train_batch['chipseq']})
                cost += sub_cost
            cost /= 10.
            val_cost = sess.run(c2d_cost, feed_dict={c2d.input: validation_data['chipseq']})

            print('Train cost: {}'.format(cost))
            print('Validation cost: {}'.format(val_cost))
            if val_cost < globalMinLoss:
                globalMinLoss = cost.copy()
                save_path = c2d_saver.save(d2c_saver, os.path.join(FLAGS.savePath, 'c2d_model.ckpt'))
                print('Model saved in file: %s' % FLAGS.savePath)

   
def parse_parameters(config, architecture_path='architecture.json'):
    #####################################
    # Architecture and model definition #
    #####################################

    # Read in the architecture parameters, defined by the user
    print('Constructing architecture and model definition')
    with open(architecture_path) as fp:
        architecture_template = byteify(json.load(fp))

    # if the architecture.json is read from pre-trained project directory, then just copy and continue with that
    if 'Inputs' in architecture_template.keys():
        architecture = architecture_template
    else:
        architecture = {'Modules': {}, 'Scaffold': {}}
        architecture['Scaffold'] = architecture_template['Scaffold']
        architecture['Inputs'] = config['Options']['Inputs']
        architecture['Outputs'] = config['Options']['Outputs']

        for key in architecture['Inputs']+architecture['Outputs']:
            architecture['Modules'][key] = copy.deepcopy(architecture_template['Modules'])
            architecture['Modules'][key]['input_height'] = config['Tracks'][key]['input_height']
            architecture['Modules'][key]['Layer1']['filter_height'] = config['Tracks'][key]['input_height']
            # Parameters customized for specific tracks are read from architecture.json and updates the arch. dict.
            if key in architecture_template.keys():
                for key_key in architecture_template[key].keys():
                    sub_val = architecture_template[key][key_key]
                    if type(sub_val) == dict:
                        for key_key_key in sub_val:
                            architecture['Modules'][key][key_key][key_key_key] = sub_val[key_key_key]
                    else:
                        architecture['Modules'][key][key_key] = sub_val
    return architecture

if __name__ == '__main__':
    try:
        tf.app.run()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
