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
import visualization as viz
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

    # define neural network

    dna2chipseq = ConvolutionalContainer('dnaseq',
                       architecture_path=FLAGS.architecture)

    predictor = ConvolutionalContainer('chipseq',
                       architecture_path=FLAGS.architecture)

    with tf.variable_scope('chipseq'):
        dna_before_softmax = Dense(4*self.architecture['Modules']['dnaseq']['input_width'], name='FC_before_softmax')(predictor.representation)
        dna_before_softmax = tf.reshape(dna_before_softmax,
                                                 [-1, 4, self.architecture['Modules']['dnaseq']['input_width'], 1])
        dna_after_softmax = multi_softmax(dna_before_softmax, axis=1, name='multiSoftmax')


    estimator = ConvolutionalContainer('dnaseq',
                                       architecture_path=FLAGS.architecture, input=dna_after_softmax)



    # save resulting modified architecture and configuration
    json.dump(model.architecture, open(FLAGS.savePath + "/architecture.json", 'w'))
    json.dump(model.config, open(FLAGS.savePath + "/configuration.json", 'w'))

    # input training and validation data
    train_h5_handle  = h5py.File(os.path.join(FLAGS.dataDir, config['Options']['DataName'], 'train.h5'),'r')
    validation_h5_handle  = h5py.File(os.path.join(FLAGS.dataDir, config['Options']['DataName'], 'validation.h5'),'r')

    # create iterator over training data
    data = MultiModalData(train_h5_handle, batch_size=FLAGS.batchSize)
    batcher = data.batcher()
    print('Storing validation data to the memory')
    try:
        all_keys = list(set(model.architecture['Inputs'] + model.architecture['Outputs']))
        validation_data = {key: validation_h5_handle[key][:] for key in all_keys}
    except KeyError:
        print('Make sure that the configuration file contains the correct track names (keys), '
              'which should match the hdf5 keys')


    ####################
    # Launch the graph #
    ####################
    model.initialize()
    model.create_monitor_variables(show_filters=False)
    model.saver()


    with open((FLAGS.savePath + "/" + "train.txt"), "w") as train_file:
        train_file.write('Loss\n')
    with open((FLAGS.savePath + "/" + "validation.txt"), "w") as validation_file:
        validation_file.write('Loss\n')


    ## select some (10) good quality signals for prediction overlay during training
    tfval = np.ones((validation_data[key].shape[0]), dtype=bool)
    for key in model.architecture['Outputs']:
        idx = np.argsort(validation_data[key].reshape(validation_data[key].shape[0],-1).sum(axis=1))


    idx = idx[-5:]
    input_for_prediction = {key: validation_data[key][idx] for key in model.architecture['Inputs']}
    orig_output = {key: validation_data[key][idx] for key in model.architecture['Outputs']}

    pickle.dump(orig_output, open(os.path.join(FLAGS.resultsDir, FLAGS.runName, 'originals.pck'), "wb"))

    ######## TRAIN #########
    globalMinLoss = np.inf
    step = 0
    train_size = train_h5_handle.values()[0].shape[0]

    print('Pre-train validation run:')
    # return_dict = model.validate(validation_data, accuracy=True)
    # print("Pre-train validation loss: " + str(return_dict['cost']))
    # print("Pre-train validation accuracy (%): " + str(
    #     100. * return_dict['accuracy_' + key] / validation_data.values()[0].shape[0]))
    case=True
    prev = np.Inf
    totalIterations = 1000
    # FLAGS.strategy=3
    for it in range(totalIterations):


        epoch = int(it * 10 * FLAGS.batchSize/train_size)

        print('Epoch: ' + str(epoch) + ', Iterations: ' + str(it))
        print('Number of examples seen: ' + str(it * 10 * FLAGS.batchSize))


        return_dict_train = Counter({})


        t_batcher, t_trainer = 0, 0
        for iterationNo in tq(range(10)):
            with Timer() as t:
                train_batch = batcher.next()
            t_batcher += t.secs
            with Timer() as t:
                tmp = model.train(train_batch, batch_size=FLAGS.batchSize)
                train_summary = tmp['summary']
                return_dict = Counter(tmp)

            t_trainer += t.secs

            return_dict_train += return_dict


            # return_dict_train += Counter(ret2)
            step += 1
        print('Batcher time: ' + "%.3f" % t_batcher)
        print('Trainer time: ' + "%.3f" % t_trainer)
        for key in return_dict_train.keys():
            if type(key) is not str:
                return_dict_train[key] /= iterationNo
        return_dict_valid = model.validate(validation_data, accuracy=True)


        # for every 50 iteration,
        if it % 5 ==0:

            if 'dnaseq' not in model.outputs.keys():
                predicted_dict = model.predict(input_for_prediction)
                pickle.dump(predicted_dict, open(os.path.join(FLAGS.resultsDir, FLAGS.runName, 'pred_viz_{}.pck'.format(it)), "wb"))

                if FLAGS.visualizePrediction == 'online':
                    viz.plot_prediction(predicted_dict, orig_output,
                                            name='iteration_{}'.format(it),
                                            save_dir=os.path.join(FLAGS.resultsDir, FLAGS.runName),
                                            strand=model.config['Options']['Strand'])
            else:
                feed_d = {val: input_for_prediction[key] for key, val in model.inputs.items()}
                feed_d.update({val: orig_output[key] for key, val in model.outputs.items()})
                feed_d.update({model.dropout: 1.,
                               model.keep_prob_input: 1.,
                               model.inp_size: input_for_prediction.values()[0].shape[0],
                               K.learning_phase(): 0})
                weights, pred_vec = model.sess.run([model.dna_before_softmax, model.predictions['dnaseq']], feed_d)
                predicted_dict={'dna_before_softmax':weights,
                                'prediction': pred_vec}
                pickle.dump(predicted_dict, open(os.path.join(FLAGS.resultsDir, FLAGS.runName, 'pred_viz_{}.pck'.format(it)), "wb"))
                if FLAGS.visualizePrediction == 'online':
                    viz.visualize_dna(weights, pred_vec,
                                  name='iteration_{}'.format(it),
                                  save_dir=os.path.join(FLAGS.resultsDir, FLAGS.runName) )
        #
        write_to_txt(return_dict_train)
        write_to_txt(return_dict_valid, batch_size=validation_data.values()[0].shape[0], case='validation')

        model.summarize( train_summary = train_summary,
                         validation_summary=return_dict_valid['summary'],
                         step=step)

        if return_dict_valid['cost'] < globalMinLoss:
            globalMinLoss = return_dict_valid['cost']
            for track_name, saver in model.savers_dict.items():
                save_path = saver.save(model.sess, os.path.join(FLAGS.savePath, track_name+'_model.ckpt'))
            print('Model saved in file: %s' % FLAGS.savePath)

    model.sess.close()



def write_to_txt(return_dict, batch_size=FLAGS.batchSize, case='train', verbose=True):
    save_path = os.path.join(FLAGS.resultsDir,FLAGS.runName)
    line_to_write = ''
    for key, val in return_dict.items():
        if key == 'cost':
            cur_line = str(return_dict['cost'])
            line_to_write += str(return_dict[key])
        elif (key == '_') or (key == 'summary'):
            continue
        else:
            cur_line = str(100. * return_dict[key] / batch_size)
            line_to_write += '\t' + cur_line
        if verbose:
            print(case + '\t' + key + ': ' + cur_line)

    with open((save_path + "/" + case + ".txt"), "a") as fp:
        fp.write(line_to_write + '\n')


if __name__ == '__main__':
    try:
        tf.app.run()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
