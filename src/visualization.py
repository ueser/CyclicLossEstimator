import pdb, traceback, sys #
from matplotlib import pylab as pl
import numpy as np
import h5py
import os, io, sys
import pdb
from math import sqrt
import tensorflow as tf
from tqdm import tqdm as tq
import cPickle as pickle
import tensorflow as tf
import matplotlib


### ###
## The following functions are adapted from https://github.com/kundajelab/deeplift :
# plot_weights_given_ax, plot_weights, plot_a, plot_c, plot_t, plot_g
#############################


################################################################################
# Main
################################################################################
def main():
    flags = tf.app.flags
    flags.DEFINE_string('runName', 'experiment', 'Running name.')
    flags.DEFINE_string('resultsDir', '../results', 'Directory for results data')
    flags.DEFINE_boolean('makeGif', True, 'Make gif from png files')
    flags.DEFINE_boolean('makePng', False, 'Make png from saved prediction pickles')
    flags.DEFINE_string('vizType', 'dnaseq', 'data type to be vizualized')
    flags.DEFINE_integer('startFrom', 0, 'Start from the iteration number.')
    FLAGS = flags.FLAGS
    save_dir = os.path.join(FLAGS.resultsDir, FLAGS.runName)

    if FLAGS.makePng:
        pckl_files = [fname for fname in os.listdir(save_dir) if 'pred_viz' in fname]
        orig_file = [fname for fname in os.listdir(save_dir) if 'originals.pck' in fname]
        pred_dict = pickle.load(open(os.path.join(save_dir, pckl_files[0]), 'r'))
        print('pred_dict.keys() = ' + str(pred_dict.keys()))
        if ('dna_before_softmax_blah' in pred_dict.keys()): # EDIT
            qq=0
            for f_ in tq(pckl_files):
                pred_dict = pickle.load(open(os.path.join(save_dir, f_),'r'))
                iter_no  = int(f_.split('.')[0].split('_')[-1])
                if iter_no < FLAGS.startFrom:
                    continue
                qq+=1
                # print('\nplotting {} of {}'.format(qq, len(pckl_files)))
                weights = pred_dict['dna_before_softmax']
                pred_vec = pred_dict['prediction']
                visualize_dna(weights, pred_vec,
                          name='iteration_{}'.format(iter_no),
                          save_dir=save_dir, verbose=False)

        elif FLAGS.vizType == 'tssseq':
            qq = 0
            orig_output = pickle.load(open(os.path.join(save_dir, orig_file[0]), 'r'))
            strand = 'Single'
            for f_ in tq(pckl_files):
                pred_dict = pickle.load(open(os.path.join(save_dir, f_), 'r'))
                iter_no = int(f_.split('.')[0].split('_')[-1])
                if iter_no < FLAGS.startFrom:
                    continue
                qq += 1
                # print('\nplotting {} of {}'.format(qq, len(pckl_files)))

                if (qq==1) and (pred_dict.values()[0].shape[1]==2*orig_output.values()[0].shape[2]):
                    strand = 'Double'
                plot_prediction(pred_dict, orig_output,
                            name='iteration_{}'.format(iter_no),
                            save_dir=save_dir,
                            strand=strand,
                            title=iter_no)

        else:
            raise NotImplementedError

    if FLAGS.makeGif:
        print('Making gif animation ... ')
        import imageio
        images = []
        png_files = [fname for fname in os.listdir(save_dir) if '.png' in fname]
        sorted_idx = np.argsort([int(f_.split('.')[0].split('_')[-1]) for f_ in png_files])

        for ix in tq(sorted_idx):
            filename = os.path.join(save_dir,png_files[ix])
            images.append(imageio.imread(filename))
        imageio.mimsave(os.path.join(save_dir,'prediction_viz.gif'), images, duration = 0.25)


################################################################################
# Auxilary Functions
################################################################################


def plot_prediction(pred_vec, orig_vec = None, save_dir = '../results/', name = 'profile_prediction', strand = 'Single', title = 'profile', verbose = True):
    save_dir = os.path.join(save_dir, 'visualization')
    if not tf.gfile.Exists(save_dir):
        tf.gfile.MakeDirs(save_dir)
    pl.ioff()
    if len(pred_vec) == 1:
        if verbose:
            print('\nPlotting predicted {} data'.format(pred_vec.keys()[0]))
        fig, axarr = pl.subplots(pred_vec.values()[0].shape[0])
        if strand == 'Double':
            to_size = pred_vec.values()[0].shape[1] / 2
            for ix in range(pred_vec.values()[0].shape[0]):
                for jx, key in enumerate(pred_vec.keys()):
                    if orig_vec is not None:
                        axarr[ix].plot(orig_vec[key][ix, 0, :] / np.max(orig_vec[key][ix, :, :] + 1e-7),
                                       label = key + '_Original',
                                       color = 'g'
                                      )
                        axarr[ix].plot(-orig_vec[key][ix, 1, :] / np.max(orig_vec[key][ix, :, :] + 1e-7),
                                       color = 'g'
                                      )
                    axarr[ix].plot(pred_vec[key][ix, :to_size] / np.max(pred_vec[key][ix, :]),
                                   label = key + '_Prediction',
                                   color = 'r'
                                  )
                    axarr[ix].plot(-pred_vec[key][ix, to_size:] / np.max(pred_vec[key][ix, :]),
                                   color = 'r'
                                  )
                    axarr[ix].axis('off')
            axarr[0].set_title(pred_vec.keys()[0] + '_' + str(title))
        else:
            for ix in range(pred_vec.values()[0].shape[0]):
                for jx, key in enumerate(pred_vec.keys()):
                    if orig_vec is not None:
                        axarr[ix].plot(orig_vec[key][ix, 0, :] / np.max(orig_vec[key][ix, 0, :] + 1e-7),
                                       label = key + '_Original',
                                       color = 'g'
                                      )
                    axarr[ix].plot(pred_vec[key][ix, :] / np.max(pred_vec[key][ix, :]),
                                   label = key + '_Prediction',
                                   color = 'r'
                                  )
                    axarr[ix].axis('off')
                    axarr[0].set_title(pred_vec.keys()[0] + '_' + str(title))
    else: #
        fig, axarr = pl.subplots(pred_vec.values()[0].shape[0],len(pred_vec))
        if strand=='Double':
            to_size = pred_vec.values()[0].shape[1]/2
            for ix in range(pred_vec.values()[0].shape[0]):
                for jx, key in enumerate(pred_vec.keys()):
                    if orig_vec is not None:
                        axarr[ix, jx].plot(orig_vec[key][ix, 0, :]/np.sum(orig_vec[key][ix,:,:]+ 1e-7), label=key+'_Original', color='g')
                        axarr[ix, jx].plot(-orig_vec[key][ix, 1, :]/np.sum(orig_vec[key][ix,:,:]+ 1e-7), color='g')
                    axarr[ix, jx].plot(pred_vec[key][ix, :to_size], label=key+'_Prediction', color='r')
                    axarr[ix, jx].plot(-pred_vec[key][ix, to_size:], color='r')
                    axarr[ix, jx].axis('off')
            axarr[0, 0].set_title(pred_vec.keys()[0])
            axarr[0, 1].set_title(pred_vec.keys()[1])


        else:
            for ix in range(pred_vec.values()[0].shape[0]):
                for jx, key in enumerate(pred_vec.keys()):
                    if orig_vec is not None:
                        axarr[ix, jx].plot(orig_vec[key][ix,0, :] / np.max(orig_vec[key][ix,0, :] + 1e-7),
                                           label=key + '_Original', color='g')
                    axarr[ix, jx].plot(pred_vec[key][ix, :]/np.max(pred_vec[key][ix, :]), label=key + '_Prediction', color='r')
                    axarr[ix, jx].axis('off')

            axarr[0, 1].set_title(pred_vec.keys()[0])
            axarr[0, 1].set_title(pred_vec.keys()[1])

    pl.savefig(os.path.join(save_dir,
                            name + '.png'),
               format = 'png'
              )
    pl.close(fig)



def put_kernels_on_grid(kernel, pad = 1):

    ''' modified from @kukuruza: https://gist.github.com/kukuruza/03731dc494603ceab0c5
    Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7


def visualize_filters():
    raise NotImplementedError



def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                     facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                     facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base],
                                              width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                              width=1.0, height=0.2 * height, facecolor=color, edgecolor=color,
                                              fill=True))

default_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}

def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if (array.shape[0] == 4 and array.shape[1] != 4):
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))

    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)
    ax.axis('off')



def plot_weights(array,
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={},
                 ax=[]):
    # fig = plt.figure(figsize=(20,2))
    # ax = fig.add_subplot(111)
    plot_weights_given_ax(ax=ax, array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight)

def visualize_dna(weights, pred_vec, save_dir = '../results/', name = 'dna_prediction', verbose = True, viz_mode = 'energy', orig = False):
    save_dir = os.path.join(save_dir, 'dnaseq_viz')
    if not tf.gfile.Exists(save_dir):
        tf.gfile.MakeDirs(save_dir)
    pl.ioff()
    fig = pl.figure(figsize = (20, 20))
    if verbose:
        print('\nPlotting predicted DNA sequence')
    for ix in tq(range(pred_vec.shape[0])):
        ax = fig.add_subplot(pred_vec.shape[0], 1, ix + 1)
        H = abs((.25 * np.log2(.25 + 1e-7) - pred_vec[ix, :, :, 0] * np.log2(pred_vec[ix, :, :, 0] + 1e-7)).sum(axis = 0))
        H = np.tile(H, 4).reshape(4, pred_vec.shape[2], 1)
        if viz_mode == 'energy':
            multiplier = weights[ix]
        elif viz_mode == 'weblogo':
            multiplier = pred_vec[ix, :, :, :]
            if orig:
                H = abs((.25 * np.log2(.25 + 1e-7) - weights[ix, :, :, 0] * np.log2(weights[ix, :, :, 0] + 1e-7)).sum(axis = 0))
                H = np.tile(H, 4).reshape(4, weights.shape[2], 1)
        plot_weights(multiplier * H,
                     height_padding_factor = 0.2,
                     length_padding = 1.0,
                     colors = default_colors,
                     subticks_frequency = pred_vec.shape[2]/2,
                     plot_funcs = default_plot_funcs,
                     highlight = {},
                     ax = ax
                    )
    plt.title(name)
    pl.savefig(os.path.join(save_dir, name + '.png'), format = 'png')
    pl.close(fig)

if __name__=='__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
