"""
Use this script to visualize the output of a trained speech-model.
Usage: python visualize.py /path/to/audio /path/to/training/json.json \
            /path/to/model

Code modified by Ravi
"""

from __future__ import absolute_import, division, print_function
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time 

from data_generator import DataGenerator
from model import compile_output_fn
from utils import argmax_decode, load_model

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def plot_ctc(prediction, f, target_dir):
    """
    args - prediction -> the ctc loss as numpy array
            f -> The name of the file, this will be the name of the numpy file
            target_dir
    """ 
    softmax_img_file = os.path.join(target_dir,"{}.png".format(f[:-4]))
    print ("As Image: {}".format(softmax_img_file))
    sm = softmax(prediction.T)
    sm = np.vstack((sm[0], sm[2], sm[3:][::-1]))
    fig, ax = plt.subplots()
    ax.pcolor(sm, cmap=plt.cm.Greys_r)
    column_labels = [chr(i) for i in range(97, 97 + 26)] + ['space', 'blank']
    ax.set_yticks(np.arange(sm.shape[0]) + 0.5, minor=False)
    ax.set_yticklabels(column_labels[::-1], minor=False)
    plt.savefig(softmax_img_file)


def save_ctc_loss(prediction, f, target_dir):
    """
    args - prediction -> the ctc loss as numpy array
            f -> The name of the file, this will be the name of the output file
    """
    softmax_file = os.path.join(target_dir,"{}.npy".format(f[:-4]))
    print ("Saving network output to: {}".format(softmax_file))
    np.save(softmax_file, prediction)

def main(test_dir, train_desc_file, load_dir, weights_file= None, target_dir = None):
    print ("Loading model")
    t1 = time.time()
    model = load_model(load_dir, weights_file)
    datagen = DataGenerator()
    datagen.load_train_data(train_desc_file)
    datagen.fit_train(100)
    t2 = time.time()
    print ("Compiling test function...")
    test_fn = compile_output_fn(model)
    delt = t2 - t1
    print("The time taken to load a model and compile test function is : {} sec".format(delt))
    for f in os.listdir(test_dir):
        if f.endswith(".wav"):
            t1 = time.time()
            inputs = [datagen.normalize(datagen.featurize(os.path.join(test_dir, f)))]
            prediction = np.squeeze(test_fn([inputs, True]))
            t2 = time.time()
            delt = t2 - t1
            ## The prediction numpy array can be redirected to a prefix beam search decoder
            print ("Prediction for {0}: {1}.".format(f, argmax_decode(prediction)))
            print("It took {} seconds".format(delt))
            #plot_ctc(prediction, f, target_dir)
            #save_ctc_loss(prediction, f, t.arget_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str,
                        help='Path to the directory containing the audio')
    parser.add_argument('train_desc_file', type=str,
                        help='Path to the training JSON-line file. This will '
                             'be used to extract feature means/variance')
    parser.add_argument('load_dir', type=str,
                        help='Directory where a trained model is stored.')
    parser.add_argument('--weights_file', type=str, default=None,
                        help='Path to a model weights file')
    parser.add_argument('--targets_dir', type=str, default=None,
                        help='Directory to store the exact ctc probablities, not needed if you are not storing the ctc probabilities')
    args = parser.parse_args()

    main(args.test_dir, args.train_desc_file, args.load_dir, args.weights_file)
