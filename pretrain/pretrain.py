#! /usr/bin/env python

import itertools
import argparse
import os
import noscope
import cv2
import numpy as np
from noscope import np_utils


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        d = cPickle.load(fo)
    return d

def load_all():
    train_labels = []
    train = None
    for i in xrange(1, 6):
        d = unpickle('cifar-10-batches-py/data_batch_%d' % i)
        train_labels += d['labels']
        if train is None:
            train = d['data']
        else:
            train = np.concatenate((train, d['data']))

    d = unpickle('cifar-10-batches-py/test_batch')
    val_labels = d['labels']
    val = d['data']

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)


    def transform(images, RESOL=50):
        images = images.reshape((images.shape[0], 32, 32, 3))
        ret = np.zeros((images.shape[0], RESOL, RESOL, 3), dtype=images.dtype)
        resol = (RESOL, RESOL)
        for i in xrange(len(images)):
            ret[i, :] = cv2.resize(images[i, :], resol, interpolation=cv2.INTER_CUBIC)
        ret = ret.astype('float32')
        ret /= 255
        ret -= 0.5
        ret *= 2
        return ret

    train = transform(train)
    val = transform(val)

    X_train = train
    X_test = val
    Y_train = np_utils.to_categorical(train_labels, num_classes=10)
    Y_test = np_utils.to_categorical(val_labels, num_classes=10)

    return X_train, Y_train, X_test, Y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--base_name', required=True, help='Base output name')
    # Regression or not
    parser.add_argument('--regression', dest='regression', action='store_true')
    parser.add_argument('--no-regression', dest='regression', action='store_false')
    parser.set_defaults(regression=False)
    # Binary or not (for classification)
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.add_argument('--no-binary', dest='binary', action='store_false')
    parser.set_defaults(binary=True)
    args = parser.parse_args()

    def check_args(args):
        if args.regression:
            if args.binary:
                print 'WARNING: Setting args.binary to False'
                args.binary = False
        else:
            # Check here?
            pass
    check_args(args)

    X_train, Y_train, X_test, Y_test = load_all()
    data = (X_train, Y_train, X_test, Y_test)
    nb_classes = 10
    nb_epoch = 6

    base_fname = args.base_name
    output_dir = args.output_dir
    model_name = 'convnet'
    params = list(itertools.product(
            *[[X_train.shape[1:]], [nb_classes],
              [0], [32, 64, 128], [1, 2, 3],
              [1]]))
    for param in params:
        param_base_fname = base_fname + '_' + model_name + '_' + '_'.join(map(str, param[2:]))
        model_fname = os.path.join(
                output_dir, param_base_fname + '.h5')

        model = noscope.Models.generate_conv_net(*param, regression=False)
        model.fit(X_train, Y_train,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True)
        model.save(model_fname)

    '''# Generally regression requires more iterations to converge.
    # Or so sleep deprived DK thinks
    nb_epoch = 5 + 5 * args.regression
    print 'Trying VGG-style nets....'
    # CIFAR10 based architectures
    noscope.Models.try_params(
            noscope.Models.generate_conv_net,
            list(itertools.product(
                    *[[X_train.shape[1:]], [nb_classes],
                      [0], [32, 64, 128], [1, 2, 3],
                      [1]])),
            data,
            args.output_dir,
            args.base_name,
            'convnet',
            'cifar10',
            regression=args.regression,
            nb_epoch=nb_epoch)'''

if __name__ == '__main__':
    main()
