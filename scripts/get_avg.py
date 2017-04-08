#! /usr/bin/env python

import itertools
import argparse
import noscope
import numpy as np
from keras.utils import np_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV input filename')
    parser.add_argument('--video_in', required=True, help='Video input filename')
    parser.add_argument('--objects', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--num_frames', required=True, type=int, help='Number of frames')
    parser.add_argument('--txt_out', required=True, help='Output')
    args = parser.parse_args()

    objects = args.objects.split(',')
    # for now, we only care about one object, since
    # we're only focusing on the binary task
    assert len(objects) == 1

    print 'Preparing data....'
    data, nb_classes = noscope.DataUtils.get_data(
            args.csv_in, args.video_in,
            binary=True,
            num_frames=args.num_frames,
            OBJECTS=objects,
            regression=False,
            resol=(50, 50),
            center=False)
    X_train, Y_train, X_test, Y_test = data

    X_avg = np.mean(X_train, axis=0)
    with open(args.txt_out, 'w') as f:
        for b in np.nditer(X_avg):
            f.write(str(b) + '\n')


if __name__ == '__main__':
    main()
