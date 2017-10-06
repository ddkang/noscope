import argparse
import itertools
import functools
import numpy as np
import pandas as pd
import noscope
from noscope import np_utils

# The data is too big to actually hold,
# so the methods will return the arrays directly
class DataLoader(object):
    def __init__(self, start_frame, nb_frames, resol, labels, output_base):
        self.start_frame = start_frame
        self.nb_frames = nb_frames
        self.resol = resol
        self.labels = labels
        # Output base for how where to store metadata files
        self.output_base = output_base

    def load_labels(self, csv_fname):
        df = pd.read_csv(csv_fname)
        df = df[df['frame'] >= self.start_frame]
        df = df[df['frame'] < self.start_frame + self.nb_frames]
        df['frame'] -= self.start_frame
        df = df[df['object_name'].isin(self.labels)]
        return df

    def load_video(self, video_fname):
        # FIXME: have it dispatch based on video_fname
        return noscope.VideoUtils.get_all_frames(
                self.nb_frames, video_fname,
                start=self.start_frame, scale=self.resol)

    def split(self, X, Y, keep_fraction=0.25, train_ratio=0.8):
        mean = np.mean(X, axis=0)
        np.save(self.output_base + '.npy', mean)

        p = np.random.permutation(len(X))
        p = p[0 : int(len(p) * keep_fraction)]
        X, Y = X[p], Y[p]
        X -= mean

        def split(arr):
            # 250 -> 100, 50, 100
            ind = int(len(arr) * train_ratio)
            if ind > 50000:
                ind = len(arr) - 50000
            return arr[:ind], arr[ind:]

        X_train, X_test = split(X)
        Y_train, Y_test = split(Y)

        # return X_train, X_test, Y_train, Y_test
        return X_train, Y_train, X_test, Y_test

    def get_evaluation_data(self, csv_fname, video_fname):
        print '\tParsing %s, extracting %s' % (csv_fname, str(self.labels))
        Y = self.load_labels(csv_fname)
        print '\tRetrieving all frames from %s' % video_fname
        X = self.load_video(video_fname)
        return X, Y

    def get_train_data(self, csv_fname, video_fname, keep_fraction=0.25, train_ratio=0.8):
        X, Y = self.get_evaluation_data(csv_fname, video_fname)
        print '\tSplitting data into training and test sets'
        return self.split(X, Y, keep_fraction, train_ratio)


class DetectionLoader(DataLoader):
    def load_labels(self, csv_fname):
        df = super(DetectionLoader, self).load_labels(csv_fname)
        d = {}
        for row in df.itertuples():
            if row.frame not in d or d[row.frame].confidence < row.confidence:
                d[row.frame] = row
        norm = (0, 0, 0, 0)
        for frame in d:
            row = d[frame]
            xmin, ymin, xmax, ymax = max(0, row.xmin), max(0, row.ymin), row.xmax, row.ymax
            xcent = (xmax + xmin) / 2
            ycent = (ymax + ymin) / 2
            # d[frame] = (0, 1, max(0, row.xmin), max(0, row.ymin), row.xmax, row.ymax)
            d[frame] = (0, 1, xcent, ycent, xmax - xcent, ymax - ycent)
            norm = map(lambda i: max(norm[i], d[frame][i + 2]), range(len(norm)))
        with open(self.output_base + '-norm.txt', 'w') as f:
            f.write(str(norm))
        for frame in d:
            row = d[frame][2:]
            d[frame] = tuple([0, 1] + map(lambda i: row[i] / norm[i], range(len(norm))))
        for frame in xrange(self.nb_frames):
            if frame not in d:
                d[frame] = (1, 0, 0, 0, 0, 0)

        Y = np.zeros((self.nb_frames, 6)) # FIXME: 6
        for i in range(self.nb_frames):
            Y[i] = d[i]
        return Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV input filename')
    parser.add_argument('--video_in', required=True, help='Video input filename')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--base_name', required=True, help='Base output name')
    parser.add_argument('--objects', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--avg_fname', required=True)
    parser.add_argument('--num_frames', type=int, help='Number of frames')
    parser.add_argument('--start_frame', type=int)
    args = parser.parse_args()

    def check_args(args):
        assert args.objects is not None
    check_args(args)

    objects = args.objects.split(',')
    # for now, we only care about one object, since
    # we're only focusing on the binary task
    assert len(objects) == 1

    print 'Preparing data....'
    loader = DetectionLoader(args.start_frame, args.num_frames, (50, 50),
                             objects, args.base_name)
    data = loader.get_train_data(args.csv_in, args.video_in)
    X_train, Y_train, X_test, Y_test = data

    nb_epoch = 1
    runner = noscope.Models.DetectionModel()
    runner.try_params(
            noscope.Models.generate_conv_net,
            list(itertools.product(
                    *[[X_train.shape[1:]], [6],
                      # [1], [16, 32], [1, 2, 3, 4]])),
                      [64, 128], [16, 32], [2, 3, 4]])),
            data,
            args.output_dir,
            args.base_name,
            'convnet')

if __name__ == '__main__':
    main()
