import argparse
import itertools
import functools
import numpy as np
import pandas as pd
import noscope
from noscope import np_utils

def to_test_train(avg_fname, all_frames, bbox_dict, train_ratio=0.8):
    print all_frames.shape

    frames = sorted(bbox_dict.keys())

    X = all_frames
    mean = np.mean(X, axis=0)
    np.save(avg_fname, mean)

    X = X[frames]
    Y = np.array(map(lambda x: bbox_dict[x], frames))

    p = np.random.permutation(len(X))
    p = p[0 : len(p) // 4]
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

    return X_train, X_test, Y_train, Y_test


def get_bbox(csv_fname, base_name, limit=None, start=0, labels=['person']):
    df = pd.read_csv(csv_fname)
    df = df[df['frame'] >= start]
    df = df[df['frame'] < start + limit]
    df['frame'] -= start
    df = df[df['object_name'].isin(labels)]
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
    with open(base_name + '-norm.txt', 'w') as f:
        f.write(str(norm))
    for frame in d:
        row = d[frame][2:]
        d[frame] = tuple([0, 1] + map(lambda i: row[i] / norm[i], range(len(norm))))
    for frame in xrange(limit):
        if frame not in d:
            d[frame] = (1, 0, 0, 0, 0, 0)
    return d

def get_bbox_data(csv_fname, video_fname, avg_fname, base_name,
                  num_frames=None, start_frame=0,
                  OBJECTS=['person'], resol=(50, 50),
                  center=True, dtype='float32'):
    print '\tParsing %s, extracting %s' % (csv_fname, str(OBJECTS))
    bbox_dict = get_bbox(csv_fname, base_name, limit=num_frames, labels=OBJECTS, start=start_frame)
    print '\tRetrieving all frames from %s' % video_fname
    all_frames = noscope.VideoUtils.get_all_frames(
            num_frames, video_fname, scale=resol, start=start_frame)
    print '\tSplitting data into training and test sets'
    X_train, X_test, Y_train, Y_test = to_test_train(avg_fname, all_frames, bbox_dict)

    print 'train ex: %d, test ex: %d' % (len(X_train), len(X_test))
    print 'shape of image: ' + str(X_train[0].shape)

    data = (X_train, Y_train, X_test, Y_test)
    return data

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
    data = get_bbox_data(
            args.csv_in, args.video_in, args.avg_fname, args.base_name,
            num_frames=args.num_frames, start_frame=args.start_frame,
            OBJECTS=objects,
            resol=(50, 50))
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
