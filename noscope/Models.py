import enum
import sklearn
import sklearn.metrics
import time
import os
import tempfile
import StatsUtils
import DataUtils
import Metrics
import numpy as np
import keras
import keras.optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import np_utils

def generate_conv_net_base(
        input_shape, nb_classes,
        nb_dense=128, nb_filters=32, nb_layers=1, lr_mult=1,
        kernel_size=(3, 3), stride=(1, 1),
        use_bn=True):
    assert nb_layers >= 1
    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size,
                     strides=stride, padding='same', input_shape=input_shape))
    if use_bn: model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (3, 3), padding='same'))
    if use_bn: model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    for i in range(1, nb_layers):
        factor = 2 ** i
        model.add(Conv2D(nb_filters * factor, (3, 3), padding='same'))
        if use_bn: model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(nb_filters * factor, (3, 3), padding='same'))
        if use_bn: model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(nb_dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    return model

# FIXME: detection is single bbox only
PredType = enum.Enum('PredType', 'BINARY REGRESSION DETECTION')
from keras import backend as K
def detection_act(t, relu=False):
    t1 = keras.activations.softmax(t[:, 0:2])
    t2 = t[:, 2:]
    if relu:
        t2 = keras.activations.relu(t2)
    return K.concatenate([t1, t2], axis=-1)

def generate_conv_net(input_shape, nb_classes,
                      nb_dense=128, nb_filters=32, nb_layers=1, lr_mult=1,
                      use_bn=True, pred_type=PredType.BINARY):

    model = generate_conv_net_base(
            input_shape, nb_classes,
            nb_dense=nb_dense, nb_filters=nb_filters, nb_layers=nb_layers,
            lr_mult=lr_mult, use_bn=use_bn)
    if pred_type == PredType.BINARY:
        model.add(Activation('softmax'))
    elif pred_type == PredType.REGRESSION:
        # clip?
        pass
    elif pred_type == PredType.DETECTION:
        model.add(Activation(detection_act))
    return model


class NoScopeModel(object):
    def __init__(self):
        pass

    def get_callbacks(self, model_fname):
        return [ModelCheckpoint(model_fname)]

    def get_optimizer(self):
        return keras.optimizers.RMSprop(lr=0.001)

    def get_loss(self):
        raise NotImplementedError

    def get_metrics(self):
        raise NotImplementedError

    def evaluate_model(self, model, X_test, Y_test, batch_size=256):
        raise NotImplementedError

    def compile_model(self, model):
        model.compile(loss=self.get_loss(), optimizer=self.get_optimizer(), metrics=self.get_metrics())

    def train_model(self, model, X_train, Y_train, batch_size=32, nb_epoch=1):
        temp_fname = tempfile.mkstemp(suffix='.hdf5', dir='/tmp/')[1]
        begin_train = time.time()
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=nb_epoch,
                  shuffle=True,
                  class_weight='auto',
                  callbacks=self.get_callbacks(temp_fname))
        train_time = time.time() - begin_train
        os.remove(temp_fname)
        return train_time

    # NOTE: assumes first two parameters are: (image_size, nb_classes)
    def try_params(self, model_gen, params, data,
                   output_dir, base_fname, model_name):
        def metrics_names(metrics):
            return sorted(metrics.keys())
        def metrics_to_list(metrics):
            return map(lambda key: metrics[key], metrics_names(metrics))

        summary_csv_fname = os.path.join(
                output_dir, base_fname + '_' + model_name + '_summary.csv')

        X_train, Y_train, X_test, Y_test = data
        to_write = []
        for param in params:
            param_base_fname = base_fname + '_' + model_name + '_' + '_'.join(map(str, param[2:]))
            model_fname = os.path.join(output_dir, param_base_fname + '.h5')
            csv_fname = os.path.join(output_dir, param_base_fname + '.csv')

            model = model_gen(*param, pred_type=self.pred_type)
            self.compile_model(model)

            train_time = self.train_model(model, X_train, Y_train)
            metrics = self.evaluate_model(model, X_test, Y_test)

            model.save(model_fname)

            to_write.append(list(param[2:]) + [train_time] + metrics_to_list(metrics))
            print param
            print train_time, metrics
            print
        print to_write
        # First two params don't need to be written out
        param_column_names = map(lambda i: 'param' + str(i), xrange(len(params[0]) - 2))
        column_names = param_column_names + ['train_time'] + metrics_names(metrics)
        DataUtils.output_csv(summary_csv_fname, to_write, column_names)


class BinaryClassificationModel(NoScopeModel):
    def __init__(self, **kwargs):
        super(BinaryClassificationModel, self).__init__(**kwargs)
        self.pred_type = Predtype.BINARY

    def get_loss(self):
        return 'categorical_crossentropy'

    def get_metrics(self):
        return ['accuracy']

    # FIXME: figure out how to deal with this + multiclass
    def evaluate_model(self, model, X_test, Y_test, batch_size=256):
        begin = time.time()
        # predicted_labels = model.predict_classes(X_test, batch_size=batch_size, verbose=0)
        proba = model.predict(X_test, batch_size=batch_size, verbose=0)
        # predicted_labels = np_utils.probas_to_classes(proba)
        end = time.time()
        test_time = end - begin

        '''confusion = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
        # Minor smoothing to prevent division by 0 errors
        TN = float(confusion[0][0]) + 1
        FN = float(confusion[1][0]) + 1
        TP = float(confusion[1][1]) + 1
        FP = float(confusion[0][1]) + 1
        metrics = {'recall': TP / (TP + FN),
                   'specificity': TN / (FP + TN),
                   'precision': TP / (TP + FP),
                   'npv':  TN / (TN + FN),
                   'fpr': FP / (FP + TN),
                   'fdr': FP / (FP + TP),
                   'fnr': FN / (FN + TP),
                   'accuracy': (TP + TN) / (TP + FP + TN + FN),
                   'f1': (2 * TP) / (2 * TP + FP + FN),
                   'test_time': test_time}'''
        metrics = Metrics.classification_metrics(proba, Y_test)
        metrics['test_time'] = test_time
        return metrics

class RegressionModel(NoScopeModel):
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)
        self.pred_type = PredType.REGRESSION

    def get_loss(self):
        return 'mean_squared_error'

    def get_metrics(self):
        return ['mean_squared_error']

    def evaluate_model(self, model, X_test, Y_test, batch_size=256):
        begin = time.time()
        raw_predictions = model.predict(X_test, batch_size=batch_size, verbose=0)
        end = time.time()
        mse = sklearn.metrics.mean_squared_error(Y_test, raw_predictions)
        iou = [Metrics.box_iou(x, y) for x, y in zip(raw_predictions, Y_test)]
        iou = np.array(iou)

        metrics = {}
        metrics['iou'] = np.mean(iou)
        metrics['mse'] = mse
        metrics['test_time'] = end - begin
        return metrics

class DetectionModel(NoScopeModel):
    def __init__(self, **kwargs):
        super(DetectionModel, self).__init__(**kwargs)
        self.pred_type = PredType.DETECTION

    def get_loss(self):
        def det_loss(y_true, y_pred):
            yt1 = y_true[:, 0:2]
            yp1 = y_pred[:, 0:2]
            yt2 = y_true[:, 2:]
            yp2 = y_pred[:, 2:]
            # FIXME: lambda?
            return keras.losses.categorical_crossentropy(yt1, yp1) + \
                y_true[:, 1] * keras.losses.mse(yt2, yp2)
        return det_loss

    def get_metrics(self):
        def classification_metric(y_true, y_pred):
            yt = y_true[:, 0:2]
            yp = y_pred[:, 0:2]
            return keras.metrics.categorical_accuracy(yt, yp)
        def box_mse(y_true, y_pred):
            yt = y_true[:, 2:]
            yp = y_pred[:, 2:] * K.expand_dims(y_true[:, 1], axis=1)
            return keras.metrics.mse(yt, yp)
        def box_iou(y_true, y_pred):
            yt = y_true[:, 2:]
            yp = y_pred[:, 2:]
            return np.mean(map(lambda x, y: Metrics.box_iou(x, y), zip(yt, yp)))
        return [classification_metric, box_mse]#, box_iou]

    def evaluate_model(self, model, X_test, Y_test, batch_size=256):
        begin = time.time()
        Y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
        end = time.time()

        box_inds = Y_test[:, 1] == 1
        Y_pred_box = Y_pred[box_inds, 2:]
        Y_true_box = Y_test[box_inds, 2:]
        mse = sklearn.metrics.mean_squared_error(Y_true_box, Y_pred_box)
        iou = [Metrics.box_iou(x, y) for x, y in zip(Y_true_box, Y_pred_box)]

        proba = Y_pred[:, 0:2]
        metrics = Metrics.classification_metrics(proba, Y_test[:, 0:2])
        metrics['iou'] = np.mean(iou)
        metrics['mse'] = mse
        metrics['test_time'] = end - begin
        return metrics

# FIXME: untested
class MulticlassClassificationModel(NoScopeModel):
    def __init__(self, **kwargs):
        super(MulticlassClassificationModel, self).__init__(**kwargs)
        # FIXME
        self.pred_type = PredType.BINARY

    def get_loss(self):
        return 'categorical_crossentropy'

    def get_metrics(self):
        return ['accuracy']

    def evaluate_model(self, model, X_test, Y_test, batch_size=256):
        begin = time.time()
        proba = model.predict(X_test, batch_size=batch_size, verbose=0)
        test_time = time.time() - begin

        predicted_labels = np_utils.probas_to_classes(proba)
        true_labels = np_utils.probas_to_classes(Y_test)
        precision, recall, fbeta, support = \
            sklearn.metrics.precision_recall_fscore_support(predicted_labels, true_labels)
        accuracy = sklearn.metrics.accuracy_score(predicted_labels, true_labels)

        num_penalties, thresh_low, thresh_high = \
            StatsUtils.yolo_oracle(Y_test[:, 1], proba[:, 1])
        windowed_acc, windowed_supp = StatsUtils.windowed_accuracy(predicted_labels, Y_test)

        metrics = {'precision': precision,
                   'recall': recall,
                   'fbeta': fbeta,
                   'support': support,
                   'accuracy': accuracy,
                   'penalities': num_penalties,
                   'windowed_accuracy': windowed_acc,
                   'windowed_support': windowed_supp,
                   'test_time': test_time}
        return metrics
