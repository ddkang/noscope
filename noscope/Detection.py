import keras
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization

from keras import backend as K
def detection_act(t, activation='sigmoid'):
    t1 = keras.activations.softmax(t[:, 0:2])
    t2 = t[:, 2:]
    if activation == 'relu':
        t2 = keras.activations.relu(t2)
    elif activation == 'sigmoid':
        t2 = keras.activations.sigmoid(t2)
    return K.concatenate([t1, t2], axis=-1)

def det_loss(y_true, y_pred):
    yt1 = y_true[:, 0:2]
    yp1 = y_pred[:, 0:2]
    yt2 = y_true[:, 2:]
    yp2 = y_pred[:, 2:]
    # FIXME: lambda?
    return keras.losses.categorical_crossentropy(yt1, yp1) + \
        y_true[:, 1] * keras.losses.mse(yt2, yp2) / K.mean(y_true[:, 1])
return det_loss

def classification_metric(y_true, y_pred):
    yt = y_true[:, 0:2]
    yp = y_pred[:, 0:2]
    return keras.metrics.categorical_accuracy(yt, yp)
def box_mse(y_true, y_pred):
    yt = y_true[:, 2:]
    yp = y_pred[:, 2:] * K.expand_dims(y_true[:, 1], axis=1)
    return keras.metrics.mse(yt, yp) / K.mean(y_true[:, 1])
def box_iou(y_true, y_pred):
    yt = y_true[:, 2:]
    yp = y_pred[:, 2:]
    return np.mean(map(lambda x, y: Metrics.box_iou(x, y), zip(yt, yp)))

