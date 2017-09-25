import sklearn
import sklearn.metrics
import StatsUtils
import np_utils

def classification_metrics(proba, Y_test):
    predicted_labels = np_utils.probas_to_classes(proba)
    true_labels = np_utils.probas_to_classes(Y_test)
    precision, recall, fbeta, support = \
        sklearn.metrics.precision_recall_fscore_support(predicted_labels, true_labels)
    accuracy = sklearn.metrics.accuracy_score(predicted_labels, true_labels)

    num_penalties, thresh_low, thresh_high = \
        StatsUtils.yolo_oracle(Y_test[:, 1], proba[:, 1])
    windowed_acc, windowed_supp = StatsUtils.windowed_accuracy(predicted_labels, Y_test)

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

    metrics = {'precision': precision,
               'recall': recall,
               'fbeta': fbeta,
               'support': support,
               'accuracy': accuracy,
               'penalities': num_penalties,
               'windowed_accuracy': windowed_acc,
               'windowed_support': windowed_supp}
    return metrics

def box_iou(box1, box2):
    def to_cent(box):
        xmin, ymin, xmax, ymax = box
        xcent = (xmax + xmin) / 2
        ycent = (ymax + ymin) / 2
        return (xcent, ycent, xmax - xcent, ymax - ycent)
    # box1 = to_cent(box1)
    # box2 = to_cent(box2)
    if box1[0] + box1[2] <= box2[0] - box2[2] or \
            box2[0] + box2[2] <= box1[0] - box1[2] or \
            box1[1] + box1[3] <= box2[1] - box2[3] or \
            box2[1] + box2[3] <= box1[1] - box1[3]:
        return 0.0
    else:
        xA = min(box1[0] + box1[2], box2[0] + box2[2])
        yA = min(box1[1] + box1[3], box2[1] + box2[3])
        xB = max(box1[0] - box1[2], box2[0] - box2[2])
        yB = max(box1[1] - box1[3], box2[1] - box2[3])
        interArea = (xA - xB) * (yA - yB)
        box1Area = (2 * box1[2]) * (2 * box1[3])
        box2Area = (2 * box2[2]) * (2 * box2[3])
        return max(interArea / float(box1Area + box2Area - interArea), 0.0)
