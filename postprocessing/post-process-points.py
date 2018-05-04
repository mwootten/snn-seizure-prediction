import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--predicted", default="predicted",
                    help="Specify the file containing the predictions")
parser.add_argument("-t", "--true", default="true",
                    help="Specify the values containing the correct values")
parser.add_argument("-o", "--output",
                    help="Specify the image file for the output")

args = parser.parse_args()

predicted = list(map(float, open(args.predicted, 'r').readlines()))
true = list(map(float, open(args.true, 'r').readlines()))

def classify(value, threshold):
    if value > threshold:
        return 1
    else:
        return 0

def confusion_matrix(threshold):
    predicted_classification = [classify(value, threshold) for value in predicted]
    pairs = zip(predicted_classification, true)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for pair in pairs:
        if pair == (0, 0):
            tn += 1
        if pair == (0, 1):
            fn += 1
        if pair == (1, 0):
            fp += 1
        if pair == (1, 1):
            tp += 1

    return (tn, fn, fp, tp)

def sensitivity(threshold):
    tn, fn, fp, tp = confusion_matrix(threshold)
    if (tp + fn == 0):
        return 1
    return tp / (tp + fn)

def specificity(threshold):
    tn, fn, fp, tp = confusion_matrix(threshold)
    if (tn + fp == 0):
        return 1
    return tn / (tn + fp)

def roc_points():
    '''
    Get a bunch of points on the ROC graph. The basic idea here is to iterate
    over all the reasonable thresholds (the unique values in the "predicted" list)
    and then compute the corresponding point on the graph. This can then be
    run through a trapezoidal integral approximation to get the AUROC.

    The pairs are (1 - specificity, sensitivity), or (FPR, TPR)
    '''

    thresholds = [0, *sorted(predicted), 1]
    fpr = [1 - specificity(threshold) for threshold in thresholds]
    tpr = [sensitivity(threshold) for threshold in [0, *sorted(predicted), 1]]
    return (np.array(fpr), np.array(tpr))

def plot_curve(roc, file):
    (fpr, tpr) = roc
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Reciever operating characteristic')
    plt.savefig(file)

def gini_coefficient(points):
    '''
    Calculate the area between the ROC curve and the no-discrimination (TPR/FPR
    or 45 degree line) from [0, 1] using a trapezoidal approximation.
    '''
    (fpr, tpr) = points
    xs = fpr
    ys = tpr - fpr

    widths = -np.diff(xs)
    lvals = ys[:-1]
    rvals = ys[1:]
    return 2 * sum(1/2 * (lvals + rvals) * widths)

roc = roc_points()
print(roc)
