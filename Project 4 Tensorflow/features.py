# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples
import matplotlib.pylab as plt # Yantian
from threading import Thread # Yantian
import time

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.
colorbar()
    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features = basicFeatureExtractor(datum)

    "*** YOUR CODE HERE ***"
    new_features1 = np.zeros(4, int)
    count = 0
    for i in range(DIGIT_DATUM_HEIGHT):
        for j in range(DIGIT_DATUM_WIDTH):
            if datum[i][j] == 0:
                count += 1

    if 0<= count <=499:
        new_features1[0] = 1
    elif 500<= count <=600:
        new_features1[1] = 1
    elif 600< count <=670:
        new_features1[2] = 1
    elif 670< count <=784:
        new_features1[3] = 1

    new_features2 = np.zeros(2, int)
    count = 0
    for i in range(DIGIT_DATUM_HEIGHT):
        for j in range(DIGIT_DATUM_WIDTH):
            if datum[i][j] == 1:
                count += 1

    if 0<= count <=50:
        new_features2[0] = 1
    else:
        new_features2[1] = 1

    new_features3 = np.zeros(2, int)
    white_regions = 0
    visited = set()
    for i in range(DIGIT_DATUM_HEIGHT):
        for j in range(DIGIT_DATUM_WIDTH):
            if (datum[i][j] == 0) and ((i,j) not in visited):
                nbbrs = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1),
                             (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
                dfs(i, j, visited, nbbrs, datum)
                white_regions += 1
    if white_regions == 1:
        new_features3[0] = 1
    else:
        new_features3[1] = 1

    return np.concatenate((features, new_features1, new_features2, new_features3))

def dfs(i, j, visited, nbbrs, datum):
    if (i, j) not in visited:
        visited.add((i, j))
    for a, b in nbbrs:
        if (0 <= a < DIGIT_DATUM_HEIGHT) and (0 <= b < DIGIT_DATUM_WIDTH) and ((a, b) not in visited):
            if datum[a][b] == 0:
                nbbrs = [(a - 1, b - 1), (a - 1, b), (a - 1, b + 1), (a, b - 1), (a, b + 1),
                             (a + 1, b - 1), (a + 1, b), (a + 1, b + 1)]
                dfs(a, b, visited, nbbrs, datum)


def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit
    
    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    for i in range(len(trainPredictions)):
        prediction = trainPredictions[i]
        truth = trainLabels[i]
        if (prediction != truth):
    #         print "==================================="
            print "Mistake on example %d" % i
            print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
