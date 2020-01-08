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

def getContiguousWhiteSpaces(graph):

    def getPixel(x, y, graph):
        return graph[y][x]
    
    def findSingleWhiteSpaceCoords(graph, closedSet1, closedSet2):
        """Finds a white space's coordinates on the graph that is not contained in the closed sets passed in"""

        N = 28
        
        for i in range(N):
            for j in range(N):
                randomPixel = getPixel(i, j, graph)
                if (randomPixel == 0) and ( (i,j) not in closedSet1) and ( (i,j) not in closedSet2):
                    
                    return (i, j)

        return (-1, -1)             #Couldn't find any more whitespace

    def addNeighborsToFringe(x, y, fringe):
        if x > 0:
            fringe.push( (x-1,y) )
        
        if x < len(graph)-1:
            fringe.push( (x+1,y) )
        
        if y > 0:
            fringe.push( (x,y-1) )

        if y < len(graph)-1:
            fringe.push( (x,y+1) )

    fringe = util.Queue()
    closedSet1      = set()
    closedSet2      = set()
    closedSet2Plus  = set()
    
    #Finding a first whitespace
    x, y = findSingleWhiteSpaceCoords(graph, closedSet1, closedSet2)

    #Adding it to the fringe
    fringe.push( (x,y) )

    #Keep going until I've found 1 total contiguous whitespace
    while not fringe.isEmpty():
        (x,y) = fringe.pop()

        if (x,y) not in closedSet1:
            closedSet1.add( (x,y) )

            #Only add white pixels' neighbors to fringe
            if 0 == getPixel( x, y, graph ):
                addNeighborsToFringe( x, y, fringe )

    #============
    #After finding one contiguous whitespace, another one may exist
    nextWhiteSpace = findSingleWhiteSpaceCoords(graph, closedSet1, closedSet2)

    #Handles case that I couldn't find another white space in a different group
    if nextWhiteSpace == (-1, -1):
        return [1, 0, 0]

    #Add this nextWhiteSpace to the fringe and find all its neighbors
    fringe.push( nextWhiteSpace )
    
    #Keep going until I've found my second contiguous whitespace
    while not fringe.isEmpty():
        (x,y) = fringe.pop()

        if (x,y) not in closedSet2:
            closedSet2.add( (x,y) )

            #Only add white pixels' neighbors to fringe
            if 0 == getPixel( x, y, graph ):
                addNeighborsToFringe( x, y, fringe )    
    
    #============
    #After finding two contiguous whitespaces, another one may exist
    nextWhiteSpace = findSingleWhiteSpaceCoords(graph, closedSet1, closedSet2)

    #Handles case that I couldn't find another white space in a different group (only have 2 whitespaces)
    if nextWhiteSpace == (-1, -1):
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
        
        Amount of whitespace
        Number of contiguous areas of whitespace

        Binary value
    ##
    """
    features = basicFeatureExtractor(datum)

    "*** YOUR CODE HERE ***"
    listRep = []

    for i in range(28):
        listRep.append( [] )
        index = i*28
        for j in range(28):
            listRep[i].append( features[index+j] )
        
    
    #A 3 element ternary list with 1 value turned on for the number of whiteSpaces [1, 2, >2]
    contiguousWhiteSpaces = getContiguousWhiteSpaces(listRep)

    contiguousWhiteSpaces = np.array( contiguousWhiteSpaces )

    features = np.concatenate( (features, contiguousWhiteSpaces), axis=0 )

    return features
    


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
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
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
