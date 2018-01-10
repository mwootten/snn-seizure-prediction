# PCA taken from:
# https://glowingpython.blogspot.com/2011/07/principal-component-analysis-with-numpy.html

# When running PCA:
# The variables are the times
# The observations are the channels

import numpy as np
import itertools
from typing import List
import sys

def princomp(A):
    """
    Performs principal components analysis (PCA) on the n-by-p data matrix A.
    Rows of A correspond to observations, columns to variables.

    Returns :
    coeff :
        is a p-by-p matrix, each column containing coefficients for one
        principal component.
    score :
        the principal component scores; that is, the representation of A in the
        principal component space. Rows of SCORE correspond to observations,
        columns to components.
    latent :
        a vector containing the eigenvalues of the covariance matrix of A.
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = np.linalg.eig(np.cov(M)) # attention:not always sorted
    score = np.dot(coeff.T,M) # projection of the data in the new space
    return (coeff, score, latent)

def readFileSequence(fileNames):
    allFiles = [np.fromfile(f, dtype=np.dtype('i4')) for f in fileNames]
    return np.concatenate(allFiles)


def makeChannelMatrix(fileNames):
    channelNames = []
    observations = []
    fileGroups = itertools.groupby(fileNames, lambda name: name.split("-")[0])
    for (channelName, fileNames) in fileGroups:
        channelNames.append(channelName)
        observationRow = readFileSequence(fileNames)
        observationRowMatrix = np.reshape(observationRow, (-1, observationRow.shape[0]))
        observations.append(observationColumn)
    channelMatrix = np.concatenate(observations)
    print(channelMatrix.shape)
    return channelMatrix

def readRow2D(fileName):
    row1d = np.fromfile(fileName, dtype=np.dtype('i4'))
    row2d = np.reshape(row1d, (-1, row1d.shape[0]))
    return row2d

def makeSingleSliceMatrix(fileNames):
    return np.concatenate(list(map(readRow2D, fileNames)))

def performPCA(fileNames):
    pcaMatrix = makeSingleSliceMatrix(fileNames)
    coeff, score, latent = princomp(pcaMatrix)
    component1 = np.asarray(coeff[:, 1].round(), dtype=np.int32)
    outputName = 'PCA-' + '-'.join(fileNames[0].split('-')[1:])
    component1.tofile(outputName)

performPCA(sys.argv[1:])
