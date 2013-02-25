# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 23:56:22 2013

@author: nicholas
"""

import numpy as np
from pylab import plot, show, title, xlabel, ylabel, subplot, grid
from scipy import arange, misc
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pylearn2.utils import serial

from random import randint, shuffle
#from dispims import dispims


import csv
import numpy as np

from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix



if __name__ == '__main__':
    which_set = 'train'
    csv_file = open("../data/train.csv", 'r')
    reader = csv.reader(csv_file)
    row = reader.next()
    
    y_list = []
    X_list = []
    
    for row in reader:
        if which_set == 'train':
            y_str, X_row_str = row
            y = int(y_str)
            y_list.append(y)
        else:
            X_row_str ,= row
        X_row_strs = X_row_str.split(' ')
        X_row = map(lambda x: float(x), X_row_strs)
        X_list.append(X_row)
        
    X = np.asarray(X_list)
    X = X.reshape((X.shape[0],48,48))
    if which_set == 'train':
        y = np.asarray(y_list)
    
        one_hot = np.zeros((y.shape[0],7),dtype='float32')
        for i in xrange(y.shape[0]):
            one_hot[i,y[i]] = 1.
        y = one_hot
    else:
        y = None
    
    # leave out validation set
    X = X[0:3584, :]
    if y is not None:
        y = y[0:3584, :]
    
    # standardize:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    
    X = (X - mean) / (1e-4 + std)
    
    # subsample images:
    num_images = X.shape[0]
    print 'image shape', X.shape
    
    new_X = np.zeros((num_images*36*2, 42, 42), dtype='float32')
    new_y = np.zeros((num_images*36*2, 7), dtype='float32')
    k = 0
    for i in xrange(0, 6):
        for j in xrange(0, 6):
            new_X[k*num_images:(k+1)*num_images,:] = X[:,i:i+42,j:j+42]
            new_y[k*num_images:(k+1)*num_images,:] = y[:,:]
            k+=1
    
    num_half_subsamples = num_images*36
    print new_X[0,:]
    print new_X[-(num_half_subsamples+1),:]
    
    
    # mirror images horizontally:
    
    for i in xrange(42):
        new_X[num_half_subsamples:,:,i] = new_X[:num_half_subsamples,:,41-i]
    
    new_y[num_half_subsamples:,:] = new_y[:num_half_subsamples,:]
    
    #plt.imshow(new_X[-1,:], cmap = cm.Greys_r)
    #plt.show()
    
    print new_X.shape
    new_X = new_X.reshape((new_X.shape[0],42*42))
    
    indexes = np.asarray(range(new_X.shape[0]), dtype='int32')
    shuffle(indexes)
    new_X = new_X[indexes,:]
    new_y = new_y[indexes,:]
    
    np.save("../data/train_X", new_X)
    np.save("../data/train_y", new_y)
    
    """validation will be performed by propagating 10 subsamples of a base 
    image. Validation will not be subsampled into a new larger set. This will
    be done by a special validation function. The same function will be applied
    during testing."""