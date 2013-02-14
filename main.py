# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:56:19 2013

@author: Nicholas Léonard
"""

import theano.tensor as T
import theano
import numpy
import scipy
import cPickle
import sys


import functools

import warnings
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)

from collections import OrderedDict

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
#from pylearn2.datasets.dataset import Dataset
from pylearn2.training_algorithms.sgd import SGD, EpochCounter, ExponentialDecayOverEpoch, MomentumAdjustor
from pylearn2.train import Train
from pylearn2.monitor import Monitor
from pylearn2.costs.cost import MethodCost
from pylearn2.models.mlp import MLP, RectifiedLinear, Linear, Softmax
from pylearn2.space import VectorSpace

from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

from contest_dataset import ContestDataset

import matplotlib.pyplot as pyplot

class Tanh(Linear):
    def __init__(self,
                 dim,
                 layer_name,
                 irange = None,
                 istdev = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 mask_weights = None,
                 max_row_norm = None,
                 max_col_norm = None,
                 copy_input = 0,
                 use_bias = True):
                     
        Linear.__init__(self, dim=dim, layer_name=layer_name, irange=irange, 
                        istdev=istdev, sparse_init=sparse_init, 
                        sparse_stdev=sparse_stdev, include_prob=include_prob, 
                        init_bias=init_bias, W_lr_scale=W_lr_scale, 
                        b_lr_scale=b_lr_scale, mask_weights=mask_weights, 
                        max_row_norm=max_row_norm, max_col_norm=max_col_norm, 
                        copy_input=copy_input, use_bias=use_bias)
                        
    def fprop(self, state_below):

        p = T.tanh(self._fprop(state_below))
        
        if self.copy_input:
            p = T.concatenate((p, state_below), axis=1)

        return p
        
def get_valid_ddm(path='../data'):
    return ContestDataset(which_set='train',
                base_path = path,
                start = 3300,
                stop = 4150,
                preprocessor = None)

def train():
    train_ddm = ContestDataset(which_set='train',
                base_path = '../data',
                start = 0,
                stop = 3300,
                preprocessor = None)
       
    valid_ddm = get_valid_ddm()
    
    nvis = train_ddm.get_design_matrix().shape[1]
    nout = train_ddm.get_targets().shape[1]
    ntrain = train_ddm.get_design_matrix().shape[0]
    nvalid = valid_ddm.get_design_matrix().shape[0]
    batch_size = 50
    num_train_batch = ntrain/batch_size
    
    print "nvis, nout :", nvis, nout
    print "ntrain :", ntrain
    print "nvalid :", nvalid
    
    
    layer_sizes = [nvis, 4000, 4000]
    
    # Hidden Layers:
    layers = []
    for i in range(len(layer_sizes)-1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i+1]
        print n_in, n_out
        layers.append(RectifiedLinear( 
                            dim = n_out,
                            layer_name="hidden"+str(i),
                            left_slope = 0.0,
                            sparse_init = 15,
                            #istdev = 0.01,
                            max_col_norm = 15,
                            #irange = numpy.sqrt(6. / (n_in + n_out))
                        )
                    )
    
    # Output Layer
    layers.append(
        Softmax(layer_name='output', n_classes=nout, irange=0.1)
    )
    
    print layers
    # Build Multi-layer perceptron from layers:
    model = MLP(layers,
            batch_size = batch_size,
            dropout_probs = [0.1,0.25,0.25],
            #dropout_scales = None,
            nvis=nvis)
    
    extensions = [   
        ExponentialDecayOverEpoch(decay_factor=0.998, min_lr=0.000001),
        MomentumAdjustor(final_momentum=0.99, start=0, saturate=300),
        MonitorBasedSaveBest(
            channel_name = "valid_output_misclass",
            save_path = "mlp_2_best.pkl" \
        )
    ]
    
    monitoring_dataset = {'train': train_ddm, 'valid': valid_ddm}
    algorithm = SGD(learning_rate=0.00001,
                    cost=MethodCost('cost_from_X', supervised = True),
                    batch_size=batch_size,
                    batches_per_iter=num_train_batch,
                    monitoring_dataset=monitoring_dataset,
                    termination_criterion=EpochCounter(1000),
                    init_momentum=0.5,
                    train_iteration_mode='random_uniform')
    
    
    learner = Train(dataset=train_ddm,
                    model=model,
                    algorithm=algorithm,
                    extensions=extensions)
                
    learner.main_loop()

def validate(model_path):
    from pylearn2.utils import serial
    try:
        model = serial.load(model_path)
    except Exception, e:
        print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
        print e
    
    dataset = get_valid_ddm()
    
    
    X = model.get_input_space().make_batch_theano()
    Ta = model.get_output_space().make_batch_theano()
    
    C = model.valid_cost_from_X(X, Ta)
    C2 = model.cost_from_X(X, Ta)
    
    
    from theano import tensor as T
    
    Y = model.fprop(X, apply_dropout=False)
    A = T.mean(T.cast(T.eq(T.argmax(Y, axis=1), T.argmax(Ta, axis=1)), dtype='int32'))
    
    from theano import function
    
    y, y2, acc = function([X, Ta], [C, C2, A])(dataset.X.astype(X.dtype), dataset.y.astype(Ta.dtype))
    
   
    print y, y2, acc
    
if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'validate':
        validate(sys.argv[2])
        
