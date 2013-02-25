# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:56:19 2013

@author: Nicholas Léonard
"""


import sys

from pylearn2.datasets.preprocessing import Standardize

from contest_dataset import ContestDataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.monitor import Monitor

from hps import HPS

import numpy as np
import theano.tensor as T
from theano import config
from theano import function


class MyHPS(HPS):
    def get_classification_accuracy(self, model, minibatch, target):
        
        patches = []
        patches.append(minibatch[:,:42,:42])
        patches.append(minibatch[:,6:,:42])
        patches.append(minibatch[:,6:,6:])
        patches.append(minibatch[:,:42,6:])
        patches.append(minibatch[:,3:45,3:45])
        for i in xrange(5):
            mirror_patch = []
            for j in xrange(42):
                mirror_patch.append(patches[i][:,:,42-(j+1):42-j])
            patches.append(T.concatenate(mirror_patch,axis=2))
        #Y = model.fprop(patches[-1], apply_dropout=False) 
        Y_list = []
        for patch in patches:
            Y_list.append(model.fprop(patch, apply_dropout=False))
         
        Y = T.mean(T.stack(Y_list), axis=(1,2))
        return T.mean(T.cast(T.eq(T.argmax(Y, axis=1), 
                           T.argmax(target, axis=1)), dtype='int32'),
                           dtype=config.floatX)

def get_valid_ddm(path='../data'):
    return ContestDataset(which_set='train',
                base_path = path,
                start = 3584,
                stop = 4096,
                preprocessor = Standardize())
                
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
        """train_ddm = ContestDataset(which_set='train',
                base_path = "../data/",
                start = 0,
                stop = 3584,
                preprocessor = Standardize())"""
        train_ddm = DenseDesignMatrix(
                    X=np.load("../data/train_X.npy"),
                    y=np.load("../data/train_y.npy"),
                    view_converter = DefaultViewConverter(shape=[42,42,1]))
        #preprocessor = Standardize()
        #preprocessor.apply(train_ddm)
        
        valid_ddm = get_valid_ddm()
        experiment_id = int(sys.argv[2])
        start_config_id = None
        if len(sys.argv) > 3:
            start_config_id = int(sys.argv[3])
        log_channel_names = ['train_output_misclass',
                            'Validation Classification Accuracy']
        mbsb_channel_name = 'Validation Classification Accuracy'
        hps = HPS(experiment_id = experiment_id, 
                  train_ddm = train_ddm, 
                  valid_ddm = valid_ddm, 
                  log_channel_names = log_channel_names,
                  mbsb_channel_name = mbsb_channel_name)
        hps.run(start_config_id)
            
    elif sys.argv[1] == 'validate':
        validate(sys.argv[2])
    else:
        print """Usage: python main.py train "experiment_id" ["config_id"]
                    or
                        python main.py validate "path/to/model.pkl"
              """
        
