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

from hps2 import HPS, HPSData
from pylearn2.training_algorithms.sgd import SGD

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
        """for i in xrange(5):
            mirror_patch = []
            for j in xrange(42):
                mirror_patch.append(patches[i][:,:,42-(j+1):42-j])
            patches.append(T.concatenate(mirror_patch,axis=2))"""
       
        """for patch in patches:
            Y_list.append(model.fprop(patch, apply_dropout=False))
         
        Y = T.mean(T.stack(Y_list), axis=(1,2))"""
        Y = model.fprop(patches[-1], apply_dropout=False) 
        i = 1
        for patch in patches[:-1]:
            Y = Y + model.fprop(patch, apply_dropout=False)
            i+=1
        print i
        Y = Y/float(i)
        return T.mean(T.cast(T.eq(T.argmax(Y, axis=1), 
                           T.argmax(target, axis=1)), dtype='int32'),
                           dtype=config.floatX)
    def get_trainingAlgorithm(self, config_id, config_class, cost):
        if 'sgd' in config_class:
            (learning_rate,batch_size,init_momentum,train_iteration_mode) \
                = self.select_train_sgd(config_id)
            num_train_batch = (self.ntrain/batch_size)/8
            print "num training batches:", num_train_batch
            termination_criterion \
                = self.get_termination(config_id, config_class)
            return SGD( learning_rate=learning_rate,
                        cost=cost,
                        batch_size=batch_size,
                        batches_per_iter=num_train_batch,
                        monitoring_dataset=self.monitoring_dataset,
                        termination_criterion=termination_criterion,
                        init_momentum=init_momentum,
                        train_iteration_mode=train_iteration_mode) 
        else:
            raise HPSData("training class not supported:"+str(config_class))

def get_valid_ddm(path='../data'):
    return ContestDataset(which_set='train',
                base_path = path,
                start = 3584,
                stop = 4096,
                preprocessor = Standardize(),
                fit_preprocessor = True)
                
def validate(model_path):
    from pylearn2.utils import serial
    try:
        model = serial.load(model_path)
    except Exception, e:
        print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
        print e
    
    dataset = get_valid_ddm()
    
    # use smallish batches to avoid running out of memory
    batch_size = 32
    model.set_batch_size(batch_size)
    # dataset must be multiple of batch size of some batches will have
    # different sizes. theano convolution requires a hard-coded batch size
    """ m = dataset.X.shape[0]
    extra = batch_size - m % batch_size
    assert (m + extra) % batch_size == 0
    import numpy as np
    if extra > 0:
        dataset.X = np.concatenate((dataset.X, np.zeros((extra, dataset.X.shape[1]),
        dtype=dataset.X.dtype)), axis=0)
    assert dataset.X.shape[0] % batch_size == 0"""
    
    
    X = model.get_input_space().make_batch_theano()
    Ta = model.get_output_space().make_batch_theano()
    patches = []
    patches.append(X[:,:42,:42,:])
    patches.append(X[:,6:,:42,:])
    patches.append(X[:,6:,6:,:])
    patches.append(X[:,:42,6:,:])
    patches.append(X[:,3:45,3:45,:])
    for i in xrange(5):
        mirror_patch = []
        for j in xrange(42):
            mirror_patch.append(patches[i][:,:,42-(j+1):42-j,:])
        patches.append(T.concatenate(mirror_patch,axis=2))
   
    """for patch in patches:
        Y_list.append(model.fprop(patch, apply_dropout=False))
     
    Y = T.mean(T.stack(Y_list), axis=(1,2))"""
    Y = model.fprop(patches[-1], apply_dropout=False) 
    i = 1
    for patch in patches[:-1]:
        Y = Y + model.fprop(patch, apply_dropout=False)
        i+=1
    print i
    Y = Y/float(i)
    A = T.cast(T.eq(T.argmax(Y, axis=1), 
                       T.argmax(Ta, axis=1)), dtype='int32')
    
    from theano import function
    
    f = function([X, Ta], A)
    
    
    Acc = []
    
    for i in xrange(dataset.X.shape[0] / batch_size):
        x_arg = dataset.X[i*batch_size:(i+1)*batch_size,:]
        y_arg = dataset.y[i*batch_size:(i+1)*batch_size,:]
        if X.ndim > 2:
            x_arg = dataset.get_topological_view(x_arg)
        Acc.append(f(x_arg.astype(X.dtype), y_arg.astype(Ta.dtype)))
    #print Acc[0].shape
    Acc = np.concatenate(Acc)
    #print Acc, Acc.shape, dataset.X.shape, dataset.y.shape
    assert Acc.ndim == 1
    assert Acc.shape[0] == dataset.X.shape[0]
    # discard any zero-padding that was used to give the batches uniform size
    #Acc = Acc[:m]
    
    print Acc.mean()
    
    
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
        task_id = int(sys.argv[2])
        start_config_id = None
        if len(sys.argv) > 3:
            start_config_id = int(sys.argv[3])
        log_channel_names = ['train_output_misclass',
                            'Validation Classification Accuracy']
        mbsb_channel_name = 'Validation Missclassification'
        hps = MyHPS(dataset_name = "Emotion Recognition Augmented",
                  task_id = task_id, 
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
        
