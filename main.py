# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:56:19 2013

@author: Nicholas Léonard
"""

import numpy
import scipy
import cPickle
import sys, time

from pylearn2.utils import serial
import theano.tensor as T
from theano import function
import theano

import functools

import warnings
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)

from collections import OrderedDict

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
#from pylearn2.datasets.dataset import Dataset
from pylearn2.training_algorithms.sgd import SGD, EpochCounter, \
    ExponentialDecayOverEpoch, MomentumAdjustor
from pylearn2.train import Train
from pylearn2.monitor import Monitor
from pylearn2.costs.cost import MethodCost, SumOfCosts
from pylearn2.models.mlp import MLP, ConvRectifiedLinear, RectifiedLinear, \
    Linear, Softmax, Tanh, Sigmoid, SoftmaxPool, Uniform, Normal, Sparse, \
    WeightDecay
from pylearn2.space import VectorSpace
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.datasets.preprocessing import Standardize

from multiprocessing import Process
from contest_dataset import ContestDataset
from database import DatabaseHandler

"""
TODO:
    add affineless softmax layer for Ian
    add sql monitor
    add config desc, model_desc, etc.
"""

class HPSData( Exception ): pass

class HPS:#(Process):
    """
    Hyper Parameter Search
    
    Maps pylearn2 to a postgresql database. The idea is to accumulate 
    structured data concerning the hyperparameter optimization of 
    pylearn2 models and various datasets. With enough such structured data,
    one could train a meta-model that could be used for efficient 
    sampling of hyper parameter configurations.
    
    Jobman doesn't provide this since its data is unstructured and 
    decentralized. To centralize hyper parameter data, we would need to 
    provide it in the form of a ReSTful web service API.
    
    For now, I just use it instead of the jobman database to try various 
    hyperparameter configurations.
    
    """
    def __init__(self, experiment_id, train_ddm, valid_ddm):
        self.experiment_id = experiment_id
        
        self.train_ddm = train_ddm
        self.valid_ddm = valid_ddm
        self.nvis = self.train_ddm.get_design_matrix().shape[1]
        self.nout = self.train_ddm.get_targets().shape[1]
        self.ntrain = self.train_ddm.get_design_matrix().shape[0]
        self.nvalid = self.valid_ddm.get_design_matrix().shape[0]
        self.monitoring_dataset \
            = {'train': self.train_ddm, 'valid': self.valid_ddm}
            
        print "nvis, nout :", self.nvis, self.nout
        print "ntrain :", self.ntrain
        print "nvalid :", self.nvalid
        
        #Process.__init__(self)
    def run(self, start_config_id = None):
        self.db = DatabaseHandler()
        print 'running'
        while True:
            if start_config_id is None:
                (config_id, model_id, ext_id, train_id,
                    dataset_id, random_seed, batch_size) \
                     = self.select_next_config(self.experiment_id)
            else:
                (config_id, model_id, ext_id, train_id,
                    dataset_id, random_seed, batch_size) \
                     = self.select_config(start_config_id)
            start_config_id = None
            
            (dataset_desc, nvis) = self.select_dataset(dataset_id)
            
            model = self.get_model(model_id, random_seed, batch_size, nvis)
            
            extensions = self.get_extensions(ext_id)
            
            # TODO: add monitor based info to db
            # TODO: add MonitorBasedSaveBest to db and decouple from mlp
            # we could have a default name for last layer
            save_path = "mlp_"+str(config_id)+"_best.pkl"
            extensions.append(
                MonitorBasedSaveBest(
                    channel_name \
                        = "valid_"+str(model.layers[-1].layer_name)+"_misclass",
                    save_path = save_path \
                )
            )
            
            algorithm = self.get_trainingAlgorithm(train_id, batch_size)
            
            print 'sgd complete'
            learner = Train(dataset=self.train_ddm,
                            model=model,
                            algorithm=algorithm,
                            extensions=extensions)
            print 'learning'     
            learner.main_loop()
            
            # validate:
            try:
                model = serial.load(save_path)
            except Exception, e:
                print save_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
                print e
            X = model.get_input_space().make_batch_theano()
            Ta = model.get_output_space().make_batch_theano()
            
            C = model.valid_cost_from_X(X, Ta)
            C2 = model.cost_from_X(X, Ta)
            
            Y = model.fprop(X, apply_dropout=False)
            A = T.mean(T.cast(T.eq(T.argmax(Y, axis=1), T.argmax(Ta, axis=1)), dtype='int32'))
            
            y, y2, acc = function([X, Ta], [C, C2, A])(valid_ddm.X.astype(X.dtype), valid_ddm.y.astype(Ta.dtype))
            self.set_accuracy(config_id, float(acc))
            self.set_end_time(config_id)
    def get_trainingAlgorithm(self, train_id, batch_size):
        #TODO add cost to db
        num_train_batch = self.ntrain/batch_size
        train_class = self.select_trainingAlgorithm(train_id)
        if train_class == 'stochasticgradientdescent':
            (learning_rate, term_id, init_momentum, train_iteration_mode,
             cost_id) = self.select_train_stochasticGradientDescent(train_id)
            termination_criterion = self.get_termination(term_id)
            cost = self.get_cost(cost_id)
            return SGD( learning_rate=learning_rate,
                        cost=cost,
                        batch_size=batch_size,
                        batches_per_iter=num_train_batch,
                        monitoring_dataset=self.monitoring_dataset,
                        termination_criterion=termination_criterion,
                        init_momentum=init_momentum,
                        train_iteration_mode=train_iteration_mode) 
        else:
            raise HPSData("training class not supported:"+train_class)
    def get_cost(self, cost_id):
        cost_class = self.select_cost(cost_id)
        if cost_class == 'methodcost':
            (method, supervised) = self.select_cost_methodCost(cost_id)
            return MethodCost(method=method, supervised=supervised)
        elif cost_class == 'weightdecay':
            coeff = self.select_cost_weightDecay(cost_id)
            return WeightDecay(coeffs=coeff)
        elif cost_class == 'multi':
            cost_array = self.select_cost_multi(cost_id)
            costs = []
            for sub_cost_id in cost_array:
                costs.append(self.get_cost(sub_cost_id))
            return SumOfCosts(costs)
        else:
            raise HPSData("cost class not supported:"+str(cost_class))
    def get_model(self, model_id, random_seed, batch_size, nvis):
        model_class = self.select_model(model_id)
        if model_class == 'mlp':
            (input_layer_id, output_layer_id) \
                 = self.select_model_mlp(model_id)
            
            # TODO allow nesting of MLPs
            # TODO add dropout to layers
            # TODO add full graph capability to MLP 
            # and this part (should be made recursive):
            # TODO refactor get_graph
            
            input_layer = self.get_layer(input_layer_id)
            layers = [input_layer]
            prev_layer_id = input_layer_id
            while True:
                next_layer_id \
                    = self.select_output_layer(model_id, prev_layer_id)
                next_layer = self.get_layer(next_layer_id)
                layers.append(next_layer)
                if next_layer_id == output_layer_id:
                    # we have reached the end of the graph:
                    break
                prev_layer_id = next_layer_id
            
            # temporary hack until we get graph version of MLP:
            dropout_probs = []
            dropout_scales = []
            for layer in layers:
                dropout_probs.append(layer.dropout_prob)
                dropout_scales.append(layer.dropout_scale)
            # create MLP:
            model = MLP(layers,
                        batch_size=batch_size,
                        dropout_probs=dropout_probs,
                        dropout_scales=dropout_probs,
                        nvis=nvis,
                        random_seed=random_seed)   
            print 'mlp is built'
            return model
    def get_layer(self, layer_id):
        """Creates a Layer instance from its definition in the database."""
        (layer_class, layer_name, dim, 
         dropout_prob, dropout_scale) = self.select_layer(layer_id)
        if layer_class == 'linear':
            (init_id, init_bias, 
             W_lr_scale, b_lr_scale, 
             max_row_norm, max_col_norm) = self.select_layer_linear(layer_id)
            init_weights = self.get_init(init_id)
            layer = Linear(dim=dim, layer_name=layer_name, 
                           init_weights=init_weights, init_bias=init_bias,
                           W_lr_scale=W_lr_scale, b_lr_scale=b_lr_scale,                 
                           max_row_norm=max_row_norm, 
                           max_col_norm=max_col_norm)
        elif layer_class == 'tanh':
            (init_id, init_bias, 
             W_lr_scale, b_lr_scale, 
             max_row_norm, max_col_norm) = self.select_layer_tanh(layer_id)
            init_weights = self.get_init(init_id)
            layer = Tanh(dim=dim, layer_name=layer_name, 
                           init_weights=init_weights, init_bias=init_bias,
                           W_lr_scale=W_lr_scale, b_lr_scale=b_lr_scale,                 
                           max_row_norm=max_row_norm, 
                           max_col_norm=max_col_norm)
        elif layer_class == 'sigmoid':
            (init_id, init_bias, 
             W_lr_scale, b_lr_scale, 
             max_row_norm, max_col_norm) \
                 = self.select_layer_sigmoid(layer_id) 
            init_weights = self.get_init(init_id)
            layer = Sigmoid(dim=dim, layer_name=layer_name, 
                           init_weights=init_weights, init_bias=init_bias,
                           W_lr_scale=W_lr_scale, b_lr_scale=b_lr_scale,                 
                           max_row_norm=max_row_norm, 
                           max_col_norm=max_col_norm)
        elif layer_class == 'softmaxpool':
            (detector_layer_dim, pool_size,
	        init_id, init_bias,
	        W_lr_scale, b_lr_scale) \
                 = self.select_layer_softmaxpool(layer_id) 
            init_weights = self.get_init(init_id)
            layer = SoftmaxPool(dim=dim, layer_name=layer_name, 
                           init_weights=init_weights, init_bias=init_bias,
                           W_lr_scale=W_lr_scale, b_lr_scale=b_lr_scale)
        elif layer_class == 'softmax':
            (init_id, init_bias, 
             W_lr_scale, b_lr_scale, 
             max_row_norm, max_col_norm) \
                 = self.select_layer_softmax(layer_id) 
            init_weights = self.get_init(init_id)
            layer = Softmax(dim=dim, layer_name=layer_name, 
                           init_weights=init_weights, init_bias=init_bias,
                           W_lr_scale=W_lr_scale, b_lr_scale=b_lr_scale,                 
                           max_row_norm=max_row_norm, 
                           max_col_norm=max_col_norm)
        elif layer_class == 'rectifiedlinear':
            (init_id, init_bias, 
             W_lr_scale, b_lr_scale, 
             max_row_norm, max_col_norm,
             left_slope) = self.select_layer_rectifiedlinear(layer_id) 
            init_weights = self.get_init(init_id)
            layer = RectifiedLinear(dim=dim, layer_name=layer_name, 
                           init_weights=init_weights, init_bias=init_bias,
                           W_lr_scale=W_lr_scale, b_lr_scale=b_lr_scale,                 
                           max_row_norm=max_row_norm, 
                           max_col_norm=max_col_norm, left_slope=left_slope)
        elif layer_class == 'convrectifiedlinear':
            (init_id, init_bias, 
             W_lr_scale, b_lr_scale, 
             max_row_norm, max_col_norm) \
                 = self.select_layer_convrectifiedlinear(layer_id) 
            init_weights = self.get_init(init_id)
            layer = ConvRectifiedLinear(dim=dim, layer_name=layer_name, 
                           init_weights=init_weights, init_bias=init_bias,
                           W_lr_scale=W_lr_scale, b_lr_scale=b_lr_scale,                 
                           max_row_norm=max_row_norm, 
                           max_col_norm=max_col_norm, left_slope=left_slope)
        layer.dropout_prob = dropout_prob
        layer.dropout_scale= dropout_scale
        return layer
    def get_termination(self, term_id):
        term_class = self.select_termination(term_id)
        if term_class == 'epochcounter':
            max_epochs = self.select_term_epochCounter(term_id)
            return EpochCounter(max_epochs)
        else:
            raise HPSData("Termination class not supported:"+term_class)
    def get_init(self, init_id):
        init_class = self.select_init(init_id)
        if init_class == 'uniform':
            init_range = self.select_init_uniform(init_id)
            return Uniform(init_range = init_range)
        elif init_class == 'normal':
            stdev = self.select_init_normal(init_id)
            return Normal(stdev = stdev)
        elif init_class == 'sparse':
            (sparseness, stdev) = self.select_init_sparse(init_id)
            return Sparse(sparseness=sparseness, stdev=stdev)
    def get_extensions(self, ext_id):
        ext_class = self.select_extension(ext_id)
        if ext_class == 'exponentialdecayoverepoch':
            (decay_factor, min_lr) \
                =  self.select_ext_exponentialDecayOverEpoch(ext_id)
            return [ExponentialDecayOverEpoch(decay_factor=decay_factor,
                                             min_lr=min_lr)]
        elif ext_class == 'momentumadjustor':
            (final_momentum, start_epoch, saturate_epoch) \
                = self.select_ext_momentumAdjustor(ext_id)
            return [MomentumAdjustor(final_momentum=final_momentum,
                                    start=start_epoch, 
                                    saturate=saturate_epoch)]
        elif ext_class == 'multi':
            ext_array = self.select_ext_multi(ext_id)
            extensions = []
            for sub_ext_id in ext_array:
                extensions.extend(self.get_extensions(sub_ext_id))
            return extensions
    def set_end_time(self, config_id):
        return self.db.executeSQL("""
        UPDATE hps.config 
        SET end_time = now()
        WHERE config_id = %s
        """, (config_id,), self.db.COMMIT)  
    def set_accuracy(self, config_id, accuracy):
        return self.db.executeSQL("""
        INSERT INTO hps.validation_accuracy (config_id, accuracy)
        VALUES (%s, %s)
        """, (config_id, accuracy), self.db.COMMIT)  
    def select_trainingAlgorithm(self, train_id):
        row = self.db.executeSQL("""
        SELECT train_class
        FROM hps.trainingAlgorithm
        WHERE train_id = %s
        """, (train_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No trainingAlgorithm for train_id="+str(train_id))
        return row[0]
    def select_train_stochasticGradientDescent(self, train_id):
        row = self.db.executeSQL("""
        SELECT learning_rate, term_id, init_momentum, train_iteration_mode,
               cost_id
        FROM hps.train_stochasticGradientDescent
        WHERE train_id = %s
        """, (train_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No stochasticGradientDescent for train_id=" \
                +str(train_id))
        return row
    def select_termination(self, term_id):
        row = self.db.executeSQL("""
        SELECT term_class
        FROM hps.termination
        WHERE term_id = %s
        """, (term_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No termination for term_id="+str(term_id))
        return row[0]
    def select_term_multi(self, term_id):
        pass
    def select_term_epochCounter(self, term_id):
        row = self.db.executeSQL("""
        SELECT max_epochs
        FROM hps.term_epochcounter
        WHERE term_id = %s
        """, (term_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No epochCounter term for term_id="+str(term_id))
        return row[0]
    def select_cost(self, cost_id):
        row = self.db.executeSQL("""
        SELECT cost_class
        FROM hps.cost
        WHERE cost_id = %s
        """, (cost_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No cost for cost_id="+str(cost_id))
        return row[0]
    def select_cost_methodCost(self, cost_id):
        row = self.db.executeSQL("""
        SELECT method_name, supervised
        FROM hps.cost_methodCost
        WHERE cost_id = %s
        """, (cost_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No methodCost for cost_id="+str(cost_id))
        return row
    def select_cost_weightDecay(self, cost_id):
        row = self.db.executeSQL("""
        SELECT decay_coeff
        FROM hps.cost_weightDecay
        WHERE cost_id = %s
        """, (cost_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No weightDecay for cost_id="+str(cost_id))
        return row   
    def select_cost_multi(self, cost_id):
        row = self.db.executeSQL("""
        SELECT cost_array
        FROM hps.cost_multi
        WHERE cost_id = %s
        """, (cost_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No multi cost for cost_id="+str(cost_id))
        return row[0] 
    def select_extension(self, ext_id):
        row = self.db.executeSQL("""
        SELECT ext_class
        FROM hps.extension
        WHERE ext_id = %s
        """, (ext_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No extension for ext_id="+str(ext_id))
        return row[0]
    def select_ext_exponentialDecayOverEpoch(self, ext_id):
        row = self.db.executeSQL("""
        SELECT decay_factor, min_lr
        FROM hps.ext_exponentialDecayOverEpoch
        WHERE ext_id = %s
        """, (ext_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No exponentialDecayOverEpoch ext for ext_id=" \
                +str(ext_id))
        return row
    def select_ext_momentumAdjustor(self, ext_id):
        row = self.db.executeSQL("""
        SELECT final_momentum, start_epoch, saturate_epoch
        FROM hps.ext_momentumAdjustor
        WHERE ext_id = %s
        """, (ext_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No momentumAdjustor extension for ext_id=" \
                +str(ext_id))
        return row
    def select_ext_multi(self, ext_id):
        row = self.db.executeSQL("""
        SELECT ext_array
        FROM hps.ext_multi
        WHERE ext_id = %s
        """, (ext_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No multiple extension for ext_id=" \
                +str(ext_id))
        return row[0]
    def select_output_layer(self, model_id, input_layer_id):
        row = self.db.executeSQL("""
        SELECT output_layer_id
        FROM hps.mlp_graph AS a
        WHERE (model_id, input_layer_id) = (%s, %s)
        """, (model_id, input_layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No output layer for input layer_id=" \
                +str(input_layer_id)+" and model_id="+str(model_id))
        return row[0]
    def select_next_config(self, experiment_id):
        row = None
        for i in xrange(10):
            c = self.db.conn.cursor()
            c.execute("""
            BEGIN;
    
            SELECT  config_id, model_id, ext_id, train_id,
                    dataset_id, random_seed, batch_size  
            FROM hps.config 
            WHERE experiment_id = %s AND start_time IS NULL 
            LIMIT 1 FOR UPDATE;
            """, (experiment_id,))
            row = c.fetchone()
            if row is not None and row:
                break
            time.sleep(0.1)
            c.close()
        if not row or row is None:
            raise HPSData("No more configurations for experiment_id=" \
                +str(experiment_id)+" "+row)
        (config_id, model_id, ext_id, train_id,
         dataset_id, random_seed, batch_size) = row
        c.execute("""
        UPDATE hps.config
        SET start_time = now() 
        WHERE config_id = %s;
        """, (config_id,))
        self.db.conn.commit()
        c.close()
        return (config_id, model_id, ext_id, train_id,
                dataset_id, random_seed, batch_size)
    def select_config(self, config_id):
        row = None
        for i in xrange(10):
            c = self.db.conn.cursor()
            c.execute("""
            BEGIN;
    
            SELECT  config_id, model_id, ext_id, train_id,
                    dataset_id, random_seed, batch_size  
            FROM hps.config 
            WHERE config_id = %s 
            LIMIT 1 FOR UPDATE;
            """, (config_id,))
            row = c.fetchone()
            if row is not None and row:
                break
            time.sleep(0.1)
            c.close()
        if not row or row is None:
            raise HPSData("No more configurations for experiment_id=" \
                +str(experiment_id)+" "+row)
        (config_id, model_id, ext_id, train_id,
         dataset_id, random_seed, batch_size) = row
        c.execute("""
        UPDATE hps.config
        SET start_time = now() 
        WHERE config_id = %s;
        """, (config_id,))
        self.db.conn.commit()
        c.close()
        return (config_id, model_id, ext_id, train_id,
                dataset_id, random_seed, batch_size)
    def select_model(self, model_id):
        row = self.db.executeSQL("""
        SELECT model_class
        FROM hps.model
        WHERE model_id = %s
        """, (model_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No model for model_id="+str(model_id))
        return row[0]
    def select_model_mlp(self, model_id):
        row = self.db.executeSQL("""
        SELECT input_layer_id, output_layer_id
        FROM hps.model_mlp
        WHERE model_id = %s
        """, (model_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No configuration for model_id="+str(model_id))
        return row
    def select_dataset(self, dataset_id):
        row = self.db.executeSQL("""
        SELECT dataset_desc, dataset_nvis
        FROM hps.dataset
        WHERE dataset_id = %s
        """, (dataset_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No dataset for dataset_id="+str(dataset_id))
        return row
    def select_layer(self, layer_id):
        row = self.db.executeSQL("""
        SELECT layer_class, layer_name, dim, dropout_prob, dropout_scale
        FROM hps.layer
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No layer for layer_id="+str(layer_id))
        return row
    def select_layer_linear(self, layer_id):
        row = self.db.executeSQL("""
        SELECT init_id, init_bias, 
               W_lr_scale, b_lr_scale,
               max_row_norm, max_col_norm
        FROM hps.layer_linear
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No linear layer for layer_id="+str(layer_id))
        return row
    def select_layer_softmax(self, layer_id):
        row = self.db.executeSQL("""
        SELECT init_id, init_bias, 
               W_lr_scale, b_lr_scale,
               max_row_norm, max_col_norm
        FROM hps.layer_softmax
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No softmax layer for layer_id="+str(layer_id))
        return row
    def select_layer_rectifiedlinear(self, layer_id):
        row = self.db.executeSQL("""
        SELECT init_id, init_bias, 
               W_lr_scale, b_lr_scale,
               max_row_norm, max_col_norm,
               left_slope
        FROM hps.layer_rectifiedlinear
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No rectifiedlinear layer for layer_id="\
                +str(layer_id))
        return row
    def select_layer_softmaxpool(self, layer_id):
        row = self.db.executeSQL("""
        SELECT detector_layer_dim	, pool_size,
	         init_id, init_bias,
	         W_lr_scale, b_lr_scale
        FROM hps.layer_softmaxpool
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No softmaxpool layer for layer_id="+str(layer_id))
        return row
    def select_layer_convrectifiedlinear(self, layer_id):
        row = self.db.executeSQL("""
        SELECT output_channels, 
                kernel_shape1, kernel_shape2, 
                pool_shape1, pool_shape2,
                pool_stride, border_mode,
                init_bias, W_lr_scale,
                b_lr_scale, left_slope,
                max_kernel_norm,
        FROM hps.layer_convrectifiedlinear
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No convrectifiedlinear layer for layer_id=" \
                +str(layer_id))
        return row
    def select_init(self, init_id):
        row = self.db.executeSQL("""
        SELECT init_class
        FROM hps.init
        WHERE init_id = %s
        """, (init_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No init weights for init_id="+str(init_id))
        return row[0]
    def select_init_uniform(self, init_id):
        row = self.db.executeSQL("""
        SELECT init_range
        FROM hps.init_uniform
        WHERE init_id = %s
        """, (init_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No init_uniform for init_id="+str(init_id))
        return row[0]
    def select_init_normal(self, init_id):
        row = self.db.executeSQL("""
        SELECT init_stdev
        FROM hps.init_normal
        WHERE init_id = %s
        """, (init_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No init_normal for init_id="+str(init_id))
        return row[0]
    def select_init_sparse(self, init_id):
        row = self.db.executeSQL("""
        SELECT init_sparseness, init_stdev
        FROM hps.init_sparse
        WHERE init_id = %s
        """, (init_id,), self.db.FETCH_ONE)  
        if not row or row is None:
            raise HPSData("No init_sparse for init_id="+str(init_id))
        return row
    def select_preprocess(self, preprocess_id):
        row =  self.db.executeSQL("""
        SELECT dataset_desc, dataset_nvis
        FROM hps.dataset
        WHERE dataset_id = %s
        """, (preprocess_id,), self.db.FETCH_ONE)

        
    
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
        train_ddm = ContestDataset(which_set='train',
                    base_path = '../data',
                    start = 0,
                    stop = 3584,
                    preprocessor = Standardize())
           
        valid_ddm = get_valid_ddm()
        experiment_id = int(sys.argv[2])
        start_config_id = None
        if len(sys.argv) > 3:
            start_config_id = int(sys.argv[3])
        hps = HPS(experiment_id, train_ddm, valid_ddm)
        hps.run(start_config_id)
            
    elif sys.argv[1] == 'validate':
        validate(sys.argv[2])
    else:
        print """Usage: python main.py train "experiment_id" ["config_id"]
                    or
                        python main.py validate "path/to/model.pkl"
              """
        
