# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:56:19 2013

@author: Nicholas Léonard
"""

import time

from pylearn2.utils import serial
import theano.tensor as T
from theano import config
import numpy as np
from theano import function

#from pylearn2.utils.iteration import (
#    FiniteDatasetIterator,
#    resolve_iterator_class
#)

#from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from pylearn2.training_algorithms.sgd import SGD, MomentumAdjustor
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train import Train
from pylearn2.costs.cost import MethodCost, SumOfCosts
from pylearn2.models.mlp import MLP, ConvRectifiedLinear, RectifiedLinear, \
    Softmax, WeightDecay, Sigmoid
from pylearn2.models.maxout import Maxout, MaxoutConvC01B
from pylearn2.monitor import Monitor
from pylearn2.space import VectorSpace, Conv2DSpace
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
#from pylearn2.datasets.preprocessing import Standardize
from pylearn2.train_extensions import TrainExtension

from database import DatabaseHandler

class ExponentialDecayOverEpoch(TrainExtension):
    """
    This is a callback for the SGD algorithm rather than the Train object.
    This anneals the learning rate by dividing by decay_factor after each
    epoch. It will not shrink the learning rate beyond min_lr.
    """
    def __init__(self, decay_factor, min_lr):
        if isinstance(decay_factor, str):
            decay_factor = float(decay_factor)
        if isinstance(min_lr, str):
            min_lr = float(min_lr)
        assert isinstance(decay_factor, float)
        assert isinstance(min_lr, float)
        self.__dict__.update(locals())
        del self.self
        self._count = 0
    def on_monitor(self, model, dataset, algorithm):
        if self._count == 0:
            self._cur_lr = algorithm.learning_rate.get_value()
        self._count += 1
        self._cur_lr = max(self._cur_lr * self.decay_factor, self.min_lr)
        algorithm.learning_rate.set_value(np.cast[config.floatX](self._cur_lr))

    def __call__(self, algorithm):
        if self._count == 0:
            self._base_lr = algorithm.learning_rate.get_value()
        self._count += 1
        cur_lr = self._base_lr / (self.decay_factor ** self._count)
        new_lr = max(cur_lr, self.min_lr)
        new_lr = np.cast[config.floatX](new_lr)
        algorithm.learning_rate.set_value(new_lr)   

"""
TODO:
    add affineless softmax layer for Ian
    add preprocessing
    add worker_id to log (get from sequence)
    saves best model on disk. At end of experiment, saves best model
    to database as blob.
    make configuration Class 
"""

class HPSModelPersist(TrainExtension):
    pass

class HPSLog(TrainExtension):
    """
    A callback that saves a copy of the model every time it achieves
    a new minimal value of a monitoring channel.
    """
    def __init__(self, channel_names, db, config_id):
        """
        Parameters
        ----------
        channel_names: the name of the channels we want to persist to db
        db: the DatabaseHandler
        """

        self.config_id = config_id
        self.channel_names = channel_names
        self.db = db
        self.epoch_count = 0


    def on_monitor(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier. If it's the
        case, saves the model.

        Parameters
        ----------
        model : pylearn2.models.model.Model
                model.monitor must contain a channel with name given by 
                self.channel_name
        dataset : pylearn2.datasets.dataset.Dataset
            not used
        algorithm : TrainingAlgorithm
            not used
        """
        print "config_id: ", self.config_id
        monitor = model.monitor
        channels = monitor.channels
        
        for channel_name in self.channel_names:
            channel = channels[channel_name]
            val_record = channel.val_record
            channel_value = float(val_record[-1])
            self.set_training_log(channel_name, channel_value)

        self.epoch_count += 1
        
    def set_training_log(self, channel_name, channel_value):
        self.db.executeSQL("""
        INSERT INTO hps2.training_log (config_id, epoch_count, channel_name, 
                                      channel_value)
        VALUES (%s, %s, %s, %s)
        """, 
        (self.config_id, self.epoch_count, channel_name, channel_value),
        self.db.COMMIT)
            
class HPSData( Exception ): pass

class HPS:
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
    def __init__(self, 
                 dataset_name,
                 task_id,
                 train_ddm, valid_ddm, 
                 log_channel_names,
                 test_ddm = None,
                 save_prefix = "model_",
                 mbsb_channel_name = None):
        self.dataset_name = dataset_name
        self.task_id = task_id
        
        self.train_ddm = train_ddm
        self.valid_ddm = valid_ddm
        self.test_ddm = test_ddm
        self.monitoring_dataset = {'train': train_ddm}
        
        self.nvis = self.train_ddm.get_design_matrix().shape[1]
        self.nout = self.train_ddm.get_targets().shape[1]
        self.ntrain = self.train_ddm.get_design_matrix().shape[0]
        self.nvalid = self.valid_ddm.get_design_matrix().shape[0]
        self.ntest = 0
        if self.test_ddm is not None:
            self.ntest = self.test_ddm.get_design_matrix().shape[0]
        
        self.log_channel_names = log_channel_names
        self.save_prefix = save_prefix
        # TODO store this in data for each experiment or dataset
        self.mbsb_channel_name = mbsb_channel_name
        
        print "nvis, nout :", self.nvis, self.nout
        print "ntrain :", self.ntrain
        print "nvalid :", self.nvalid
        
    def run(self, start_config_id = None):
        self.db = DatabaseHandler()
        print 'running'
        while True:
            
            (config_id, config_class, model, learner, algorithm) \
                = self.get_config(start_config_id)
            start_config_id = None
            print 'learning'     
            learner.main_loop()
            
            self.set_end_time(config_id)
    def get_config(self, start_config_id = None):
        if start_config_id is not None:
            (config_id,config_class,random_seed,ext_array) \
                = self.select_config(start_config_id)
        else:
            (config_id,config_class,random_seed,ext_array) \
                = self.select_next_config()
        # model (could also return Cost)
        (weight_decay, model, batch_size) \
            = self.get_model(config_id, config_class)
        
        # prepare monitor
        self.prep_valtest_monitor(model, batch_size)
        
        # extensions
        extensions = self.get_extensions(ext_array, config_id)
        
        costs = [MethodCost(method='cost_from_X', supervised=True)]
        if weight_decay is not None:
            costs.append(WeightDecay(coeffs=weight_decay))
        if len(costs) > 1:
            cost = SumOfCosts(costs)
        else:
            cost = costs[0]
    
        # training algorithm
        algorithm = self.get_trainingAlgorithm(config_id, config_class, cost)
        
        print 'sgd complete'
        learner = Train(dataset=self.train_ddm,
                        model=model,
                        algorithm=algorithm,
                        extensions=extensions)
        return (config_id, config_class, model, learner, algorithm)
    def get_classification_accuracy(self, model, minibatch, target):
        Y = model.fprop(minibatch, apply_dropout=False)
        return T.mean(T.cast(T.eq(T.argmax(Y, axis=1), 
                               T.argmax(target, axis=1)), dtype='int32'),
                               dtype=config.floatX)
    def prep_valtest_monitor(self, model, batch_size):
        if self.topo_view:
            print "topo view"
            minibatch = T.as_tensor_variable(
                            self.valid_ddm.get_batch_topo(batch_size), 
                            name='minibatch'
                        )
        else:
            print "design view"
            minibatch = T.as_tensor_variable(
                            self.valid_ddm.get_batch_design(batch_size), 
                            name='minibatch'
                        )
        target = T.matrix('target')
        Accuracy = self.get_classification_accuracy(model, minibatch, target)           
        monitor = Monitor.get_monitor(model)
        
        monitor.add_dataset(self.valid_ddm, 'sequential', batch_size)
        monitor.add_channel("Validation Classification Accuracy",
                            (minibatch, target),
                            Accuracy,
                            self.valid_ddm)
        monitor.add_channel("Validation Missclassification",
                            (minibatch, target),
                            1.0-Accuracy,
                            self.valid_ddm)
                            
        if self.test_ddm is not None:
            monitor.add_dataset(self.test_ddm, 'sequential', batch_size)
            monitor.add_channel("Test Classification Accuracy",
                                (minibatch, target),
                                Accuracy,
                                self.test_ddm)
                                
    def get_trainingAlgorithm(self, config_id, config_class, cost):
        if 'sgd' in config_class:
            (learning_rate,batch_size,init_momentum,train_iteration_mode) \
                = self.select_train_sgd(config_id)
            num_train_batch = (self.ntrain/batch_size)
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
    def get_model(self, config_id, config_class):
        if 'mlp' in config_class:
            (layer_array,batch_size,input_space_id,dropout_include_probs,
                dropout_scales,dropout_input_include_prob,
                 dropout_input_scale,weight_decay,nvis) \
                    = self.select_model_mlp(config_id)
            input_space = None
            self.topo_view = False
            if input_space_id is not None:
                input_space = self.get_space(input_space_id)
                if isinstance(input_space, Conv2DSpace):
                    self.topo_view = True
                assert nvis is None
            if (input_space_id is None) and (nvis is None):
                # default to training set nvis
                nvis = self.nvis
            layers = []
            for layer_id in layer_array:
                layer = self.get_layer(layer_id)
                layers.append(layer)
            # output layer is always called "output":
            layers[-1].layer_name = "output"
            # create MLP:
            model = MLP(layers=layers,
                        input_space=input_space,nvis=nvis,
                        batch_size=batch_size,
                        dropout_include_probs=dropout_include_probs,
                        dropout_scales=dropout_scales,
                        dropout_input_include_prob=dropout_input_include_prob,
                        dropout_input_scale=dropout_input_scale)   
            print 'mlp is built'
            return (weight_decay, model, batch_size)
    def get_layer(self, layer_id):
        """Creates a Layer instance from its definition in the database."""
        (layer_class, layer_name) = self.select_layer(layer_id)
        if layer_class == 'maxout':
            (num_units,num_pieces,pool_stride,randomize_pools,irange,
                 sparse_init,sparse_stdev,include_prob,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm, max_row_norm) \
                     = self.select_layer_maxout(layer_id)
            return Maxout(num_units=num_units,num_pieces=num_pieces,
                           pool_stride=pool_stride,layer_name=layer_name,
                           randomize_pools=randomize_pools,
                           irange=irange,sparse_init=sparse_init,
                           sparse_stdev=sparse_stdev,
                           include_prob=include_prob,
                           init_bias=init_bias,W_lr_scale=W_lr_scale, 
                           b_lr_scale=b_lr_scale,max_col_norm=max_col_norm,
                           max_row_norm=max_row_norm)
        elif layer_class == 'softmax':
            (n_classes,irange,istdev,sparse_init,W_lr_scale,b_lr_scale, 
                 max_row_norm,no_affine,max_col_norm) \
                     = self.select_layer_softmax(layer_id) 
            return Softmax(n_classes=n_classes,irange=irange,istdev=istdev,
                            sparse_init=sparse_init,W_lr_scale=W_lr_scale,
                            b_lr_scale=b_lr_scale,max_row_norm=max_row_norm,
                            no_affine=no_affine,max_col_norm=max_col_norm,
                            layer_name=layer_name)
        elif layer_class == 'rectifiedlinear':
            (dim,irange,istdev,sparse_init,sparse_stdev,include_prob,
                init_bias,W_lr_scale,b_lr_scale,left_slope,max_row_norm,
                max_col_norm,use_bias)\
                    = self.select_layer_rectifiedlinear(layer_id)
            return RectifiedLinear(dim=dim,irange=irange,istdev=istdev,
                                    sparse_init=sparse_init,
                                    sparse_stdev=sparse_stdev,
                                    include_prob=include_prob,
                                    init_bias=init_bias,
                                    W_lr_scale=W_lr_scale,
                                    b_lr_scale=b_lr_scale,
                                    left_slope=left_slope,
                                    max_row_norm=max_row_norm,
                                    max_col_norm=max_col_norm,
                                    use_bias=use_bias,
                                    layer_name=layer_name)
        elif layer_class == 'convrectifiedlinear':
            (output_channels,kernel_width,pool_width,pool_stride,irange,
                border_mode,sparse_init,include_prob,init_bias,W_lr_scale,
                b_lr_scale,left_slope,max_kernel_norm) \
                    = self.select_layer_convrectifiedlinear(layer_id) 
            return ConvRectifiedLinear(output_channels=output_channels,
                        kernel_shape=(kernel_width, kernel_width),
                        pool_shape=(pool_width, pool_width),
                        pool_stride=(pool_stride, pool_stride),
                        layer_name=layer_name, irange=irange,
                        border_mode=border_mode,sparse_init=sparse_init,
                        include_prob=include_prob,init_bias=init_bias,
                        W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                        left_slope=left_slope,
                        max_kernel_norm=max_kernel_norm)
        elif layer_class == 'maxoutconvc01b':
            (num_channels,num_pieces,kernel_width,pool_width,pool_stride,
                irange	,init_bias,W_lr_scale,b_lr_scale,pad,fix_pool_shape,
                fix_pool_stride,fix_kernel_shape,partial_sum,tied_b,
                max_kernel_norm,input_normalization,output_normalization) \
                    = self.select_layer_maxoutConvC01B(layer_id) 
            return MaxoutConvC01B(layer_name=layer_name,
                                  num_channels=num_channels,
                                  num_pieces=num_pieces,
                                  kernel_shape=(kernel_width,kernel_width),
                                  pool_shape=(pool_width, pool_width),
                                  pool_stride=(pool_stride,pool_stride),
                                  irange=irange,init_bias=init_bias,
                                  W_lr_scale=W_lr_scale,
                                  b_lr_scale=b_lr_scale,pad=pad,
                                  fix_pool_shape=fix_pool_shape,
                                  fix_pool_stride=fix_pool_stride,
                                  fix_kernel_shape=fix_kernel_shape,
                                  partial_sum=partial_sum,tied_b=tied_b,
                                  max_kernel_norm=max_kernel_norm,
                                  input_normalization=input_normalization,
                                  output_normalization=output_normalization)
        elif layer_class == 'sigmoid':
            (dim,irange,istdev,sparse_init,sparse_stdev,include_prob,init_bias,
                W_lr_scale,b_lr_scale,max_col_norm,max_row_norm) \
                    = self.select_layer_sigmoid(layer_id)
            return Sigmoid(layer_name=layer_name,dim=dim,irange=irange,
                           istdev=istdev,
                           sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                           include_prob=include_prob,init_bias=init_bias,
                           W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                           max_col_norm=max_col_norm,
                           max_row_norm=max_row_norm)
        else:
            assert False
    def get_termination(self, config_id, config_class):
        terminations = []
        if 'epochcounter' in config_class:
            print 'epoch_counter'
            max_epochs = self.select_term_epochCounter(config_id)
            terminations.append(EpochCounter(max_epochs))
        if 'monitorbased' in config_class:
            print 'monitor_based'
            (proportional_decrease, max_epochs, channel_name) \
                = self.select_term_monitorBased(config_id)
            terminations.append(
                MonitorBased(
                    prop_decrease = proportional_decrease, 
                    N = max_epochs, channel_name = channel_name
                )
            )
        if len(terminations) > 1:
            return And(terminations)
        elif len(terminations) == 0:
            return None
        return terminations[0]
    def get_space(self, space_id):
        space_class = self.select_space(space_id)
        if space_class == 'conv2dspace':
            (num_row, num_column, num_channels, axes_char) \
                = self.select_space_conv2DSpace(space_id)
            if axes_char == 'b01c':
                axes = ('b', 0, 1, 'c')
            elif axes_char == 'c01b':
                axes = ('c', 0, 1, 'b')
            print axes
            return Conv2DSpace(shape=(num_row, num_column), 
                               num_channels=num_channels, axes=axes)
        else:
            raise HPSData("Space class not supported:"+str(space_class))
    def get_extensions(self, ext_array, config_id):
        if ext_array is None:
            return []
        extensions = []
        for ext_id in ext_array:
            ext_class = self.select_extension(ext_id)
            if ext_class == 'exponentialdecayoverepoch':
                (decay_factor, min_lr) \
                    =  self.select_ext_exponentialDecayOverEpoch(ext_id)
                extensions.append(
                    ExponentialDecayOverEpoch(
                        decay_factor=decay_factor,min_lr=min_lr
                    )
                )
            elif ext_class == 'momentumadjustor':
                (final_momentum, start_epoch, saturate_epoch) \
                    = self.select_ext_momentumAdjustor(ext_id)
                extensions.append(
                    MomentumAdjustor(
                        final_momentum=final_momentum,
                        start=start_epoch, 
                        saturate=saturate_epoch
                    )
                )
            else:
                raise HPSData("ext class not supported:"+str(ext_class))
        # monitor based save best
        if self.mbsb_channel_name is not None:
            save_path = self.save_prefix+str(config_id)+"_best.pkl"
            extensions.append(MonitorBasedSaveBest(
                    channel_name = self.mbsb_channel_name,
                    save_path = save_path
                )
            )
        
        # HPS Logger
        extensions.append(
            HPSLog(self.log_channel_names, self.db, config_id)
        )
        return extensions
    def select_train_sgd(self, config_id):
        row = self.db.executeSQL("""
        SELECT learning_rate,batch_size,init_momentum,train_iteration_mode
        FROM hps2.train_sgd
        WHERE config_id = %s
        """, (config_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No stochasticGradientDescent for config_id=" \
                +str(config_id))
        return row
    def set_end_time(self, config_id):
        return self.db.executeSQL("""
        UPDATE hps2.config 
        SET end_time = now()
        WHERE config_id = %s
        """, (config_id,), self.db.COMMIT)  
    def set_accuracy(self, config_id, accuracy):
        return self.db.executeSQL("""
        INSERT INTO hps2.validation_accuracy (config_id, accuracy)
        VALUES (%s, %s)
        """, (config_id, accuracy), self.db.COMMIT)  
    def select_space(self, space_id):
        row = self.db.executeSQL("""
        SELECT space_class
        FROM hps2.space
        WHERE space_id = %s
        """, (space_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No space for space_id="+str(space_id))
        return row[0]
    def select_space_conv2DSpace(self, space_id):
        row = self.db.executeSQL("""
        SELECT num_row, num_column, num_channel, axes
        FROM hps2.space_conv2DSpace
        WHERE space_id = %s
        """, (space_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No conv2DSpace for space_id="+str(space_id))
        return row
    def select_extension(self, ext_id):
        row = self.db.executeSQL("""
        SELECT ext_class
        FROM hps2.extension
        WHERE ext_id = %s
        """, (ext_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No extension for ext_id="+str(ext_id))
        return row[0]
    def select_ext_exponentialDecayOverEpoch(self, ext_id):
        row = self.db.executeSQL("""
        SELECT decay_factor, min_lr
        FROM hps2.ext_exponentialDecayOverEpoch
        WHERE ext_id = %s
        """, (ext_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No exponentialDecayOverEpoch ext for ext_id=" \
                +str(ext_id))
        return row
    def select_ext_momentumAdjustor(self, ext_id):
        row = self.db.executeSQL("""
        SELECT final_momentum, start_epoch, saturate_epoch
        FROM hps2.ext_momentumAdjustor
        WHERE ext_id = %s
        """, (ext_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No momentumAdjustor extension for ext_id=" \
                +str(ext_id))
        return row
    def select_next_config(self):
        row = None
        for i in xrange(10):
            c = self.db.conn.cursor()
            c.execute("""
            BEGIN;
    
            SELECT config_id,config_class,random_seed,ext_array
            FROM hps2.config 
            WHERE start_time IS NULL AND task_id = %s
            LIMIT 1 FOR UPDATE;
            """, (self.task_id,))
            row = c.fetchone()
            if row is not None and row:
                break
            time.sleep(0.1)
            c.close()
        if not row or row is None:
            raise HPSData("No more configurations for task_id=" \
                +str(self.task_id)+" "+row)
        (config_id,config_class,random_seed,ext_array) = row
        c.execute("""
        UPDATE hps2.config
        SET start_time = now() 
        WHERE config_id = %s;
        """, (config_id,))
        self.db.conn.commit()
        c.close()
        return (config_id,config_class,random_seed,ext_array)
    def select_config(self, config_id):
        row = None
        for i in xrange(10):
            c = self.db.conn.cursor()
            c.execute("""
            BEGIN;
    
            SELECT config_id,config_class,random_seed,ext_array
            FROM hps2.config 
            WHERE config_id = %s 
            LIMIT 1 FOR UPDATE;
            """, (config_id,))
            row = c.fetchone()
            if row is not None and row:
                break
            time.sleep(0.1)
            c.close()
        if not row or row is None:
            raise HPSData("No more configurations for config_id=" \
                +str(config_id)+", row:"+str(row))
        (config_id,config_class,random_seed,ext_array) = row
        c.execute("""
        UPDATE hps2.config
        SET start_time = now() 
        WHERE config_id = %s;
        """, (config_id,))
        self.db.conn.commit()
        c.close()
        return (config_id,config_class,random_seed,ext_array)
    def select_model_mlp(self, config_id):
        row = self.db.executeSQL("""
        SELECT  layer_array,batch_size,input_space_id,dropout_include_probs,
                dropout_scales,dropout_input_include_prob,
                 dropout_input_scale,weight_decay,nvis		
        FROM hps2.model_mlp
        WHERE config_id = %s
        """, (config_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No model mlp for config_id="+str(config_id))
        return row
    def select_layer(self, layer_id):
        row = self.db.executeSQL("""
        SELECT layer_class, layer_name
        FROM hps2.layer
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No layer for layer_id="+str(layer_id))
        return row
    def select_layer_maxout(self, layer_id):
        row = self.db.executeSQL("""
        SELECT   num_units,num_pieces,pool_stride,randomize_pools,irange,
                 sparse_init,sparse_stdev,include_prob,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm,max_row_norm
        FROM hps2.layer_maxout
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No maxout layer for layer_id="+str(layer_id))
        return row
    def select_layer_softmax(self, layer_id):
        row = self.db.executeSQL("""
        SELECT  n_classes,irange,istdev,sparse_init,W_lr_scale,b_lr_scale, 
                max_row_norm,no_affine,max_col_norm
        FROM hps2.layer_softmax
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No softmax layer for layer_id="+str(layer_id))
        return row
    def select_layer_rectifiedlinear(self, layer_id):
        row = self.db.executeSQL("""
        SELECT  dim,irange,istdev,sparse_init,sparse_stdev,include_prob,
                init_bias,W_lr_scale,b_lr_scale,left_slope,max_row_norm,
                max_col_norm,use_bias
        FROM hps2.layer_rectifiedlinear
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No rectifiedlinear layer for layer_id="\
                +str(layer_id))
        return row
    def select_layer_convrectifiedlinear(self, layer_id):
        row = self.db.executeSQL("""
        SELECT  output_channels,kernel_width,pool_width,pool_stride,irange,
                border_mode,sparse_init,include_prob,init_bias,W_lr_scale,
                b_lr_scale,left_slope,max_kernel_norm
        FROM hps2.layer_convrectifiedlinear
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No convrectifiedlinear layer for layer_id=" \
                +str(layer_id))
        return row
    def select_layer_maxoutConvC01B(self, layer_id):
        row = self.db.executeSQL("""
        SELECT  num_channels,num_pieces,kernel_width,pool_width,pool_stride,
                irange	,init_bias,W_lr_scale,b_lr_scale,pad,fix_pool_shape,
                fix_pool_stride,fix_kernel_shape,partial_sum,tied_b,
                max_kernel_norm,input_normalization,output_normalization
        FROM hps2.layer_maxoutConvC01B
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No maxoutConvC01B layer for layer_id=" \
                +str(layer_id))
        return row
    def select_layer_sigmoid(self, layer_id):
        row = self.db.executeSQL("""
        SELECT  dim,irange,istdev,sparse_init,sparse_stdev,include_prob,init_bias,
                W_lr_scale,b_lr_scale,max_col_norm,max_row_norm
        FROM hps2.layer_sigmoid
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No sigmoid layer for layer_id=" \
                +str(layer_id))
        return row
    def select_term_epochCounter(self, config_id):
        row = self.db.executeSQL("""
        SELECT ec_max_epoch
        FROM hps2.term_epochcounter
        WHERE config_id = %s
        """, (config_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No epochCounter term for config_id="\
                +str(config_id))
        return row[0]
    def select_term_monitorBased(self, config_id):
        row = self.db.executeSQL("""
        SELECT proportional_decrease, mb_max_epoch, channel_name
        FROM hps2.term_monitorBased
        WHERE config_id = %s
        """, (config_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No monitorBased term for config_id="\
                +str(config_id))
        return row
    def select_preprocess(self, preprocess_id):
        row =  self.db.executeSQL("""
        SELECT dataset_desc, dataset_nvis
        FROM hps2.dataset
        WHERE dataset_id = %s
        """, (preprocess_id,), self.db.FETCH_ONE)
