# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:56:19 2013

@author: Nicholas Léonard
"""

import time, sys

from pylearn2.utils import serial
from itertools import izip
from pylearn2.utils import safe_zip
from collections import OrderedDict
from pylearn2.utils import safe_union

import numpy as np

from theano.gof.op import get_debug_values
from theano.printing import Print
from theano import function
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T
import theano

from pylearn2.linear.matrixmul import MatrixMul

from pylearn2.models.model import Model

from pylearn2.utils import sharedX

from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d

from pylearn2.training_algorithms.sgd import SGD, MomentumAdjustor
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train import Train
from pylearn2.costs.cost import SumOfCosts, Cost
from pylearn2.costs.mlp import WeightDecay, L1WeightDecay
from pylearn2.models.mlp import MLP, ConvRectifiedLinear, \
    RectifiedLinear, Softmax, Sigmoid, Linear, Tanh, max_pool_c01b, \
    max_pool, Layer
from pylearn2.models.maxout import Maxout, MaxoutConvC01B
from pylearn2.monitor import Monitor
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace, Space
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.train_extensions import TrainExtension
from pylearn2.datasets import preprocessing as pp
from pylearn2.datasets.cifar100 import CIFAR100

from database import DatabaseHandler

def batched_tensordot(x, y, axes=2):
    """
    :param x: A Tensor with sizes e.g.: for  3D (dim1, dim3, dim2)
    :param y: A Tensor with sizes e.g.: for 3D (dim1, dim2, dim4)
    :param axes: an integer or array. If an integer, the number of axes
                 to sum over. If an array, it must have two array
                 elements containing the axes to sum over in each tensor.

                 Note that the default value of 2 is not guaranteed to work
                 for all values of a and b, and an error will be raised if
                 that is the case. The reason for keeping the default is to
                 maintain the same signature as numpy's tensordot function
                 (and np.tensordot raises analogous errors for non-compatible
                 inputs).

                 If an integer i, it is converted to an array containing
                 the last i dimensions of the first tensor and the first
                 i dimensions of the second tensor:
                     axes = [range(a.ndim - i, b.ndim), range(i)]

                 If an array, its two elements must contain compatible axes
                 of the two tensors. For example, [[1, 2], [2, 0]] means sum
                 over the 2nd and 3rd axes of a and the 3rd and 1st axes of b.
                 (Remember axes are zero-indexed!) The 2nd axis of a and the
                 3rd axis of b must have the same shape; the same is true for
                 the 3rd axis of a and the 1st axis of b.
    :type axes: int or array-like of length 2
    This function computes the tensordot product between the two tensors, by
    iterating over the first dimension using scan.
    Returns a tensor of size e.g. if it is 3D: (dim1, dim3, dim4)
    Example:
    >>> first = T.tensor3('first')
    >>> second = T.tensor3('second')
    >>> result = batched_dot(first, second)
    :note:  This is a subset of numpy.einsum, but we do not provide it for now.
    But numpy einsum is slower than dot or tensordot:
    http://mail.scipy.org/pipermail/numpy-discussion/2012-October/064259.html
    """
    if isinstance(axes, list):
        axes = np.asarray(axes)-1
    elif isinstance(axes, np.ndarray):
        axes -= 1
                
    result, updates = theano.scan(fn=lambda x_mat, y_mat:
            theano.tensor.tensordot(x_mat, y_mat, axes),
            outputs_info=None,
            sequences=[x, y],
            non_sequences=None)
    return result
    
class MLPCost(Cost):
    supervised = True
    def __init__(self, cost_type='default', missing_target_value=None):
        self.__dict__.update(locals())
        del self.self
        self.use_dropout = False
    
    def setup_dropout(self, default_input_include_prob=.5, 
                        input_scales=None, input_include_probs=None, 
                        default_input_scale=2.):
        """
        During training, each input to each layer is randomly included or excluded
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for each
        layer's input scale is determined by the same scheme as the input probabilities.
        """

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self
        
        self.use_dropout=True

    def get_gradients(self, model, X, Y=None, ** kwargs):
        """
        model: a pylearn2 Model instance
        X: a batch in model.get_input_space()
        Y: a batch in model.get_output_space()

        returns: gradients, updates
            gradients:
                a dictionary mapping from the model's parameters
                         to their gradients
                The default implementation is to compute the gradients
                using T.grad applied to the value returned by __call__.
                However, subclasses may return other values for the gradient.
                For example, an intractable cost may return a sampling-based
                approximation to its gradient.
            updates:
                a dictionary mapping shared variables to updates that must
                be applied to them each time these gradients are computed.
                This is to facilitate computation of sampling-based approximate
                gradients.
                The parameters should never appear in the updates dictionary.
                This would imply that computing their gradient changes
                their value, thus making the gradient value outdated.
        """

        try:
            if Y is None:
                cost = self(model=model, X=X, **kwargs)
            else:
                cost = self(model=model, X=X, Y=Y, **kwargs)
        except TypeError,e:
            # If anybody knows how to add type(seslf) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling "+str(type(self))+".__call__"
            print str(type(self))
            print e.message
            raise e

        if cost is None:
            raise NotImplementedError(str(type(self))+" represents an intractable "
                    " cost and does not provide a gradient approximation scheme.")

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'raise')

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates
        
    def __call__(self, model, X, Y, ** kwargs):
        if self.use_dropout:
            Y_hat = model.dropout_fprop(X, default_input_include_prob=self.default_input_include_prob,
                    input_include_probs=self.input_include_probs, default_input_scale=self.default_input_scale,
                    input_scales=self.input_scales
                    )
        else:
            Y_hat = model.fprop(X)
        
        if self.missing_target_value is not None:
            assert (self.cost_type == 'default')
            costMatrix = model.layers[-1].cost_matrix(Y, Y_hat)
            costMatrix *= T.neq(Y, -1)  # This sets to zero all elements where Y == -1
            cost = model.cost_from_cost_matrix(costMatrix)
        else:
            if self.cost_type == 'default':
                cost = model.cost(Y, Y_hat)
            elif self.cost_type == 'nll':
                cost = (-Y * T.log(Y_hat)).sum(axis=1).mean()
            elif self.cost_type == 'crossentropy':
                cost = (-Y * T.log(Y_hat) - (1 - Y) \
                    * T.log(1 - Y_hat)).sum(axis=1).mean()
            else:
                raise NotImplementedError()
        return cost
        
    def get_test_cost(self, model, X, Y):
        use_dropout = self.use_dropout
        self.use_dropout = False
        cost = self.__call__(model, X, Y)
        self.use_dropout = use_dropout
        return cost

class Convolution(Layer):
    """
        WRITEME
    """

    def __init__(self,
                 output_channels,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 activation_function = 'tanh',
                 irange = None,
                 border_mode = 'valid',
                 sparse_init = None,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_kernel_norm = None,
                 pool_type = 'max',
                 detector_normalization = None,
                 output_normalization = None):
        """

            include_prob: probability of including a weight element in the set
            of weights initialized to U(-irange, irange). If not included
            it is initialized to 0.

        """
        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space
        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [self.input_space.shape[0] - self.kernel_shape[0] + 1,
                self.input_space.shape[1] - self.kernel_shape[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [self.input_space.shape[0] + self.kernel_shape[0] - 1,
                    self.input_space.shape[1] + self.kernel_shape[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                num_channels = self.output_channels,
                axes = ('b', 'c', 0, 1))

        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = conv2d.make_random_conv2D(
                    irange = self.irange,
                    input_space = self.input_space,
                    output_space = self.detector_space,
                    kernel_shape = self.kernel_shape,
                    batch_size = self.mlp.batch_size,
                    subsample = (1,1),
                    border_mode = self.border_mode,
                    rng = rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                    num_nonzero = self.sparse_init,
                    input_space = self.input_space,
                    output_space = self.detector_space,
                    kernel_shape = self.kernel_shape,
                    batch_size = self.mlp.batch_size,
                    subsample = (1,1),
                    border_mode = self.border_mode,
                    rng = rng)
        W, = self.transformer.get_params()
        W.name = 'W'

        self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
        self.b.name = 'b'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape

        if self.mlp.batch_size is None:
            raise ValueError("Tried to use a convolutional layer with an MLP that has "
                    "no batch size specified. You must specify the batch size of the "
                    "model because theano requires the batch size to be known at "
                    "graph construction time for convolution.")

        assert self.pool_type in ['max', 'mean']

        dummy_detector = sharedX(self.detector_space.get_origin_batch(self.mlp.batch_size))
        if self.pool_type == 'max':
            dummy_p = max_pool(bc01=dummy_detector, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape)
        elif self.pool_type == 'mean':
            dummy_p = mean_pool(bc01=dummy_detector, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape)
        dummy_p = dummy_p.eval()
        self.output_space = Conv2DSpace(shape=[dummy_p.shape[2], dummy_p.shape[3]],
                num_channels = self.output_channels, axes = ('b', 'c', 0, 1) )

        print 'Output space: ', self.output_space.shape



    def censor_updates(self, updates):

        if self.max_kernel_norm is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1,2,3)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x', 'x', 'x')


    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * abs(W).sum()

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_topo(self):
        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp,rows,cols,inp))

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1,2,3)))

        return OrderedDict([
                            ('kernel_norms_min'  , row_norms.min()),
                            ('kernel_norms_mean' , row_norms.mean()),
                            ('kernel_norms_max'  , row_norms.max()),
                            ])

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        if self.activation_function is None:
            d = z
        elif self.activation_function == 'tanh':
            d = T.tanh(z)
        elif self.activation_function == 'sigmoid':
            d = T.nnet.sigmoid(z)
        elif self.activation_function == 'softmax':
            d = T.nnet.softmax(z)
        else:
            raise NotImplementedError()

        self.detector_space.validate(d)

        if not hasattr(self, 'detector_normalization'):
            self.detector_normalization = None

        if self.detector_normalization:
            d = self.detector_normalization(d)

        assert self.pool_type in ['max', 'mean']
        if self.pool_type == 'max':
            p = max_pool(bc01=d, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape)
        elif self.pool_type == 'mean':
            p = mean_pool(bc01=d, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape)

        self.output_space.validate(p)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p

class ExponentialDecayOverEpoch(TrainExtension):
    """
    This is a callback for the SGD algorithm rather than the Train obj.
    This anneals the lr by dividing by decay_factor after each
    epoch. It will not shrink the learning rate beyond 
    min_lr_scale*learning_rate.
    """
    def __init__(self, decay_factor, min_lr_scale):
        if isinstance(decay_factor, str):
            decay_factor = float(decay_factor)
        if isinstance(min_lr_scale, str):
            min_lr_scale = float(min_lr_scale)
        assert isinstance(decay_factor, float)
        assert isinstance(min_lr_scale, float)
        self.__dict__.update(locals())
        del self.self
        self._count = 0
        
    def on_monitor(self, model, dataset, algorithm):
        if self._count == 0:
            self._cur_lr = algorithm.learning_rate.get_value()
            self.min_lr = self._cur_lr*self.min_lr_scale
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
    saves best model on disk. At end of experiment, saves best model
        to database as blob.
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
        INSERT INTO hps3.training_log (config_id, epoch_count, 
                                        channel_name, channel_value)
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
    
    Curriculum:
        start with random jobs for very few epochs (30) each. 
    """
    
    def __init__(self, 
                 worker_name,
                 task_id,
                 base_channel_names = ['train_objective'],
                 save_prefix = "model_",
                 mbsb_channel_name = 'valid_hps_cost',
                 cache_dataset = True):
                 
        self.worker_name = worker_name
        self.cache_dataset = cache_dataset
        self.task_id = task_id
        self.dataset_cache = {}
        
        self.base_channel_names = base_channel_names
        self.save_prefix = save_prefix
        # TODO store this in data for each experiment or dataset
        self.mbsb_channel_name = mbsb_channel_name
        self.db = DatabaseHandler()
        
    def run(self, start_config_id = None):
        print 'running'
        while True:
            try:   
                (config_id, model, learner, algorithm) \
                    = self.get_config(start_config_id)
                start_config_id = None
                print 'learning'
                #learner.main_loop()
                learner.main_loop()
            except Exception, e:
                print e
            self.set_end_time(config_id)
            
    def get_config(self, start_config_id = None):
        if start_config_id is not None:
            (config_id,random_seed,ext_array,dataset_id,channel_array) \
                    = self.select_config(start_config_id)
        else:
            (config_id,random_seed,ext_array,dataset_id,channel_array) \
                    = self.select_next_config()

        # dataset
        self.load_dataset(dataset_id)
        
        # model
        self.load_model(config_id)
        
        # monitor:
        self.setup_monitor()
        
        # training algorithm
        algorithm = self.get_train(config_id)
        
        # extensions
        extensions = self.get_extensions(ext_array, config_id)
        
        # channels
        self.setup_channels(channel_array)
        
        # learner 
        learner = Train(dataset=self.train_ddm,
                        model=self.model,
                        algorithm=algorithm,
                        extensions=extensions)
                        
        return (config_id, self.model, learner, algorithm)
                
    def apply_preprocess(self):
        if self.preprocessor is None:
            return
        self.preprocessor.apply(self.train_ddm, can_fit=True)
        self.preprocessor.apply(self.valid_ddm, can_fit=False)
        if self.test_ddm is not None:
            self.preprocessor.apply(self.test_ddm, can_fit=False)
            
    def setup_channels(self, channel_array):
        if channel_array is None:
            return
        for channel_id in channel_array:
            row = self.db.executeSQL("""
            SELECT channel_class, monitoring_datasets
            FROM hps3.channel
            WHERE channel_id = %s
            """, (channel_id,), self.db.FETCH_ONE)
            if not row or row is None:
                raise HPSData("No channel for ext_id="+str(ext_id))
            channel_class, monitoring_datasets = row
            fn = getattr(self, 'setup_channel_'+channel_class)
            fn(channel_id,monitoring_datasets)
     
    def setup_channel_mca(self, channel_id, monitoring_datasets):
        """mean classification accuracy"""
        Y = self.model.fprop(self.minibatch)
        MCA = T.mean(T.cast(T.eq(T.argmax(Y, axis=1), 
                       T.argmax(self.target, axis=1)), dtype='int32'),
                       dtype=config.floatX)
        self.add_channel('mca',MCA,monitoring_datasets)
          
    def setup_channel_crossentropy(self,channel_id,monitoring_datasets):
        """cross-entropy"""
        Y_hat = self.model.fprop(self.minibatch)
        CE = (-self.target * T.log(Y_hat) - (1 - self.target) \
                * T.log(1 - Y_hat)).sum(axis=1).mean()
        self.add_channel('crossentropy', CE, monitoring_datasets)
                       
    def setup_monitor(self):
        if self.topo_view:
            print "topo view"
            self.minibatch = T.as_tensor_variable(
                        self.valid_ddm.get_batch_topo(self.batch_size), 
                        name='minibatch'
                    )
        else:
            print "design view"
            self.minibatch = T.as_tensor_variable(
                        self.valid_ddm.get_batch_design(self.batch_size), 
                        name='minibatch'
                    )
                        
        self.target = T.matrix('target')  
        
        self.monitor = Monitor.get_monitor(self.model)
        self.log_channel_names = []
        self.log_channel_names.extend(self.base_channel_names)
        
        self.monitor.add_dataset(self.valid_ddm, 'sequential', 
                                    self.batch_size)
        if self.test_ddm is not None:
            self.monitor.add_dataset(self.test_ddm, 'sequential', 
                                        self.batch_size)
        
    def add_channel(self, channel_name, tensor_var, 
                    dataset_names=['valid','test']):
        for dataset_name in dataset_names:
            if dataset_name == 'valid':
                ddm = self.valid_ddm
            elif dataset_name == 'test':
                if self.test_ddm is None:
                    continue
                ddm = self.test_ddm
            log_channel_name = dataset_name+'_hps_'+channel_name
            self.log_channel_names.append(log_channel_name)
            self.monitor.add_channel(log_channel_name,
                                (self.minibatch, self.target),
                                tensor_var, ddm)
                                
    def load_dataset(self, dataset_id):
        if dataset_id in self.dataset_cache:
            # if cached, load from cache
            (train_ddm, valid_ddm, test_ddm) \
                = self.dataset_cache[dataset_id]
            self.train_ddm = train_ddm
            self.valid_ddm = valid_ddm
            self.test_ddm = test_ddm
        else:
            row =  self.db.executeSQL("""
            SELECT preprocess_array,train_ddm_id,valid_ddm_id,test_ddm_id
            FROM hps3.dataset
            WHERE dataset_id = %s
            """, (dataset_id,), self.db.FETCH_ONE)
            if not row or row is None:
                raise HPSData("No dataset for dataset_id="\
                    +str(dataset_id))
            (preprocess_array,train_ddm_id,valid_ddm_id,test_ddm_id) \
                = row
            # preprocessing
            self.load_preprocessor(preprocess_array)
            # dense design matrices
            self.train_ddm = self.get_ddm(train_ddm_id)
            self.valid_ddm = self.get_ddm(valid_ddm_id)
            self.test_ddm = self.get_ddm(test_ddm_id)
            self.apply_preprocess()
            
            if self.cache_dataset:
                # cache the dataset for future use
                self.dataset_cache[dataset_id] \
                    = (self.train_ddm, self.valid_ddm, self.test_ddm)
            
        self.monitoring_dataset = {'train': self.train_ddm}
        
        self.nvis = self.train_ddm.get_design_matrix().shape[1]
        print self.train_ddm.get_targets().shape
        self.nout = self.train_ddm.get_targets().shape[1]
        self.ntrain = self.train_ddm.get_design_matrix().shape[0]
        self.nvalid = self.valid_ddm.get_design_matrix().shape[0]
        self.ntest = 0
        if self.test_ddm is not None:
            self.ntest = self.test_ddm.get_design_matrix().shape[0]
        
        print "nvis, nout :", self.nvis, self.nout
        print "ntrain :", self.ntrain
        print "nvalid :", self.nvalid
        
    def get_ddm(self, ddm_id):
        if ddm_id is None:
            return None
        row =  self.db.executeSQL("""
        SELECT ddm_class
        FROM hps3.ddm
        WHERE ddm_id = %s
        """, (ddm_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No ddm for ddm_id=" +str(ddm_id))
        ddm_class = row[0]
        fn = getattr(self, 'get_ddm_'+ddm_class)
        return fn(ddm_id)
            
    def load_preprocessor(self, preprocess_array):
        if preprocess_array is None:
            self.preprocessor = None
            return None
        preprocess_list = []
        for preprocess_id in preprocess_array:
            row =  self.db.executeSQL("""
            SELECT preprocess_class
            FROM hps3.preprocess
            WHERE preprocess_id = %s
            """, (preprocess_id,), self.db.FETCH_ONE)
            if not row or row is None:
                raise HPSData("No preprocess for preprocess_id="\
                    +str(preprocess_id))
            preprocess_class = row[0]
            fn = getattr(self, 'get_preprocess_'+preprocess_class)
            preprocess_list.append(fn(preprocess_id))
        
        if len(preprocess_list) > 1:
            preprocessor = pp.Pipeline(preprocess_list)
        else:
            preprocessor = preprocess_list[0]
        self.preprocessor = preprocessor
    
    def get_costs(self, cost_array):
        costs = []
        for cost_id in cost_array:
            costs.extend(self.get_cost(cost_id))
        
        if len(costs) > 1:
            cost = SumOfCosts(costs)
        else:
            cost = costs[0]
            
        return cost
            
    def get_cost(self, cost_id):
        row = self.db.executeSQL("""
        SELECT cost_class
        FROM hps3.cost
        WHERE cost_id = %s
        """, (cost_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No cost for cost_id="+str(cost_id))
        cost_class = row[0]
        fn = getattr(self, 'get_cost_'+cost_class)
        return fn(cost_id)
    
    def get_cost_mlp(self, cost_id):
        row = self.db.executeSQL("""
        SELECT  cost_type,cost_name,missing_target_value,
                default_dropout_prob,default_dropout_scale
        FROM hps3.cost_mlp
        WHERE cost_id = %s
        """, (cost_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No cost for cost_id="+str(cost_id)) 
        (cost_type,cost_name,missing_target_value,
            default_dropout_prob,default_dropout_scale) = row
        mlp_cost = MLPCost(cost_type=cost_type, 
                            missing_target_value=missing_target_value)
        # default monitor based save best channel:
        test_cost = mlp_cost.get_test_cost(self.model,
                                            self.minibatch,
                                            self.target)
        self.add_channel('cost',test_cost)
        
        if self.dropout:
            mlp_cost.setup_dropout(
                default_input_include_prob=(1.-default_dropout_prob),
                default_input_scale=default_dropout_scale,
                input_scales=self.input_scales,
                input_include_probs=self.input_include_probs)
        
        costs = [mlp_cost]
        if self.weight_decay:
            coeffs = []
            for layer in self.mlp.layers:
                coeffs.append(self.weight_decays[layer.layer_name])
            wd_cost = WeightDecay(coeffs)
            costs.append(wd_cost)
        if self.l1_weight_decay:
            coeffs = []
            for layer in self.mlp.layers:
                coeffs.append(self.l1_weight_decays[layer.layer_name])
            lwd_cost = L1WeightDecay(coeffs)
            costs.append(lwd_cost)
        return costs
        
    def load_model(self, config_id):
        row = self.db.executeSQL("""
        SELECT model_class		
        FROM hps3.model
        WHERE config_id = %s
        """, (config_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No model for config_id="+str(config_id))
        model_class = row[0]
        fn = getattr(self, 'get_model_'+model_class)
        self.model = fn(config_id)
        return self.model
            
    def get_model_mlp(self, config_id):
        self.dropout = False
        self.input_include_probs = {}
        self.input_scales = {}
        self.weight_decay = False
        self.l1_weight_decay = False
        
        row = self.db.executeSQL("""
        SELECT layer_array,input_space_id,batch_size,nvis		
        FROM hps3.model_mlp
        WHERE config_id = %s
        """, (config_id,), self.db.FETCH_ONE)
        
        if not row or row is None:
            raise HPSData("No model mlp for config_id="+str(config_id))
        (layer_array,input_space_id,batch_size,nvis) = row
        self.batch_size = batch_size
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
        i = 0
        for layer_id in layer_array:
            layer = self.get_layer(layer_id,i)
            layers.append(layer)
            i+=1
        # create MLP:
        model = MLP(layers=layers,input_space=input_space,nvis=nvis,
                    batch_size=batch_size)
        self.mlp = model
        return model
        
    def get_layer(self, layer_id, layer_index):
        row = self.db.executeSQL("""
        SELECT  layer_class,layer_name,dropout_scale,
                dropout_probability,weight_decay,l1_weight_decay
        FROM hps3.layer
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No layer for layer_id="+str(layer_id))
        (layer_class,layer_name,dropout_scale,dropout_prob,
            weight_decay,l1_weight_decay) = row
        fn = getattr(self, 'get_layer_'+layer_class)
        if layer_name is None:
            layer_name = layer_class+str(layer_index)
        layer = fn(layer_id, layer_name)
        # per-layer cost function parameters:
        if (dropout_scale is not None):
            self.dropout = True
            self.input_scales[layer_name] = dropout_scale
        if (dropout_prob is not None):
            self.dropout = True
            self.input_include_probs[layer_name] = (1. - dropout_prob)
        if  (weight_decay is not None):
            self.weight_decay = False
            self.weight_decays[layer_name] = weight_decay
        if  (l1_weight_decay is not None):
            self.l1_weight_decay = False
            self.l1_weight_decays[layer_name] = l1_weight_decay
        return layer
            
    def get_terminations(self, config_id, term_array):
        if term_array is None:
            return None
        terminations = []
        for term_id in term_array:
            row = self.db.executeSQL("""
            SELECT term_class
            FROM hps3.termination
            WHERE term_id = %s
            """, (term_id,), self.db.FETCH_ONE)
            if not row or row is None:
                raise HPSData("No term for term_id="+str(term_id))
            term_class = row[0]
            fn = getattr(self, 'get_term_'+term_class)
            terminations.append(fn(term_id))
        if len(terminations) > 1:
            return And(terminations)
        return terminations[0]
        
    def get_axes(self, axes_char):
        if axes_char == 'b01c':
            return ('b', 0, 1, 'c')
        elif axes_char == 'c01b':
            return ('c', 0, 1, 'b')
            
    def get_space(self, space_id):
        row = self.db.executeSQL("""
        SELECT space_class
        FROM hps3.space
        WHERE space_id = %s
        """, (space_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No space for space_id="+str(space_id))
        space_class = row[0]
        fn = getattr(self, 'get_space_'+space_class)
        return fn(space_id)
            
    def get_extensions(self, ext_array, config_id):
        if ext_array is None:
            return []
        extensions = []
        for ext_id in ext_array:
            row = self.db.executeSQL("""
            SELECT ext_class
            FROM hps3.extension
            WHERE ext_id = %s
            """, (ext_id,), self.db.FETCH_ONE)
            if not row or row is None:
                raise HPSData("No extension for ext_id="+str(ext_id))
            ext_class = row[0]
            fn = getattr(self, 'get_ext_'+ext_class)
            extensions.append(fn(ext_id))
        # monitor based save best
        if self.mbsb_channel_name is not None:
            save_path = self.save_prefix+str(config_id)+"_optimum.pkl"
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
    
    def get_train(self, config_id):
        row = self.db.executeSQL("""
        SELECT train_class		
        FROM hps3.train
        WHERE config_id = %s
        """, (config_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No train for config_id="+str(config_id))
        train_class = row[0]
        fn = getattr(self, 'get_train_'+train_class)
        return fn(config_id)
        
    def get_train_sgd(self, config_id):
        row = self.db.executeSQL("""
        SELECT  learning_rate,batch_size,init_momentum,
                train_iteration_mode,cost_array,term_array
        FROM hps3.train_sgd
        WHERE config_id = %s
        """, (config_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No stochasticGradientDescent for config_id="\
                +str(config_id))
        (learning_rate,batch_size,init_momentum,train_iteration_mode, 
            cost_array,term_array) = row
        # cost
        cost = self.get_costs(cost_array)
        
        num_train_batch = (self.ntrain/self.batch_size)
        print "num training batches:", num_train_batch
        termination_criterion \
            = self.get_terminations(config_id, term_array)
        return SGD( learning_rate=learning_rate, cost=cost,
                    batch_size=batch_size,
                    batches_per_iter=num_train_batch,
                    monitoring_dataset=self.monitoring_dataset,
                    termination_criterion=termination_criterion,
                    init_momentum=init_momentum,
                    train_iteration_mode=train_iteration_mode)
        
    def set_end_time(self, config_id):
        return self.db.executeSQL("""
        UPDATE hps3.config 
        SET end_time = now()
        WHERE config_id = %s
        """, (config_id,), self.db.COMMIT)  
    
    def get_space_conv2dspace(self, space_id):
        row = self.db.executeSQL("""
        SELECT num_row, num_column, num_channel, axes
        FROM hps3.space_conv2DSpace
        WHERE space_id = %s
        """, (space_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No conv2DSpace for space_id="+str(space_id))
        (num_row, num_column, num_channels, axes_char) = row
        axes = self.get_axes(axes_char)
        return Conv2DSpace(shape=(num_row, num_column), 
                           num_channels=num_channels, axes=axes)

    def get_ext_exponentialdecayoverepoch(self, ext_id):
        row = self.db.executeSQL("""
        SELECT decay_factor, min_lr_scale
        FROM hps3.ext_exponentialDecayOverEpoch
        WHERE ext_id = %s
        """, (ext_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No exponentialDecayOverEpoch ext for ext_id=" \
                +str(ext_id))
        (decay_factor, min_lr_scale) = row
        return ExponentialDecayOverEpoch(
            decay_factor=decay_factor,min_lr_scale=min_lr_scale
        )
        
    def get_ext_momentumadjustor(self, ext_id):
        row = self.db.executeSQL("""
        SELECT final_momentum, start_epoch, saturate_epoch
        FROM hps3.ext_momentumAdjustor
        WHERE ext_id = %s
        """, (ext_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No momentumAdjustor extension for ext_id=" \
                +str(ext_id))
        (final_momentum, start_epoch, saturate_epoch) = row
        return MomentumAdjustor(
            final_momentum=final_momentum, start=start_epoch, 
            saturate=saturate_epoch
        )
        return row
    def select_next_config(self):
        row = None
        for i in xrange(10):
            c = self.db.conn.cursor()
            c.execute("""
            BEGIN;
    
            SELECT  config_id,random_seed,ext_array,dataset_id,
                    channel_array
            FROM hps3.config 
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
        (config_id,random_seed,ext_array,dataset_id,channel_array) = row
        c.execute("""
        UPDATE hps3.config
        SET start_time = now(), worker_name = %s
        WHERE config_id = %s;
        """, (self.worker_name, config_id,))
        self.db.conn.commit()
        c.close()
        return row
        
    def select_config(self, config_id):
        row = None
        for i in xrange(10):
            c = self.db.conn.cursor()
            c.execute("""
            BEGIN;
            SELECT  config_id,random_seed,ext_array,dataset_id,
                    channel_array
            FROM hps3.config 
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
        (config_id,random_seed,ext_array,dataset_id,channel_array) = row
        c.execute("""
        UPDATE hps3.config
        SET start_time = now(), worker_name = %s
        WHERE config_id = %s;
        """, (self.worker_name, config_id,))
        self.db.conn.commit()
        c.close()
        return row
            
    def get_layer_maxout(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT   num_units,num_pieces,pool_stride,randomize_pools,irange,
                 sparse_init,sparse_stdev,include_prob,init_bias,W_lr_scale,
                 b_lr_scale,max_col_norm,max_row_norm
        FROM hps3.layer_maxout
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No maxout layer for layer_id="+str(layer_id))
        (num_units,num_pieces,pool_stride,randomize_pools,irange,
             sparse_init,sparse_stdev,include_prob,init_bias,W_lr_scale,
             b_lr_scale,max_col_norm, max_row_norm) \
                 = row
        return Maxout(num_units=num_units,num_pieces=num_pieces,
                       pool_stride=pool_stride,layer_name=layer_name,
                       randomize_pools=randomize_pools,
                       irange=irange,sparse_init=sparse_init,
                       sparse_stdev=sparse_stdev,
                       include_prob=include_prob,
                       init_bias=init_bias,W_lr_scale=W_lr_scale, 
                       b_lr_scale=b_lr_scale,max_col_norm=max_col_norm,
                       max_row_norm=max_row_norm)
                       
    def get_layer_softmax(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  n_classes,irange,istdev,sparse_init,W_lr_scale,b_lr_scale, 
                max_row_norm,no_affine,max_col_norm
        FROM hps3.layer_softmax
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No softmax layer for layer_id="+str(layer_id))
        (n_classes,irange,istdev,sparse_init,W_lr_scale,b_lr_scale, 
             max_row_norm,no_affine,max_col_norm) = row
        return Softmax(n_classes=n_classes,irange=irange,istdev=istdev,
                        sparse_init=sparse_init,W_lr_scale=W_lr_scale,
                        b_lr_scale=b_lr_scale,max_row_norm=max_row_norm,
                        no_affine=no_affine,max_col_norm=max_col_norm,
                        layer_name=layer_name)
                        
    def get_layer_rectifiedlinear(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  dim,irange,istdev,sparse_init,sparse_stdev,include_prob,
                init_bias,W_lr_scale,b_lr_scale,left_slope,max_row_norm,
                max_col_norm,use_bias
        FROM hps3.layer_rectifiedlinear
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No rectifiedlinear layer for layer_id="\
                +str(layer_id))
        (dim,irange,istdev,sparse_init,sparse_stdev,include_prob,
            init_bias,W_lr_scale,b_lr_scale,left_slope,max_row_norm,
            max_col_norm,use_bias) = row
        return RectifiedLinear(dim=dim,irange=irange,istdev=istdev,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                include_prob=include_prob,init_bias=init_bias,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                left_slope=left_slope,max_row_norm=max_row_norm,
                max_col_norm=max_col_norm,use_bias=use_bias,
                layer_name=layer_name)
                                
    def get_layer_linear(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  dim,irange,istdev,sparse_init,sparse_stdev,include_prob,
                init_bias,W_lr_scale,b_lr_scale,max_row_norm,
                max_col_norm,softmax_columns
        FROM hps3.layer_linear
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No linear layer for layer_id="\
                +str(layer_id))
        (dim,irange,istdev,sparse_init,sparse_stdev,include_prob,
            init_bias,W_lr_scale,b_lr_scale,max_row_norm,
            max_col_norm,softmax_columns) = row
        return Linear(dim=dim,irange=irange,istdev=istdev,
                sparse_init=sparse_init,sparse_stdev=sparse_stdev,
                include_prob=include_prob,init_bias=init_bias,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_row_norm=max_row_norm,max_col_norm=max_col_norm,
                layer_name=layer_name,softmax_columns=softmax_columns)
                
    def get_layer_convrectifiedlinear(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  output_channels,kernel_width,pool_width,pool_stride,irange,
                border_mode,sparse_init,include_prob,init_bias,W_lr_scale,
                b_lr_scale,left_slope,max_kernel_norm
        FROM hps3.layer_convrectifiedlinear
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No convrectifiedlinear layer for layer_id=" \
                +str(layer_id))
        (output_channels,kernel_width,pool_width,pool_stride,irange,
            border_mode,sparse_init,include_prob,init_bias,W_lr_scale,
            b_lr_scale,left_slope,max_kernel_norm) = row
        return ConvRectifiedLinear(output_channels=output_channels,
                kernel_shape=(kernel_width, kernel_width),
                pool_shape=(pool_width, pool_width),
                pool_stride=(pool_stride, pool_stride),
                layer_name=layer_name, irange=irange,
                border_mode=border_mode,sparse_init=sparse_init,
                include_prob=include_prob,init_bias=init_bias,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                left_slope=left_slope,max_kernel_norm=max_kernel_norm)
                
    def get_layer_convolution(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  output_channels,kernel_width,pool_width,pool_stride,irange,
                border_mode,sparse_init,include_prob,init_bias,W_lr_scale,
                b_lr_scale,max_kernel_norm,activation_function
        FROM hps3.layer_convolution
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No convolution layer for layer_id=" \
                +str(layer_id))
        (output_channels,kernel_width,pool_width,pool_stride,irange,
            border_mode,sparse_init,include_prob,init_bias,W_lr_scale,
            b_lr_scale,max_kernel_norm,activation_function) = row
        return Convolution(output_channels=output_channels,
                kernel_shape=(kernel_width, kernel_width),
                activation_function=activation_function,
                pool_shape=(pool_width, pool_width),
                pool_stride=(pool_stride, pool_stride),
                layer_name=layer_name, irange=irange,
                border_mode=border_mode,sparse_init=sparse_init,
                include_prob=include_prob,init_bias=init_bias,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,
                max_kernel_norm=max_kernel_norm)
                    
    def get_layer_maxoutconvc01b(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  num_channels,num_pieces,kernel_width,pool_width,pool_stride,
                irange,init_bias,W_lr_scale,b_lr_scale,pad,fix_pool_shape,
                fix_pool_stride,fix_kernel_shape,partial_sum,tied_b,
                max_kernel_norm,input_normalization,output_normalization
        FROM hps3.layer_maxoutConvC01B
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No maxoutConvC01B layer for layer_id=" \
                +str(layer_id))
        (num_channels,num_pieces,kernel_width,pool_width,pool_stride,
            irange	,init_bias,W_lr_scale,b_lr_scale,pad,fix_pool_shape,
            fix_pool_stride,fix_kernel_shape,partial_sum,tied_b,
            max_kernel_norm,input_normalization,output_normalization) \
                = self.select_layer_maxoutConvC01B(layer_id) 
        return MaxoutConvC01B(layer_name=layer_name,
                num_channels=num_channels,num_pieces=num_pieces,
                kernel_shape=(kernel_width,kernel_width),
                pool_shape=(pool_width, pool_width),
                pool_stride=(pool_stride,pool_stride),
                irange=irange,init_bias=init_bias,
                W_lr_scale=W_lr_scale,b_lr_scale=b_lr_scale,pad=pad,
                fix_pool_shape=fix_pool_shape,
                fix_pool_stride=fix_pool_stride,
                fix_kernel_shape=fix_kernel_shape,
                partial_sum=partial_sum,tied_b=tied_b,
                max_kernel_norm=max_kernel_norm,
                input_normalization=input_normalization,
                output_normalization=output_normalization)
                              
    def get_layer_sigmoid(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  dim,irange,istdev,sparse_init,sparse_stdev,include_prob,init_bias,
                W_lr_scale,b_lr_scale,max_col_norm,max_row_norm
        FROM hps3.layer_sigmoid
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No sigmoid layer for layer_id=" \
                +str(layer_id))
        (dim,irange,istdev,sparse_init,sparse_stdev,include_prob,init_bias,
            W_lr_scale,b_lr_scale,max_col_norm,max_row_norm) = row
        return Sigmoid(layer_name=layer_name,dim=dim,irange=irange,
                istdev=istdev,sparse_init=sparse_init,
                sparse_stdev=sparse_stdev, include_prob=include_prob,
                init_bias=init_bias,W_lr_scale=W_lr_scale,
                b_lr_scale=b_lr_scale,max_col_norm=max_col_norm,
                max_row_norm=max_row_norm)
                
    def get_layer_tanh(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  dim,irange,istdev,sparse_init,sparse_stdev,include_prob,init_bias,
                W_lr_scale,b_lr_scale,max_col_norm,max_row_norm
        FROM hps3.layer_tanh
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No tanh layer for layer_id=" \
                +str(layer_id))
        (dim,irange,istdev,sparse_init,sparse_stdev,include_prob,init_bias,
            W_lr_scale,b_lr_scale,max_col_norm,max_row_norm) = row
        return Tanh(layer_name=layer_name,dim=dim,irange=irange,
                istdev=istdev,sparse_init=sparse_init,
                sparse_stdev=sparse_stdev, include_prob=include_prob,
                init_bias=init_bias,W_lr_scale=W_lr_scale,
                b_lr_scale=b_lr_scale,max_col_norm=max_col_norm,
                max_row_norm=max_row_norm)
            
    def get_term_epochcounter(self, term_id):
        row = self.db.executeSQL("""
        SELECT max_epoch
        FROM hps3.term_epochcounter
        WHERE term_id = %s
        """, (term_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No epochCounter term for term_id="\
                +str(term_id))
        max_epochs = row[0]
        return EpochCounter(max_epochs)
        
        
    def get_term_monitorbased(self, term_id):
        row = self.db.executeSQL("""
        SELECT proportional_decrease, max_epoch, channel_name
        FROM hps3.term_monitorBased
        WHERE term_id = %s
        """, (term_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No monitorBased term for term_id="\
                +str(term_id))
        print 'monitor_based'
        (proportional_decrease, max_epochs, channel_name) = row
        return MonitorBased(
                prop_decrease = proportional_decrease, 
                N = max_epochs, channel_name = channel_name
            )
    
    def get_preprocess_standardize(self, preprocess_id):
        row =  self.db.executeSQL("""
        SELECT global_mean, global_std, std_eps	
        FROM hps3.preprocess_standardize
        WHERE preprocess_id = %s
        """, (preprocess_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No standardize preprocess for preprocess_id="\
                +str(preprocess_id))
        (global_mean, global_std, std_eps) = row
        return pp.Standardize(
            global_mean=global_mean, global_std=global_std, 
            std_eps=std_eps
        )
        
    def get_preprocess_zca(self, preprocess_id):
        row =  self.db.executeSQL("""
        SELECT n_components, n_drop_components, filter_bias
        FROM hps3.preprocess_zca
        WHERE preprocess_id = %s
        """, (preprocess_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No zca preprocess for preprocess_id="\
                +str(preprocess_id))
        (n_components, n_drop_components, filter_bias) =  row
        return pp.ZCA(
            n_components=n_components,filter_bias=filter_bias,
            n_drop_components=n_drop_components
        )
        
    def get_preprocess_gcn(self, preprocess_id):
        row =  self.db.executeSQL("""
        SELECT subtract_mean, std_bias, use_norm
        FROM hps3.preprocess_gcn
        WHERE preprocess_id = %s
        """, (preprocess_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No gcn preprocess for preprocess_id="\
                +str(preprocess_id))
        (subtract_mean, std_bias, use_norm) = row
        return pp.GlobalContrastNormalization(
            subtract_mean=subtract_mean, std_bias=std_bias, 
            use_norm=use_norm
        )
    
    def get_ddm_cifar100(self, ddm_id):
        row =  self.db.executeSQL("""
        SELECT  which_set, center, gcn, toronto_prepro, axes,
                start, stop, one_hot
        FROM hps3.ddm_cifar100
        WHERE ddm_id = %s
        """, (ddm_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No cifar100 ddm for ddm_id="\
                +str(ddm_id))
        (which_set, center, gcn, toronto_prepro, axes_char, start, \
            stop, one_hot) = row
        axes = self.get_axes(axes_char)
        return CIFAR100(which_set=which_set,center=center,
                    gcn=gcn,toronto_prepro=toronto_prepro,axes=axes,
                    start=start,stop=stop,one_hot=one_hot)

if __name__ == '__main__':
    worker_name = str(sys.argv[1])
    task_id = int(sys.argv[2])
    start_config_id = None
    if len(sys.argv) > 3:
        start_config_id = int(sys.argv[3])
    hps = HPS(task_id=task_id, worker_name=worker_name )
    hps.run(start_config_id)
    if len(sys.argv) < 2:
        print """
        Usage: python hps3.py "worker_name" "task_id" ["config_id"]
        """
