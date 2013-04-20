
import sys

from pylearn2.utils import serial
from itertools import izip
from pylearn2.utils import safe_zip
from collections import OrderedDict
from pylearn2.utils import safe_union

import theano.tensor as T
from theano import config
import numpy as np
from theano import function
from theano.gof.op import get_debug_values
from theano.printing import Print

from hps3 import HPS
from pylearn2.monitor import Monitor
import functools

import theano
from pylearn2.costs.cost import Cost

import os
import csv
import numpy as np

from pylearn2.monitor import Monitor
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.utils import sharedX
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils.iteration import is_stochastic
from pylearn2.utils import py_integer_types, py_float_types
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils.timing import log_timing
from theano.gof.op import get_debug_values
import logging
from collections import OrderedDict


log = logging.getLogger(__name__)

from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.string_utils import preprocess
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace, Space
from pylearn2.models.mlp import Layer
from pylearn2.utils import py_integer_types
from pylearn2.utils import sharedX
from pylearn2.training_algorithms.sgd import SGD

# The number of features in the Y vector
numberOfKeyPoints = 30

"""
TODO:
    either use composite space for targets and output (no spacial layer)
    or use theano.scan softmax.
    -1
"""

class KeypointSGD(SGD):
    """
    Stochastic Gradient Descent

    WRITEME: what is a good reference to read about this algorithm?

    A TrainingAlgorithm that does gradient descent on minibatches.

    """
    def setup(self, model, dataset):

        if self.cost is None:
            self.cost = model.get_default_cost()

        inf_params = [ param for param in model.get_params() if np.any(np.isinf(param.get_value())) ]
        if len(inf_params) > 0:
            raise ValueError("These params are Inf: "+str(inf_params))
        if any([np.any(np.isnan(param.get_value())) for param in model.get_params()]):
            nan_params = [ param for param in model.get_params() if np.any(np.isnan(param.get_value())) ]
            raise ValueError("These params are NaN: "+str(nan_params))
        self.model = model

        batch_size = self.batch_size
        if hasattr(model, "force_batch_size"):
            if model.force_batch_size > 0:
                if batch_size is not None:
                    if batch_size != model.force_batch_size:
                        if self.set_batch_size:
                            model.set_batch_size(batch_size)
                        else:
                            raise ValueError("batch_size argument to SGD conflicts with model's force_batch_size attribute")
                else:
                    self.batch_size = model.force_batch_size
        model._test_batch_size = self.batch_size
        self.monitor = Monitor.get_monitor(model)
        # TODO: come up with some standard scheme for associating training runs
        # with monitors / pushing the monitor automatically, instead of just
        # enforcing that people have called push_monitor
        assert self.monitor.get_examples_seen() == 0
        self.monitor._sanity_check()




        X = model.get_input_space().make_theano_batch(name="%s[X]" % self.__class__.__name__)
        self.topo = not X.ndim == 2

        if config.compute_test_value == 'raise':
            if self.topo:
                X.tag.test_value = dataset.get_batch_topo(self.batch_size)
            else:
                X.tag.test_value = dataset.get_batch_design(self.batch_size)

        Y = T.tensor3(name="%s[Y]" % self.__class__.__name__)


        if self.cost.supervised:
            if config.compute_test_value == 'raise':
                _, Y.tag.test_value = dataset.get_batch_design(self.batch_size, True)

            self.supervised = True
            cost_value = self.cost(model, X, Y)

        else:
            self.supervised = False
            cost_value = self.cost(model, X)
        if cost_value is not None and cost_value.name is None:
            if self.supervised:
                cost_value.name = 'objective(' + X.name + ', ' + Y.name + ')'
            else:
                cost_value.name = 'objective(' + X.name + ')'

        # Set up monitor to model the objective value, learning rate,
        # momentum (if applicable), and extra channels defined by
        # the cost
        learning_rate = self.learning_rate
        if self.monitoring_dataset is not None:
            self.monitor.setup(dataset=self.monitoring_dataset,
                    cost=self.cost, batch_size=self.batch_size, num_batches=self.monitoring_batches,
                    extra_costs=self.monitoring_costs
                    )
            if self.supervised:
                ipt = (X, Y)
            else:
                ipt = X
            dataset_name = self.monitoring_dataset.keys()[0]
            monitoring_dataset = self.monitoring_dataset[dataset_name]
            #TODO: have Monitor support non-data-dependent channels
            self.monitor.add_channel(name='learning_rate', ipt=ipt,
                    val=learning_rate, dataset=monitoring_dataset)
            if self.momentum:
                self.monitor.add_channel(name='momentum', ipt=ipt,
                        val=self.momentum, dataset=monitoring_dataset)

        params = list(model.get_params())
        assert len(params) > 0
        for i, param in enumerate(params):
            if param.name is None:
                param.name = 'sgd_params[%d]' % i

        if self.cost.supervised:
            grads, updates = self.cost.get_gradients(model, X, Y)
        else:
            grads, updates = self.cost.get_gradients(model, X)

        for param in grads:
            assert param in params
        for param in params:
            assert param in grads

        for param in grads:
            if grads[param].name is None and cost_value is not None:
                grads[param].name = ('grad(%(costname)s, %(paramname)s)' %
                                     {'costname': cost_value.name,
                                      'paramname': param.name})

        lr_scalers = model.get_lr_scalers()

        for key in lr_scalers:
            if key not in params:
                raise ValueError("Tried to scale the learning rate on " +\
                        str(key)+" which is not an optimization parameter.")

        log.info('Parameter and initial learning rate summary:')
        for param in params:
            param_name = param.name
            if param_name is None:
                param_name = 'anon_param'
            lr = learning_rate.get_value() * lr_scalers.get(param,1.)
            log.info('\t' + param_name + ': ' + str(lr))

        if self.momentum is None:
            updates.update( dict(safe_zip(params, [param - learning_rate * \
                lr_scalers.get(param, 1.) * grads[param]
                                    for param in params])))
        else:
            for param in params:
                inc = sharedX(param.get_value() * 0.)
                if param.name is not None:
                    inc.name = 'inc_'+param.name
                updated_inc = self.momentum * inc - learning_rate * lr_scalers.get(param, 1.) * grads[param]
                updates[inc] = updated_inc
                updates[param] = param + updated_inc


        for param in params:
            if updates[param].name is None:
                updates[param].name = 'sgd_update(' + param.name + ')'
        model.censor_updates(updates)
        for param in params:
            update = updates[param]
            if update.name is None:
                update.name = 'censor(sgd_update(' + param.name + '))'
            for update_val in get_debug_values(update):
                if np.any(np.isinf(update_val)):
                    raise ValueError("debug value of %s contains infs" % update.name)
                if np.any(np.isnan(update_val)):
                    raise ValueError("debug value of %s contains nans" % update.name)


        with log_timing(log, 'Compiling sgd_update'):
            if self.supervised:
                fn_inputs = [X, Y]
            else:
                fn_inputs = [X]
            self.sgd_update = function(fn_inputs, updates=updates,
                                       name='sgd_update',
                                       on_unused_input='ignore',
                                       mode=self.theano_function_mode)
        self.params = params

class MultiSoftmax(Layer):

    def __init__(self, n_groups, n_classes, layer_name, irange = None,
                 istdev = None, sparse_init = None, W_lr_scale = None,
                 b_lr_scale = None, max_row_norm = None,
                 no_affine = False, max_col_norm = None):
        """
        """

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self

        assert isinstance(n_classes, py_integer_types)

        self.output_space = MatrixSpace(n_groups, n_classes)
        self.b = sharedX( np.zeros((n_groups, n_classes,)), name = 'softmax_b')

    def get_lr_scalers(self):

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            assert isinstance(self.W_lr_scale, float)
            rval[self.W] = self.W_lr_scale

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        if self.b_lr_scale is not None:
            assert isinstance(self.b_lr_scale, float)
            rval[self.b] = self.b_lr_scale

        return rval

    def get_monitoring_channels(self):
        return OrderedDict()

    def get_monitoring_channels_from_state(self, state, target=None):
        return OrderedDict()
        
    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got "+
                    str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        if self.no_affine:
            desired_dim = self.n_classes
            assert self.input_dim == desired_dim
        else:
            desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng

        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,self.irange, (self.input_dim,self.n_groups,self.n_classes))
        elif self.istdev is not None:
            assert self.sparse_init is None
            W = rng.randn(self.input_dim,self.n_groups,self.n_classes) * self.istdev
        else:
            raise NotImplementedError()

        self.W = sharedX(W,  'softmax_W' )

        self._params = [ self.b, self.W ]

    def get_weights_topo(self):
        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()
        desired = self.W.get_value().T
        ipt = self.desired_space.format_as(desired, self.input_space)
        rval = Conv2DSpace.convert_numpy(ipt, self.input_space.axes, ('b', 0, 1, 'c'))
        return rval

    def get_weights(self):
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W.get_value()

    def set_weights(self, weights):
        self.W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        for value in get_debug_values(state_below):
            if self.mlp.batch_size is not None and value.shape[0] != self.mlp.batch_size:
                raise ValueError("state_below should have batch size "+str(self.dbm.batch_size)+" but has "+str(value.shape[0]))

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        assert self.W.ndim == 3

        Z = T.tensordot(state_below, self.W, axes=[[1],[0]]) + self.b

        rval = batched_softmax(Z)

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval

    def cost(self, Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=2).mean()

    def cost_matrix(self, Y, Y_hat):
        return -Y * T.log(Y_hat)

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W = self.W
        return coeff * abs(W).sum()

    def censor_updates(self, updates):
        return
        if self.max_row_norm is not None:
            W = self.W
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x')
        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W = self.W
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))
                
class MatrixSpace(Space):
    """A space whose points are defined as fixed-length vectors."""
    def __init__(self, num_row, num_column):
        """
        Initialize a MatrixSpace.

        Parameters
        ----------
        dim : int
            Dimensionality of a vector in this space.
        sparse: bool
            Sparse vector or not
        """
        self.num_row = num_row
        self.num_column = num_column

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        return np.zeros((self.num_row,self.num_column))

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, n):
        return np.zeros((n, self.num_row, self.num_column))

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None):
        if dtype is None:
            dtype = config.floatX

        return T.tensor3(name=name, dtype=dtype)

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        return self.num_column * self.num_row

    @functools.wraps(Space._format_as)
    def _format_as(self, batch, space):
        raise NotImplementedError()
        if isinstance(space, CompositeSpace):
            pos = 0
            pieces = []
            for component in space.components:
                width = component.get_total_dimension()
                subtensor = batch[:,pos:pos+width]
                pos += width
                formatted = VectorSpace(width).format_as(subtensor, component)
                pieces.append(formatted)
            return tuple(pieces)

        if isinstance(space, Conv2DSpace):
            if space.axes[0] != 'b':
                raise NotImplementedError("Will need to reshape to ('b',*) then do a dimshuffle. Be sure to make this the inverse of space._format_as(x, self)")
            dims = { 'b' : batch.shape[0], 'c' : space.num_channels, 0 : space.shape[0], 1 : space.shape[1] }

            shape = tuple( [ dims[elem] for elem in space.axes ] )

            rval = batch.reshape(shape)

            return rval

        raise NotImplementedError("VectorSpace doesn't know how to format as "+str(type(space)))

    def __eq__(self, other):
        return type(self) == type(other) and self.num_row == other.num_row and self.num_column == other.num_column

    def validate(self, batch):
        if not isinstance(batch, theano.gof.Variable):
            raise TypeError("MatrixSpace batch should be a theano Variable, got "+str(type(batch)))
        if batch.ndim != 3:
            raise ValueError('MatrixSpace batches must be 2D, got %d dimensions' % batch.ndim)


def batched_softmax(x):
    """
    :param x: A Tensor3
    This function computes the batched softmax
    """
                
    result, updates = theano.scan(fn=lambda x_mat:
            T.nnet.softmax(x_mat),
            outputs_info=None,
            sequences=[x],
            non_sequences=None)
    return result

class FacialKeypoint(DenseDesignMatrix):
    """
    A Pylearn2 Dataset object for accessing the data for the
    Kaggle facial-keypoint-detection contest for the IFT 6266 H13 course.
    """

    def __init__(self, which_set,
                 start=None,
                 stop=None,
                 axes=('b', 0, 1, 'c'),
                 stdev=0.8):
        """
        which_set: A string specifying which portion of the dataset
            to load. Valid values are 'train' or 'public_test'
        base_path: The directory containing the .csv files from kaggle.com.
                   If you are using this on the DIRO filesystem, you
                   can just use the default value. If you are using this
                   at home, you should download the .csv files from
                   Kaggle and set base_path to the directory containing
                   them.
        fit_preprocessor: True if the preprocessor is allowed to fit the
                   data.
        fit_test_preprocessor: If we construct a test set based on this
                    dataset, should it be allowed to fit the test set?
        """

        self.stdev = stdev
        files = {'train': 'keypoints_train.csv', 'public_test': 'keypoints_test.csv'}

        try:
            filename = files[which_set]
        except KeyError:
            raise ValueError("Unrecognized dataset name: " + which_set)

        path = os.path.join("${KEYPOINTS_DATA_PATH}/", filename)
        path = preprocess(path)
        csv_file = open(path, 'r')

        reader = csv.reader(csv_file)

        # Discard header
        row = reader.next()

        y_list = []
        X_list = []

        for row in reader:
            if which_set == 'train':
                y_float = self.readKeyPoints(row)
                X_row_str = row[numberOfKeyPoints]  # The image is at the last position
                y_list.append(y_float)
            else:
                _, X_row_str = row
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            X_list.append(X_row)

        X = np.asarray(X_list)
        if which_set == 'train':
            y = np.asarray(y_list)
        else:
            y = None

        if which_set == 'train':
            index = range(X.shape[0])
            np.random.shuffle(index)
            X = X[index,:]
            y = y[index,:]
            self.pixels = np.arange(0,98)
            y = self.make_targets(y)
        """    
        # (num_examples, num_keypoints, 2)
        y = y.reshape((y.shape[0],y.shape[1]/2,2))      
        if rescale_ratio is not None:
            y = self.rescale_keypoints(y, rescale_ratio)
        y = make_spatial_keypoints(y)"""
        
            
        if start is not None:
            assert which_set != 'public_test'
            assert isinstance(start, int)
            assert isinstance(stop, int)
            assert start >= 0
            assert start < stop
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            if y is not None:
                y = y[start:stop, :]
            print y.shape

        view_converter = DefaultViewConverter(shape=[96, 96, 1], axes=axes)

        super(FacialKeypoint, self).__init__(X=X, y=y, view_converter=view_converter)

    def adjust_for_viewer(self, X):
        return (X - 127.5) / 127.5

    def readKeyPoints(self, row):
        """
        Reads the list of keypoints from a row in the csv file
        """
        kp = [-1] * numberOfKeyPoints
        for i in range(numberOfKeyPoints):
            if row[i] is not None and row[i] != "":
                kp[i] = float(row[i])
        return kp
        
    def make_targets(self, y):
        # y : (batch_size, num_keypoints):
        # (batch_size, num_keypoints*2, 98)
        Y = np.zeros((y.shape[0], y.shape[1], 98))
        for i in xrange(y.shape[1]):
            Y[:,i,:] = np.where(y[:,i].reshape(y.shape[0],1)!=-1.,
                (np.exp(-(y[:,i].reshape(y.shape[0],1)-self.pixels)**2/(2*self.stdev**2)))/(np.sqrt(2*3.14159265359)*self.stdev),
                -1.)
        print Y.shape
        return Y

class KeypointHPS(HPS):     
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
                        
        self.target = T.tensor3('target')  
        
        self.monitor = Monitor.get_monitor(self.model)
        self.log_channel_names = []
        self.log_channel_names.extend(self.base_channel_names)
        
        self.monitor.add_dataset(self.valid_ddm, 'sequential', 
                                    self.batch_size)
        if self.test_ddm is not None:
            self.monitor.add_dataset(self.test_ddm, 'sequential', 
                                        self.batch_size)
                                        
    def get_ddm_facialkeypoint(self, ddm_id):
        row =  self.db.executeSQL("""
        SELECT which_set, start, stop, axes
        FROM fkd.ddm_facialKeypoint
        WHERE ddm_id = %s
        """, (ddm_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No facial keypoint ddm for ddm_id="\
                +str(ddm_id))
        (which_set, start, stop, axes_char) = row
        axes = self.get_axes(axes_char)
        return FacialKeypoint(which_set=which_set,axes=axes,
                                    start=start,stop=stop)
    def get_layer_multisoftmax(self, layer_id, layer_name):
        row = self.db.executeSQL("""
        SELECT  n_groups,n_classes,irange,istdev,sparse_init,W_lr_scale,
                b_lr_scale,max_row_norm,no_affine,max_col_norm
        FROM fkd.layer_multisoftmax
        WHERE layer_id = %s
        """, (layer_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No multisoftmax layer for layer_id=" \
                +str(layer_id))
        (n_groups,n_classes,irange,istdev,sparse_init,W_lr_scale,
            b_lr_scale,max_row_norm,no_affine,max_col_norm) = row
        return MultiSoftmax(n_groups=n_groups,n_classes=n_classes,
                        sparse_init=sparse_init,W_lr_scale=W_lr_scale,
                        b_lr_scale=b_lr_scale,max_row_norm=max_row_norm,
                        no_affine=no_affine,max_col_norm=max_col_norm,
                        layer_name=layer_name,irange=irange,
                        istdev=istdev)
                        
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
        return KeypointSGD( learning_rate=learning_rate, cost=cost,
                    batch_size=batch_size,
                    batches_per_iter=num_train_batch,
                    monitoring_dataset=self.monitoring_dataset,
                    termination_criterion=termination_criterion,
                    init_momentum=init_momentum,
                    train_iteration_mode=train_iteration_mode)
        
if __name__=='__main__':
    worker_name = str(sys.argv[1])
    task_id = int(sys.argv[2])
    start_config_id = None
    if len(sys.argv) > 3:
        start_config_id = int(sys.argv[3])
    hps = KeypointHPS(task_id=task_id, worker_name=worker_name )
    hps.run(start_config_id)
    if len(sys.argv) < 2:
        print """
        Usage: python test1.py "worker_name" "task_id" ["config_id"]
        """

