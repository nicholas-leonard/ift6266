
import sys

import theano.tensor as T
from theano import config
import numpy as np
from theano import function

from hps2 import HPS
from keypoints_dataset import FacialKeypoint
from pylearn2.monitor import Monitor

import theano.tensor as T
from pylearn2.costs.cost import Cost

class KeypointHPS(HPS):
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
        valMSE = MissingTargetCost()(model, minibatch, target)
        monitor = Monitor.get_monitor(model)
        
        monitor.add_dataset(self.valid_ddm, 'sequential', batch_size)
        monitor.add_channel("Validation MSE",
                            (minibatch, target),
                            valMSE,
                            self.valid_ddm)
                            
        if self.test_ddm is not None:
            monitor.add_dataset(self.test_ddm, 'sequential', batch_size)
            monitor.add_channel("Test MSE",
                                (minibatch, target),
                                valMSE,
                                self.test_ddm)
                                
        
    def get_ddm(self, ddm_id):
        if ddm_id is None:
            return None
        ddm_class = self.select_ddm(ddm_id)
        if ddm_class == 'facialkeypoint':
            (which_set, start, stop, axes_char) \
                = self.select_ddm_facialKeypoint(ddm_id)
            axes = self.get_axes(axes_char)
            return FacialKeypoint(which_set=which_set,axes=axes,
                                    start=start,stop=stop)
        else:
            raise HPSData("dataset class not supported:"+str(ddm_class))
    
    def get_cost(self, config_id, config_class):
        if 'mixture' in config_class:
            ([gater_autonomy], [expert_autonomy]) \
                = self.select_cost_mixture(config_id)
            costs = [MixtureCost(gater_autonomy=gater_autonomy,
                                 expert_autonomy=expert_autonomy)]
        elif 'mtc' in config_class:
            costs = [MissingTargetCost()]
        else:
            costs = [MethodCost(method='cost_from_X', supervised=True)]
            
        if 'kmeans' in config_class:
            kmeans_coeff = self.select_cost_kmeans(config_id)
            costs.append(KmeansCost(coeffs=kmeans_coeff))
        if 'mlp' in config_class:
            weight_decay = self.select_cost_weightdecay(config_id)
            if weight_decay is not None:
                costs.append(WeightDecay(coeffs=weight_decay))        
        if len(costs) > 1:
            return SumOfCosts(costs)
        else:
            return costs[0]
            
    def select_ddm_facialKeypoint(self, ddm_id):
        row =  self.db.executeSQL("""
        SELECT which_set, start, stop, axes
        FROM fkd.ddm_facialKeypoint
        WHERE ddm_id = %s
        """, (ddm_id,), self.db.FETCH_ONE)
        if not row or row is None:
            raise HPSData("No facial keypoint ddm for ddm_id="\
                +str(ddm_id))
        return row
        
if __name__=='__main__':
    worker_name = str(sys.argv[1])
    task_id = int(sys.argv[2])
    start_config_id = None
    if len(sys.argv) > 3:
        start_config_id = int(sys.argv[3])
    log_channel_names = ['train_objective', 'Validation MSE']
    mbsb_channel_name = 'Validation MSE'
    hps = KeypointHPS(task_id=task_id, log_channel_names=log_channel_names,
              mbsb_channel_name=mbsb_channel_name, 
              worker_name=worker_name )
    hps.run(start_config_id)
    if len(sys.argv) < 2:
        print """
        Usage: python main.py "worker_name" "task_id" ["config_id"]
        """

