import numpy as np
from core.neuron.neuron import Neuron


class MSELoss(Neuron):
    def __init__(self, name=None):
        super(MSELoss, self).__init__(output_shape=1, name=name)

    def naming(self, name):
        MSELoss.num_neurons += 1
        if not name:
            name = "MSELoss"
        self.name = name
        print('create %s' % self.name)

    def F(self):
        if not self.check_interface():
            return None
        vals = list(self.interface.values())
        return np.sum((vals[0] - vals[1])**2) / 2

    def local_grad(self):
        if not self.check_interface():
            return None
        keys = list(self.interface.keys())
        vals = list(self.interface.values())
        self.local_grad_[keys[0]] = (vals[0] - vals[1]).T
        self.local_grad_[keys[1]] = (vals[1] - vals[0]).T

    def check_interface(self):
        return super(MSELoss, self).check_interface() and len(self.interface) == 2

    def create_connection(self, child):
        pass
