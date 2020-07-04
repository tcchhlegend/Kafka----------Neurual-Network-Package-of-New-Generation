import numpy as np
from core.neuron.neuron import Neuron


class Input(Neuron):
    """  管理输入

    """
    num_inputs = 0

    def __init__(self, output_shape, name=None):
        Input.num_inputs += 1
        super(Input, self).__init__(output_shape, name=name)
        self.create_variable(self.name, shape=self.output_shape)

    def F(self):
        return self.interface[self.name]

    def local_grad(self):
        self.local_grad_[self.name] = np.eye(self.output_shape[0])

    def create_connection(self, child):
        raise RuntimeError('method `create_connection` of Input neuron should not be called.')