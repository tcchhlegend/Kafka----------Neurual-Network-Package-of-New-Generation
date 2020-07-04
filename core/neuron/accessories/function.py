import numpy as np


class Function:
    # __slots__ = "name", "children", "create_variable", "interface",\
    #             "check_interface", "requires_grad_", "parents", "output"

    def F(self):
        return NotImplementedError

    def local_grad(self):
        return NotImplementedError

    def mode_connect(self, *, neuron):
        self.children[neuron.name] = neuron
        neuron.parents[self.name] = self
        self.create_variable(neuron.name, shape=neuron.output_shape)
        return neuron

    def mode_forward(self, *, val, name, interface_full):
        if interface_full:
            return ValueError('Interface is full.')

        if name not in self.interface.keys():
            raise ValueError('The input shape of left function does not match output shape of right function.')
        self.interface[name] = val
