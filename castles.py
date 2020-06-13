import numpy as np
from neuron import Neuron
from utils import create_none_dict

class Input(Neuron):
    """  管理输入

    """
    num_inputs = 0

    def __init__(self, output_shape, name=None):
        Input.num_inputs += 1
        super(Input, self).__init__(output_shape, name=name)
        self.create_variable('x', shape=self.output_shape)

    def F(self, x):
        return x

    def local_grad(self, x):
        return x

    def create_connection(self, child):
        raise RuntimeError('method `create_connection` of Input should not be called.')





class Linear(Neuron):
    """      ``线性函数``
        在神经网络中用作全连接层
        F: ``函数值`` x --> w * x + b

        backward:
            local_grad ： 即``本单元``的输出对``各个参数``的梯度
                    dy / dx = w
                    dy / dw = x
                    dy / db = 1
            grad (total_grad) : 从``树根``开始累积到``本单元``的梯度
                    dL / dx = (dL / dy) * (dy / da) * ... * (dy / dx)
    """
    def __init__(self, output_shape):
        super(Linear, self).__init__(output_shape=output_shape)
        self.output_shape = tuple([output_shape, 1])
        self.bias = np.random.randn(output_shape, 1)
        self.create_parameter('bias', (output_shape,1) )

    def F(self):
        if not self.check_interface():
            return None
        output = self.bias
        for key in self._interface.keys():
            output += self._parameters[key] @ self._interface[key]
        return output

    def local_grad(self):
        for key in self._local_grad.keys():
            self._local_grad[key] = self._parameters[key]
        self._local_grad_params['bias'] = np.eye(self.output_shape[0])

        for key in self._interface.keys():
            if (self._interface[key] is not None) and (self._local_grad_params[key] is None):
                self._local_grad_params[key] = self._interface[key].T

    def create_variable(self, var_name, shape):
        super(Linear, self).create_variable(var_name, shape)
        self.create_parameter(var_name, shape=(self.output_shape[0], shape[0]))

    def create_parameter(self, par_name, shape):
        super(Linear, self).create_parameter(par_name, shape)
        self._parameters[par_name] = np.random.randn(shape[0], shape[1])

    def create_connection(self, child):
        self.create_variable(child.name,  shape=(child.output_shape[0], 1))
        self.create_parameter(child.name, shape=(self.output_shape[0], child.output_shape[0]))


if __name__ == '__main__':
    lin1 = Linear(10, 10)
    print(lin1.weight.shape)
    print(lin1.bias.shape)
    x = np.random.randn(20, 1)
    # y = lin1.forward(x)
    lin2 = Linear(20, 10)
    lin3 = lin1(lin2)
    y = lin3.forward(x)
    print(y.shape)
    lin3.backward()