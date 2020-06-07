import numpy as np
from neuron import Neuron

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
    def __init__(self, num_inputs, num_outputs):
        super(Linear, self).__init__()
        self._inp_shape['x'] = tuple([num_inputs, 1])
        self.output_shape = tuple([num_outputs, 1])
        self.weight = np.random.randn(num_inputs, num_outputs)
        self.bias = np.random.randn(num_outputs, 1)
        self._parameters = {'weight':self.weight, 'bias':self.bias}

    def F(self, x):
        return self.weight.T @ x + self.bias

    def local_grad(self, x):
        self._local_grad = self.weight
        self._local_grad_params['weight'] = x
        self._local_grad_params['bias'] = 1



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