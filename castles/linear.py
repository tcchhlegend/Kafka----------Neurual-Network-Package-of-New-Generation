import numpy as np
from core.neuron.neuron import Neuron


class Linear(Neuron):
    """      ``线性函数``    在神经网络中用作全连接层
    y = w1*x1 + w2*x2 + ... + wn*xn + bias

    天然参数为 bias，后天参数为各输入对应的系数矩阵

    METHOD
        naming : 默认名 linear + 序号
        F() : ``函数值`` x --> w * x + b
        local_grad() : ``梯度值`` x --> (*local_grads, *local_grad_params)
        create_connection(child) : 与相应子节点创建线性联系
        create_variable(var_name, shape) : （在create_connection中触发）创建输入 x，同时创建对应参数
        create_parameter(par_name, shape) : （在create_variable中触发）

    梯度计算规则:
            local_grad ： 即``本单元``的输出对``各个参数``的梯度
                    dy / dx = w.T
                    dy / dw = x.T
                    dy / db = 1
            grad (total_grad) : 从``树根``开始累积到``本单元``的梯度
                    dL / dx = (dL / dy) * (dy / da) * ... * (dy / dx)

            grad 左乘
            grad_params 右乘
    """
    num_lin = 0

    def __init__(self, output_shape):
        super(Linear, self).__init__(output_shape=output_shape)
        self.output_shape = tuple([output_shape, 1])
        self.bias = np.random.randn(output_shape, 1)
        self.create_parameter('bias', (output_shape,1))

    def naming(self, name):
        """
        线性函数的默认名为 linear + 序号
        """
        self.num_neurons += 1
        Linear.num_lin += 1
        if not name:
            self.name = 'linear%s' % Linear.num_lin
        print('create %s' % self.name)

    def F(self):
        if not self.check_interface():
            return None
        output = self.bias
        for key in self.interface.keys():
            output += self.parameters[key] @ self.interface[key]
        return output

    def local_grad(self):
        if not self.check_interface():
            return None

        for key in self.local_grad_.keys():
            self.local_grad_[key] = self.parameters[key]

        self.local_grad_params_['bias'] = np.array([[1]])

        for key in self.interface.keys():
            if (self.interface[key] is not None) and (self.local_grad_params_[key] is None):
                self.local_grad_params_[key] = self.interface[key]

    def create_variable(self, var_name, shape):
        super(Linear, self).create_variable(var_name, shape)
        self.create_parameter(var_name, shape=(self.output_shape[0], shape[0]))

    def create_parameter(self, par_name, shape):
        super(Linear, self).create_parameter(par_name, shape)
        self.parameters[par_name] = np.random.randn(shape[0], shape[1])

    def create_connection(self, child):
        self.create_variable(child.name,  shape=child.output_shape)
        self.create_parameter(child.name, shape=(self.output_shape[0], child.output_shape[0]))
