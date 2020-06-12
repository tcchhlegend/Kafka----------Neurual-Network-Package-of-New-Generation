import numpy as np
from dag import DAGraph
from visualization import ShowGraph

class NN(DAGraph, ShowGraph):
    """ 神经网络
    由若干起始节点和若干结束节点组成

    ATTRIBUTE
    nodes : 神经元结点

    METHOD
    forward : 前向传播 x, parameters --> y
    backward : 反向传播 grad_upper_layer --> grad_this_layer, grad_params_this_layer
    optimize : 应用相应的优化算法，更新参数的梯度
    zero_grad : 将所有结点的 _grad 清零，默认不清除 local_grad
    """
    def __init__(self, leaves, roots, create_connection=True):
        nodes = self.get_subgraph(leaves, roots)
        super(NN, self).__init__(nodes)
        if create_connection:
            self.create_connections()

    def forward(self, *args, **kwargs):
        """ 前向传播
        调用此函数说明前向传播由此发起
        因此传入参数必须是完整的
        """
        def keep_rock_n_rolling(neuron, *, input, name=None):
            if name:
                output = neuron(**{name:input})
            else:
                output = neuron(input)
            if output is not None:
                for p in neuron.parents.values():
                    keep_rock_n_rolling(p, input=output, name=neuron.name)

        for x in args:
            for inp_layer in self.leaves:
                if inp_layer.match_inp_shape(x.shape):
                    keep_rock_n_rolling(inp_layer, input=x)

        for key, val in kwargs.items():
            for inp_layer in self.leaves:
                if key in inp_layer._interface.keys():
                    if inp_layer._interface[key] is None:
                        keep_rock_n_rolling(inp_layer, input=val)
                        break

        outputs = [root.output for root in self.roots]
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def backward(self):
        """ 反向传播
        调用跳阶梯度，然后与局部梯度相乘
        :param :
        :return:
        """
        for inp_tensor in self._grad.keys():
            self._grad[inp_tensor] = np.zeros(self.inp_shape[inp_tensor])
            for parent, grad in self._grad:
                self._grad[inp_tensor] += grad @ self._local_grad[parent]

        for param in self._grad_params.keys():
            self._grad_params[param] = np.zeros(self.inp_shape[param])
            for parent, leap_grad in self._leap_grad:
                self._grad_params[param] += self._local_grad_params[param] @ leap_grad

    def optimize(self, lr, algorithm='SGD'):
        for key, val in self._grad_params.items():
            if algorithm == 'SGD':
                self._grad_params[key] -= lr * val
            if algorithm == 'AdaGrad':
                ...
            if algorithm == 'Momentum':
                ...
            if algorithm == 'Adam':
                ...

    def zero_grad(self):
        self._grad = {}
        self._grad_params = {}

    def create_connections(self):
        def conn(node):
            for c in node.children.values():
                node.create_connection(c)
        self.for_flow(conn)