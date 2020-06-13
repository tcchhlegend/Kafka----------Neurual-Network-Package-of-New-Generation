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
        INPUT
        *args : 按shape匹配到 interface 的相应位置，并传入
        *kwargs : 传入key对应的interface位置

        OUTPUT
        outputs : 列表，所有根节点上接收到的输出
                若只有一个输出，则返回的是输出数组
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

        """

        def accumulate(neuron):
            if neuron.parents == {}:
                for key, val in neuron._local_grad.items():
                    neuron._grad[key] = val
                for key, val in neuron._local_grad_params.items():
                    neuron._grad_params[key] = val
            else:
                for i, (name, p) in enumerate(neuron.parents.items()):
                    if i == 0:
                        for key, val in neuron._local_grad.items():
                            neuron._grad[key] = p._grad[neuron.name] @ val
                        for key, val in neuron._local_grad_params.items():
                            neuron._grad_params[key] = p._grad[neuron.name] @ val
                    else:
                        for key, val in neuron._local_grad.items():
                            neuron._grad[key] += p._grad[neuron.name] @ val
                        for key, val in neuron._local_grad_params.items():
                            neuron._grad_params[key] += p._grad[neuron.name] @ val

        self.back_flow(accumulate)

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