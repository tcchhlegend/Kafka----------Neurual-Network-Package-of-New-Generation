import numpy as np
from dag import DAGraph
from visualization import ShowGraph

class NN(DAGraph, ShowGraph):
    """ 神经网络
    由若干起始节点和若干结束节点组成

    ATTRIBUTE
    nodes : 神经元结点

    METHOD
    forward(*args, **kwargs) : 前向传播 x, parameters --> y
    backward() : 反向传播 grad_upper_layer --> grad_this_layer, grad_params_this_layer
    optimize(lr, algorithm="SGD") : 应用相应的优化算法，更新参数的梯度
    zero_grad() : 将所有结点的 _grad 清零，默认不清除 local_grad
    create_connections(): （初始化调用）将网络中所有有关联的节点创建连接
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
            """不能用 for_flow 的原因
            流动的顺序需要控制
            只有神经元接受了子节点的全部信息后
            才能继续流动
            需要多线程？
            control_for_flow(self, fargs, wait_until)
            未来将会重写这个函数
            """
            if name:
                output = neuron(**{name: input})
            else:
                output = neuron(input)
            if output is not None:
                for p in neuron.parents.values():
                    keep_rock_n_rolling(p, input=output, name=neuron.name)
                neuron.clear_interface()

        # for x in args:
        #     for inp_layer in self.leaves:
        #         if inp_layer.match_inp_shape(x.shape):
        #             keep_rock_n_rolling(inp_layer, input=x)

        for key, val in kwargs.items():
            for inp_layer in self.leaves:
                if key in inp_layer._interface.keys():
                    if inp_layer._interface[key] is None:
                        keep_rock_n_rolling(inp_layer, input=val, name=key)
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
                            neuron._grad[key] = val @ p._grad[neuron.name]
                        for key, val in neuron._local_grad_params.items():
                            neuron._grad_params[key] = p._grad[neuron.name] @ val

                    else:
                        for key, val in neuron._local_grad.items():
                            neuron._grad[key] += val @ p._grad[neuron.name]
                        for key, val in neuron._local_grad_params.items():
                            neuron._grad_params[key] += p._grad[neuron.name] @ val
            neuron.clear_local_grad()

        self.back_flow(accumulate)

    def optimize(self, lr, algorithm='SGD'):
        def optim(neuron, fargs=(0.01, 'SGD')):
            neuron.optimize(lr=fargs[0], algorithm=fargs[1])
            neuron.clear_grad()
        self.for_flow(optim, fargs=(lr, 'SGD'))

    def zero_grad(self):
        def zero(neuron):
            neuron.clear_grad()
        self.for_flow(zero)

    def create_connections(self):
        def conn(node):
            for c in node.children.values():
                node.create_connection(c)
        self.for_flow(conn)