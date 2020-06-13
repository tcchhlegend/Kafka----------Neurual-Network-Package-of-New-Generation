import numpy as np
from functools import partial
from dag import DAGNode
import uuid
from utils import create_none_dict, to_tuple


class Neuron(DAGNode):
    """神经元单位
    实质是一个函数，y = f(x, parameters)
    可以有很多输入，但只有一个输出！！！只有一个输出！！！只有一个输出！！！
    当输入为位置变量时，用 match_inp_shape 来寻找对应接口，
    当输入为关键字位置时，对应的key就是接口的名字

    ATTRIBUTE
    variable_names : 列表，存储变量名
    param_names : 列表，存储参数名
    _interface : 字典，存储输入
    _parameters : 字典，存储参数
    _local_grad : 字典，存储局部梯度 （输出对输入的梯度）
    _grad : 字典，自变量的梯度
    _local_grad_params : 字典，存储局部梯度 （输出对参数的梯度）
    _grad_params : 字典，存储参数的梯度
    _int_shape : 字典，存储输入数组形状
    parents : 字典，存储神经元单位的父节点 （接收本单元的输出结点）
    children : 字典，存储神经元单位的子节点 （作为本单元的输入的结点）

    METHOD (important)
    check_interface : 检查输入条件是否全部满足，如果是，返回True
    match_inp_shape : 在 _inp_shape 中找到匹配的变量的 name
    F : 求函数值
    local_grad : 求梯度值
    create_variable : 创建（注册）一个变量
    create_paramter : 创建（注册）一个参数
    create_connection : 与子节点创建连接，注册参数和设置前向传播模式
    __call__ : 若输入是Neuron，则为连接模式，若输入是数组，则为前馈模式

    """
    __slots__ = 'name', 'variable_names', 'param_names'
    num_neurons = 0

    def __init__(self, output_shape, name=None, requires_grad=True):
        self.naming(name)
        self.output_shape = to_tuple(output_shape)
        self.variable_names = []
        self.param_names = []
        self.uid = uuid.uuid1()
        super(Neuron, self).__init__()
        self._interface = create_none_dict(self.variable_names)
        self._local_grad = create_none_dict(self.variable_names)
        self._grad = create_none_dict(self.variable_names)
        self._inp_shape = create_none_dict(self.variable_names)
        self._parameters = create_none_dict(self.param_names)
        self._local_grad_params = create_none_dict(self.param_names)
        self._grad_params = create_none_dict(self.param_names)
        self.requires_grad_ = requires_grad

    def naming(self, name):
        Neuron.num_neurons += 1
        if not name:
            name = "neuron%d" % Neuron.num_neurons
        self.name = name
        print('naming neuron to %s' % self.name)

    def show_inputs(self):
        input_summary = []
        for i, (name, shape) in enumerate(self._inp_shape.items()):
            connect = None
            if self.children[name]:
                connect = self.children[name].name
            line = '%d. name: %s    shape: %s    connect to: %s' % (i + 1, name, str(shape), connect)
            input_summary.append(line)
        print('------input summary------')
        print('\n'.join(input_summary))

    def check_interface(self):
        return all([x is not None for x in self._interface.values()])

    def match_inp_shape(self, shape1):
        for name, shape2 in self._inp_shape.items():
            if tuple(shape1) == tuple(shape2):
                return name
        return None

    def F(self):
        raise NotImplementedError

    def local_grad(self):
        raise NotImplementedError

    def create_variable(self, var_name, shape):
        self.variable_names.append(var_name)
        self._interface[var_name] = None
        self._local_grad[var_name] = None
        self._grad[var_name] = None
        self._inp_shape[var_name] = tuple(shape)

    def create_parameter(self, par_name, shape):
        self.param_names.append(par_name)
        self._parameters[par_name] = None
        self._local_grad_params[par_name] = None
        self._grad_params[par_name] = None

    def create_connection(self, child):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        mode = None
        connect_output = []
        interface_full = self.check_interface()

        # ------------------Handle args---------------------
        for i, arg in enumerate(args):
            if i == 0:
                if isinstance(arg, Neuron):
                    mode = 'connect'  # 如果输入是 Neuron, 启动连接模式
                elif isinstance(arg, np.ndarray):
                    mode = 'forward'  # 如果输入时 array, 启动前馈模式

                else:
                    raise ValueError('discover unknown input type: %s' % type(arg))

            if mode == 'connect':
                neuron = arg
                self.children[neuron.name] = neuron
                neuron.parents[self.name] = self
                self.create_variable(neuron.name, shape=neuron.output_shape)
                connect_output.append(neuron)
            elif mode == 'forward':  # 否则默认输入是数组，调用forward方法
                if interface_full:
                    return ValueError('Interface is full.')
                x = arg
                name = self.match_inp_shape(x.shape)
                self._interface[name] = x
                if not name:
                    raise ValueError('The input shape of left function does not match output shape of right function.')
                interface_full = self.check_interface()

        # -------------------------------Handle kwargs------------------------------
        for i, (key, val) in enumerate(kwargs.items()):
            if i == 0:
                if isinstance(val, Neuron):
                    mode = 'connect'  # 如果输入是 Neuron, 启动连接模式
                elif isinstance(val, np.ndarray):
                    mode = 'forward'  # 如果输入时 array, 启动前馈模式

                else:
                    raise ValueError('discover unknown input type: %s' % type(arg))

            if mode == 'connect':
                neuron = val
                self.children[neuron.name] = neuron
                neuron.parents[self.name] = self
                self.create_variable(neuron.name, shape=neuron.output_shape)
                connect_output.append(neuron)
            elif mode == 'forward':  # 否则默认输入是数组，调用forward方法
                if interface_full:
                    return ValueError('Interface is full.')
                x = val
                name = key
                if name not in self._interface.keys():
                    raise ValueError('The input shape of left function does not match output shape of right function.')
                self._interface[name] = x

                interface_full = self.check_interface()

        # forward 输出
        if mode == 'forward':
            forward_output = self.F()
            if self.requires_grad_:     # 若我们要求神经元的梯度，在算完函数值以后计算梯度
                self.local_grad()
        if mode == 'forward' and self.parents == {}:
            self.output = forward_output
        if mode == 'forward' and interface_full:
            return forward_output

        # connect 输出
        if mode == 'connect':
            return connect_output

    def __repr__(self):
        return self.name


if __name__ == '__main__':
    n1 = Neuron(output_shape=10)
    n2 = Neuron(output_shape=1)
    n3 = Neuron(output_shape=3)
    n2(n1)
    n2(n3)
    n2.show_inputs()
