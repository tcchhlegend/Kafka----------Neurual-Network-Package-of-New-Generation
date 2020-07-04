import numpy as np
from core.dag.dag import DAGNode
from core.neuron.accessories.formattor import Formattor
from core.neuron.accessories.initializer import Initializer
from core.neuron.accessories.optimizer import Optimizer
from core.neuron.accessories.function import Function
from utils.utils import create_none_dict, to_tuple


class Neuron(DAGNode, Function, Formattor, Initializer, Optimizer):
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

    def naming(self, name):
        Neuron.num_neurons += 1
        if not name:
            name = "neuron%d" % Neuron.num_neurons
        self.name = name
        print('naming neuron to %s' % self.name)

    def __init__(self, output_shape, name=None, requires_grad=True):
        self.variable_names = []
        self.param_names = []
        self.naming(name)   # 属性 name
        self.output_shape = to_tuple(output_shape)  # 属性 output_shape
        # self.uuid = uuid.uuid1()   # 创建唯一 id
        super(Neuron, self).__init__()  # 属性 parents, children
        self.interface = create_none_dict(self.variable_names)
        self.local_grad_ = create_none_dict(self.variable_names)
        self.grad_ = create_none_dict(self.variable_names)
        self.inp_shape = create_none_dict(self.variable_names)
        self.parameters = create_none_dict(self.param_names)
        self.local_grad_params_ = create_none_dict(self.param_names)
        self.grad_params_ = create_none_dict(self.param_names)
        self.requires_grad_ = requires_grad

    def __repr__(self):
        return self.name


    def __call__(self, *args, **kwargs):
        '''调用函数'''
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
                neuron = self.mode_connect(neuron=arg)
                connect_output = neuron

            elif mode == 'forward':  # 否则默认输入是数组，调用forward方法
                raise ValueError('Position arguments are not allowed in `forward` mode.')

        # -------------------------------Handle kwargs------------------------------
        for i, (key, val) in enumerate(kwargs.items()):
            if i == 0:
                if isinstance(val, Neuron):
                    mode = 'connect'  # 如果输入是 Neuron, 启动连接模式
                elif isinstance(val, np.ndarray):
                    mode = 'forward'  # 如果输入时 array, 启动前馈模式

                else:
                    raise ValueError('discover unknown input type: %s' % type(val))

            if mode == 'connect':
                raise ValueError('Keyword arguments are not allowed in `connection mode`.')

            elif mode == 'forward':  # 否则默认输入是数组，调用forward方法
                self.mode_forward(val=val, name=key, interface_full=interface_full)

                interface_full = self.check_interface()

        # forward 输出
        if mode == 'forward':
            forward_output = self.F()
            if self.requires_grad_:     # 若我们要求神经元的梯度，在算完函数值以后计算梯度
                self.local_grad()
            if self.parents == {}:
                self.output = forward_output
            if interface_full:
                return forward_output

        # connect 输出
        if mode == 'connect':
            return connect_output

