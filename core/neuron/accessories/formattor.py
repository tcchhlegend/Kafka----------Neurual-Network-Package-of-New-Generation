from utils.utils import clear_dict


class Formattor:
    # __slots__ = "name", "_interface", "_inp_shape", \
    #             "_local_grad", "_local_grad_params", \
    #             "_grad", "_grad_params", "children"

    def check_interface(self):
        return all([x is not None for x in self.interface.values()])

    def match_inp_shape(self, shape1):
        for name, shape2 in self.inp_shape.items():
            if tuple(shape1) == tuple(shape2):
                return name
        return None

    # --------------三大垃圾回收------------------------
    def clear_interface(self):
        """用在执行完输入后"""
        clear_dict(self.interface)

    def clear_local_grad(self):
        """用在算完gradient后"""
        clear_dict(self.local_grad_)
        clear_dict(self.local_grad_params_)

    def clear_grad(self):
        """用在optimize后"""
        clear_dict(self.grad_)
        clear_dict(self.grad_params_)

    def show_inputs(self):
        input_summary = []
        for i, (name, shape) in enumerate(self.inp_shape.items()):
            connect = None
            if self.children[name]:
                connect = self.children[name].name
            line = '%d. name: %s    shape: %s    connect to: %s' % (i + 1, name, str(shape), connect)
            input_summary.append(line)
        print('------input summary------')
        print('\n'.join(input_summary))
