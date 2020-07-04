class Initializer:
    # __slots__ = "_interface", "_grad", "_local_grad", "_inp_shape",\
    #             "_parameters", "_local_grad_params", "_grad_params"

    def create_variable(self, var_name, shape):
        self.variable_names.append(var_name)
        self.interface[var_name] = None
        self.local_grad_[var_name] = None
        self.grad_[var_name] = None
        self.inp_shape[var_name] = tuple(shape) if shape else None

    def create_parameter(self, par_name, shape):
        self.param_names.append(par_name)
        self.parameters[par_name] = None
        self.local_grad_params_[par_name] = None
        self.grad_params_[par_name] = None
