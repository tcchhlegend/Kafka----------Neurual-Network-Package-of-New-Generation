class Optimizer:
    # __slots__ = "_grad_params", "_parameters"

    def optimize(self, lr, algorithm='SGD'):
        for key, val in self.grad_params_.items():
            print('key name:', key)
            print('param shape:',self.parameters[key].shape)
            print('grad shape: ',val.shape )
            if algorithm == 'SGD':
                self.parameters[key] -= lr * val
            if algorithm == 'AdaGrad':
                ...
            if algorithm == 'Momentum':
                ...
            if algorithm == 'Adam':
                ...
