from neuron import Neuron
from castles import Linear, MSELoss, Input
from nn import NN
import numpy as np

n1 = Linear(output_shape=1)
n2 = Linear(output_shape=2)
n3 = Linear(output_shape=1)
n2(n1)
n2(n3)

def feed_forward_network(num_features, num_layers):
    layers = []
    # 创建神经元
    for i in range(num_layers):
        features = [Neuron(output_shape=1) for j in range(num_features)]
        layers.append(features)

    # 创建连接
    for i in range(num_layers-1):
        for j in range(num_features):
            for k in range(num_features):
                parent = layers[i+1][j]
                child = layers[i][k]
                parent(child)
    return NN(layers[0], layers[-1], create_connection=False)

# ff = feed_forward_network(5, 3)
# ff.show()


x1 = np.array([[1]])
x2 = np.array([[1]])
y = np.array([[1],[0]])
print(x1.shape)
n1.create_variable('x1', shape=(1, 1))
n3.create_variable('x2', shape=(1, 1))
op = MSELoss()
op(n2)
y_true = Input(output_shape=n2.output_shape, name='y_true')
op(y_true)

nn = NN([n1, n3, y_true], [op], create_connection=True)
print(nn.nodes)
nn.show()


def print_state(n):
    print('-----------state for %s------------' % n.name)
    print('interface:\n', n._interface)
    print('parameters:\n', n._parameters)
    print('grad:\n', n._grad)
    print('grad params:\n', n._grad_params)
    print()


y = nn.forward(x1=x1, x2=x2, y_true=y)
# print('output:\n', y)
nn.backward()

for i in range(1,3+1):
    exec('print_state(n%d)'%i)
    print_state(op)
    print_state(y_true)