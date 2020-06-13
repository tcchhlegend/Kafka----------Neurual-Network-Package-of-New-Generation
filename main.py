from neuron import Neuron
from castles import Linear
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


nn = NN([n1, n3], [n2], create_connection=True)
print(nn.nodes)

# print(n1._parameters)
# print(n2._parameters)
# print(n3._parameters)
# print(n1._interface)
# print(n2._interface)
# print(n3._interface)
# print(n1.parents)
# print(n2.parents)
# print(n3.parents)

x1 = np.array([[1]])
x2 = np.array([[1]])
print(x1.shape)
n1.create_variable('x1', shape=(1, 1))
n3.create_variable('x2', shape=(1, 1))
y = nn.forward(x1=x1, x2=x2)
print('output:\n', y)
# nn.show()
nn.backward()

def print_state(n):
    print('-----------state for %s------------' % n.name)
    print('interface:\n', n._interface)
    print('grad:\n', n._grad)
    print('grad params:\n', n._grad_params)
    print()

for i in range(1,3+1):
    exec('print_state(n%d)'%i)
