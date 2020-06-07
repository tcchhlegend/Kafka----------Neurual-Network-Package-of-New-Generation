from neuron import Neuron
from nn import NN

# n1 = Neuron(output_shape=1)
# n2 = Neuron(output_shape=2)
# n3 = Neuron(output_shape=1)
# n2(n1)
# n2(n3)

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

ff = feed_forward_network(5, 3)
ff.show()

# nn = NN([n1, n3], [n2], create_connection=False)
# print(nn.nodes)
# nn.show()