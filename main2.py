from castles.linear import Linear
from core.nn.nn import NN
import cv2
import numpy as np

img = cv2.imread('datasets\\messi.jpg')
img = img.reshape([-1])
print(img.shape)

n1 = Linear(output_shape=1024)
n2 = Linear(output_shape=1)
n2(n1)

n1.create_variable('x', shape=(4320, 1))


nn = NN([n1], [n2], create_connection=True)
# nn.show()


def print_state(n):
    print('-----------state for %s------------' % n.name)
    print('interface:\n', n.interface)
    print('parameters:\n', n.parameters)
    print('grad:\n', n.grad_)
    print('grad params:\n', n.grad_params_)
    print()

x = np.random.random([4320, 1])
nn.forward(x=x)
nn.backward()
nn.optimize(0.01)

print_state(n1)
nn.zero_grad()
print_state(n1)