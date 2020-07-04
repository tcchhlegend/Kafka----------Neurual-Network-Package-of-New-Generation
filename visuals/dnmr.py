"""
DNMR, a.k.a digital nuclear magnetic resonance, is a
tool of visualizing the state of parameters in a neural
network. The fantastic strength of plotting out the parameters
allows developers to quickly check the behaviour of his/her model,
while not needing to read the harsh matrix and see what's going on.


"""

import numpy as np
from castles.castles import Linear

n1 = Linear(output_shape=10)
n2 = Linear(output_shape=10)
n3 = Linear(output_shape=1)
