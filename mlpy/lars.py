import numpy as np
import os
import mlpy
from Lars import LARS

# Load the data.
f = open(os.path.dirname(__file__) + '../data/artificial_data_X.data')
X = np.fromfile(f, dtype=np.float64, sep=' ')
X = X.reshape(-1, 5)
f.close()

f = open(os.path.dirname(__file__) + '../data/artificial_data_Y.data')
Y = np.fromfile(f, dtype=np.float64, sep=' ')
f.close()

# Perform LARS.
lars = LARS()
lars.learn(X, Y)

# Show the results.
print lars.beta()