import numpy as np
import os
from sklearn import linear_model

# Load the data.
f = open(os.path.dirname(__file__) + '../data/artificial_data_X.data')
X = np.fromfile(f, dtype=np.float64, sep=' ')
X = X.reshape(-1, 5)
f.close()

f = open(os.path.dirname(__file__) + '../data/artificial_data_Y.data')
Y = np.fromfile(f, dtype=np.float64, sep=' ')
f.close()

# Perform LARS.
lars = linear_model.LassoLars(alpha=0., normalize=True)
lars.fit(X, Y)
output = np.matrix(lars.coef_).T

# Show the results.
print output