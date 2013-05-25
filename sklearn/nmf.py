import numpy as np
import os
from sklearn import decomposition

# Load the data.
f = open(os.path.dirname(__file__) + '../data/fisheriris_data.txt')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 4)
f.close()

# Compute a nonnegative rank-two approximation
# of the measurements of the fisher iris data.
nmf = decomposition.NMF(n_components=2)
W = nmf.fit_transform(data)
H = nmf.components_

# Show the results.
print W
print "\n"
print H
print "\n"
print np.matrix(W) * np.matrix(H)