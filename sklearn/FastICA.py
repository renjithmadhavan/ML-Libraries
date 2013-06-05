import numpy as np
import os
from sklearn.decomposition import FastICA

# Load the data.
data = np.genfromtxt('../data/radical.csv', delimiter=',')

# Peform FastICA
ica = FastICA()
S = ica.fit(data).transform(data)
A = ica.get_mixing_matrix() 

# show the results.
print S
print A