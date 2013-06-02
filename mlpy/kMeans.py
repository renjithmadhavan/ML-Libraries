import numpy as np
import os
from mlpy import Kmeans

# Load the data.
f = open(os.path.dirname(__file__) + '../data/two_cluster.data')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 2)
f.close()

# Perform kmean with 2 clusters.
kmeans = Kmeans(2)
association = kmeans.compute(data)

# Show cluster association.
print np.matrix(association).T

# Show cluster centers.
print kmeans.means