import numpy as np
import os
from sklearn.cluster import KMeans

# Load the data.
f = open(os.path.dirname(__file__) + '../data/two_cluster.data')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 2)
f.close()

# Perform kmeans with 2 clusters.
kmeans = KMeans(k=2, n_init=1)
kmeans.fit(data)
labels = kmeans.labels_

# Show cluster association.
print np.matrix(labels).T

# Show cluster centers.
print kmeans.cluster_centers_