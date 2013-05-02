import numpy as np
import os
from shogun.Features import RealFeatures, BinaryLabels
from shogun.Classifier import KMeans
from shogun.Distance import EuclideanDistance

# Load the data.
f = open(os.path.dirname(__file__) + '../data/two_cluster.data')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 2)
f.close()

# Perform kmean with 2 clusters.
feat = RealFeatures(data.T)
distance = EuclideanDistance(feat, feat)
kmeans = KMeans(2, distance)
kmeans.train()

# Show cluster association.
print kmeans.apply().get_labels().T

# Show cluster centers.
print kmeans.get_cluster_centers().T
