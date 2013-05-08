import numpy as np
import os
import csv

from shogun.Features import RealFeatures
from shogun.Classifier import KernelPCA
from shogun.Distance import EuclideanDistance
from shogun.Kernel import GaussianKernel

# Load the data.
f = open(os.path.dirname(__file__) + '../data/circle_data.txt')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 2)
f.close()

# Perform Kernel PCA.
feat = RealFeatures(data.T)
kernel = GaussianKernel(feat, feat, 2.0)

preprocessor = KernelPCA(kernel)
preprocessor.set_target_dim(2)
preprocessor.init(feat)
transformedData = preprocessor.apply_to_feature_matrix(feat)

# Show transformed data.
print np.matrix(transformedData.T)