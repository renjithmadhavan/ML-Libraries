import numpy as np
import os
import csv
from mlpy import KPCA, kernel_gaussian

# Load the data.
f = open(os.path.dirname(__file__) + '../data/circle_data.txt')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 2)
f.close()

# Perform Kernel PCA.
kernel = kernel_gaussian(data, data, sigma=2) # gaussian kernel matrix
gaussian_pca = KPCA()
gaussian_pca.learn(kernel)
transformedData = gaussian_pca.transform(kernel, k=2)

# Show transformed data.
print transformedData