import numpy as np
import os
from sklearn.decomposition import KernelPCA

# Load the data.
f = open(os.path.dirname(__file__) + '../data/circle_data.txt')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 2)
f.close()

# Perform Kernel PCA.
kpca = KernelPCA(n_components=2, kernel="rbf")
transformedData = kpca.fit_transform(data)

# Show transformed data.
print transformedData