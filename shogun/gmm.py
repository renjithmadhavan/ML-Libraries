import numpy as np
import os
from shogun.Features import RealFeatures
from shogun.Distribution import GMM
from shogun.Library import Math_init_random

# Load the data.
f = open(os.path.dirname(__file__) + '../data/mvnrnd.data')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 2)
f.close()

Math_init_random(5)
feat = RealFeatures(data.T)

# Calculate mixture of Gaussians.
gmm = GMM(2, 0)
gmm.set_features(feat)
gmm.train_em()

# Vector of covariances; one for each Gaussian.
print gmm.get_nth_cov(0)
print gmm.get_nth_cov(1)
# The vector of means.
print gmm.get_nth_mean(0)
print gmm.get_nth_mean(1)
# The a priori weights of each Gaussian.
print gmm.get_coef()