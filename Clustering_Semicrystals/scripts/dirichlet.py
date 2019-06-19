import numpy as np
import sys
import os
import math
from sklearn.mixture import BayesianGaussianMixture

# The Q-tensor naturally comes with a norm of 1
# scale_factor declares how much "distance" is created through rotation
scale_factor = 10.0

data = np.loadtxt("QTensor.txt")
data[:][3:8] *= scale_factor


#Use an Infinite Gaussian Mixture Model on the data to determine the location of clusters

dpgmm = BayesianGaussianMixture(n_components=500,weight_concentration_prior_type = 'dirichlet_process',verbose=1,n_init=10,max_iter=1000).fit(data)


#Determine the most likely labels of each data point
labels = dpgmm.predict(data)
print dpgmm.get_params()
np.savetxt("labels.txt",labels)
