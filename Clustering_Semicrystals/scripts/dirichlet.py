import numpy as np
import sys
import os
import math
from sklearn.mixture import BayesianGaussianMixture

scale_factor = 10.0

data = np.loadtxt("QTensor.txt")

data[:][3:8] *= scale_factor



