import numpy as np
import sys
import os
import math

data = np.loadtxt('labels.txt')

d = {}
j = 0
for i in range(len(data)):
    val = int(data[i])
    if val not in d:
        d[val] = j
        j += 1

for i in range(len(data)):
    val = int(data[i])
    data[i] = d[val]

np.savetxt("labels2.txt",data)
