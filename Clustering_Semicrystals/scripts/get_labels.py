import numpy as np
import sys
import os
import math

molySize = 7

data = np.loadtxt('input.xyz')
labels = np.loadtxt("labels2.txt")

output = open("clusters.xyz","w")

els = ["H","C","N","O","S","F","P","Cl"]

output.write(str(len(data))+"\n\n")
j = 0
for i in range(len(data)):
    current_type = int(labels[j])%len(els)
    output.write(str(els[current_type])+"\t"+str(data[i][1])+"\t"+str(data[i][2])+"\t"+str(data[i][3])+"\n")
    if (i+1)%molySize == 0:
        j += 1

output.close()
