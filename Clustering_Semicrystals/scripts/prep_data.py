import numpy as np
import sys
import os
import math

filename = sys.argv[1]
molySize = 7

data = np.loadtxt(filename)
nAtoms = len(data)
nMoly = nAtoms/molySize
particleList = np.zeros(nMoly)


#Loop through the data and get molecule positions and orientations
molyPos = np.zeros((nMoly,3))
molyOri = np.zeros((nMoly,3))

midPoint = (molySize-1)/2
test = np.zeros(3)


for i in range(nMoly):
    j = i*molySize
    m = i*molySize+midPoint
    molyPos[i] = data[m][1:4]
    test = data[j+molySize-1][1:4] - data[j][1:4]
    test = test/np.linalg.norm(test)
    molyOri[i] = test

#molyPos contains the positions of all the beads
#molyOri contains all the orientations

#Generate a Q tensor list

molyQ = np.zeros((nMoly,5))

for i in range(nMoly):
    u = molyOri[i]
    Q = 1.5*np.outer(u,u) - 0.5*np.identity(3)
    molyQ[i][0] = Q[0][0]
    molyQ[i][1] = Q[0][1]
    molyQ[i][2] = Q[0][2]
    molyQ[i][3] = Q[1][1]
    molyQ[i][4] = Q[1][2]

outData = np.zeros((nMoly,8))

for i in range(nMoly):
    outData[i][0:3] = molyPos[i]
    outData[i][3:8] = molyQ[i]

np.savetxt("QTensor.txt",outData)

