#Alec Bowen Conversion Script
#This Python script takes all the jpeg images in a given folder, converts them to greyscale
#and outputs the pixel values in a single file. Each row consists of the pixel intensities of a
#single row of an image. A new image starts every 64 rows.

import numpy as np
import math
import sys
import os
from PIL import Image

output = open(sys.argv[1],"w")

for filename in os.listdir('.'):
    if 'jpg' in filename:
        im = np.array(Image.open(filename).convert('L'))
        numRows = len(im)
        numCols = len(im[0])
        for i in range(numRows):
            for j in range(numCols):
                output.write(str(im[i][j])+"\t")
            output.write("\n")

output.close()
