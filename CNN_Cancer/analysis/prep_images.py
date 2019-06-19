import numpy as np
import glob
from skimage.io import imread
import os

os.chdir('../tiff_images/')

filenames = glob.glob('*.tif')


for im in filenames:

    data = imread(im)
    print data.shape
    np.savetxt('../image_data/'+im+'.txt',data)

os.chdir('../analysis')
