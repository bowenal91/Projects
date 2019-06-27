import sys
import numpy as np
import glob
import PIL
import openslide
import os
import matplotlib.pyplot as plt

'''
    Script to randomly select tiles from the image that have enough information
    to meaningfully perform analysis on
'''

def get_random_tile(slide,x,y,size,ds):
    rand_x = np.random.randint(x)
    rand_y = np.random.randint(y)
    out = slide.read_region((rand_x,rand_y),0,size)
    return out

def get_threshold_tile(slide,x,y,size,ds,thresh):
    max_it = 1000
    test = 0
    for i in range(max_it):
        test = get_random_tile(slide,x,y,size,ds)
        gs = test.convert('L')
        if np.average(gs) < thresh:
            break
    return test



slide = openslide.OpenSlide(sys.argv[1])
ds = 1
size = (256,256)
slide_size = slide.level_dimensions
x = slide_size[0][0] - size[0]
y = slide_size[0][1] - size[1]

num_samples = 20
for i in range(num_samples):
    test = get_threshold_tile(slide,x,y,size,ds,175)
    test.save(sys.argv[1]+str(i)+".png")
#print test
#plt.imshow(test)
#plt.show()

