import numpy as np
import glob
import PIL
import openslide
import os
import matplotlib.pyplot as plt
import sys

slide = openslide.OpenSlide(sys.argv[1])
#print(slide.level_count)
#print(slide.level_dimensions)
#print(slide.level_downsamples)

#data = slide.get_thumbnail((512,512))
#print np.array(data)
#np.savetxt('test.txt',np.array(data)[0],fmt="%d")
#slide.get_thumbnail((256,256))
try:
    im = slide.get_thumbnail((256,256))
except:
    print("PROBLEM")
#plt.imshow(slide.get_thumbnail((256,256)))
#plt.show()

