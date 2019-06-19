#Draws rectangles around the locations on class.jpg determined to be a face

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

im = np.array(Image.open('class.jpg').convert('L'))

fig,ax = plt.subplots(1)

ax.imshow(im,cmap='gray',interpolation='nearest')

face_data = np.loadtxt('face_locations.dat')

for i in range(len(face_data)):
    rect = patches.Rectangle((face_data[i][1],face_data[i][0]),64,64,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

plt.savefig('final_image.jpg')
plt.show()
