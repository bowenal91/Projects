import numpy as np
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
import sys

dg = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        rescale=0./255,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

img = load_img(sys.argv[1])
x = img_to_array(img)
x = x.reshape((1,)+x.shape)

i = 0
for batch in dg.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='stuff', save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely


