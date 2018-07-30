import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array, array_to_img

src_file = "sample.jpg"

dest_dir = "dest"
os.makedirs(dest_dir, exist_ok=True)
dest_file = "dest/pic_{0:02d}.jpg"

samples = np.array([img_to_array(load_img(src_file))])

datagen = ImageDataGenerator(width_shift_range=0.3,
                             rotation_range=30,
                             zoom_range=0.3,
                             channel_shift_range=50,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode="reflect")

for i in range(1, 100):
    it = datagen.flow(samples)
    array_to_img(it.next()[0]).save(dest_file.format(i))
