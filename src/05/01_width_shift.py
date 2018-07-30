import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array, array_to_img

src_file = "sample.jpg"
dest_file = "shifted_sample.jpg"

sample = img_to_array(load_img(src_file))
samples = np.array([sample])

datagen = ImageDataGenerator(width_shift_range=0.3)
it = datagen.flow(samples)
array_to_img(it.next()[0]).save(dest_file)