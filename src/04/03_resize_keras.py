from keras.preprocessing.image import load_img

src_file = "sample.jpg"
dest_file = "resized_sample2.jpg"
resize = (60, 90)

img = load_img(src_file, target_size=resize)
img.save(dest_file)
