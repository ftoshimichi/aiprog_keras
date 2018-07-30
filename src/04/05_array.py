from keras.preprocessing.image import load_img, img_to_array

src_file = "sample.jpg"
resize = (60, 90)

img = load_img(src_file, target_size=resize)
img_array = img_to_array(img)
print(img_array.shape)
