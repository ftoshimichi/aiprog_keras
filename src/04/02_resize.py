from PIL import Image

src_file = "sample.jpg"
dest_file = "resized_sample.jpg"
resize = (90, 60)

img = Image.open(src_file)
img = img.resize(resize, Image.BILINEAR)
img.save(dest_file)
