from PIL import Image

src_file = "sample_portrait.jpg"
dest_file = "rotated_sample.jpg"

img = Image.open(src_file)
if img.width < img.height:
    img = img.transpose(Image.ROTATE_270)
img.save(dest_file)
