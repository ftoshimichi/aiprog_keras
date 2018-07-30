from image_utils import load_images

x_train, x_test, y_train, y_test = load_images("dirA_resized", "dirB_resized")

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
