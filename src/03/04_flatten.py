from keras.models import Sequential
from keras.layers import Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(32, 32, 3), padding='same'))
print(model.output_shape)

model.add(Flatten())
print(model.output_shape)
