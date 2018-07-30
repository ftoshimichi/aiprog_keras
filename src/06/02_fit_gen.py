import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from image_utils import load_images

x_train, x_test, y_train, y_test = load_images("dirA", "dirB")

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3),
                    input_shape=(60, 90, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
           optimizer=keras.optimizers.Adadelta(),
           metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.3,
                             rotation_range=30,
                             zoom_range=0.3,
                             horizontal_flip=True,
                             vertical_flip=True)


history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=100,
                    validation_data=(x_test, y_test))
