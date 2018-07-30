import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

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

train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   width_shift_range=0.3,
                                   rotation_range=30,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   vertical_flip=True)

train_generator = train_datagen.flow_from_directory('data/train',
                                                    target_size=(60, 90),
                                                    batch_size=32,
                                                    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory('data/test',
                                                        target_size=(60, 90),
                                                        batch_size=32,
                                                        class_mode='binary')

history = model.fit_generator(train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=train_generator.samples / 32,
                    validation_steps=test_generator.samples / 32,
                    epochs=100)
