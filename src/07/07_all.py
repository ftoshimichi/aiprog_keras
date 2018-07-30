from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import model_from_json

f = open("my_model_arch.json", "r")
json_string = f.read()
f.close()

model = model_from_json(json_string)
model.load_weights('my_model_weights.h5')
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

history = model.fit(x_train, y_train, batch_size=64, epochs=3)
