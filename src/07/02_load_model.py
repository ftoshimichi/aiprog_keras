from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

model = load_model('my_model.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

history = model.fit(x_train, y_train, batch_size=64, epochs=3)
