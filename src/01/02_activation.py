import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

x = np.arange(-3, 3, 0.1)

plt.plot(x, sigmoid(x))
plt.plot(x, tanh(x), linestyle='--')
plt.plot(x, relu(x), linestyle=':')

plt.ylim(-1.1, 1.1)
plt.show()
