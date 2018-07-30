import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def linear(x):
    return x

print(softmax(np.array([0.9, 0.2, 0.6])))
print(softmax(np.array([0.8, 0.1, 0.8])))

print(linear(np.array([0.9, 0.2, 0.6])))
print(linear(np.array([0.8, 0.1, 0.8])))
