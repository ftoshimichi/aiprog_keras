import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_images(dirA, dirB):
    rowsA, labelsA = to_array(dirA, [1])
    rowsB, labelsB = to_array(dirB, [0])
    data = np.r_[rowsA, rowsB]
    label = np.r_[labelsA, labelsB]
    return train_test_split(data, label)

def to_array(dir, label):
    rows = []
    labels = []
    for f in os.listdir(dir):
        if f.startswith(".") == False :
            row = img_to_array(load_img(dir + "/" + f))
            rows.append(row)
            labels.append(label)
    return (np.array(rows), np.array(labels))

x_train, x_test, y_train, y_test = load_images("dirA_resized", "dirB_resized")

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
