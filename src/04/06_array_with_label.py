import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

def to_array(dir, label):
    rows = []
    labels = []
    for f in os.listdir(dir):
        if f.startswith(".") == False :
            row = img_to_array(load_img(dir + "/" + f))
            rows.append(row)
            labels.append(label)
    return (np.array(rows), np.array(labels))

rowsA, labelsA = to_array("dirA_resized", [1])
rowsB, labelsB = to_array("dirB_resized", [0])

print(rowsA.shape)
print(labelsA.shape)
print(rowsB.shape)
print(labelsB.shape)
