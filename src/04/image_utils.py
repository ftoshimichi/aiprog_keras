from PIL import Image
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

def rotate(src_dir, dest_dir, showProgress=False):
    os.makedirs(dest_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f.startswith(".") == False :
            img = Image.open(src_dir + "/" + f)
            if img.width < img.height:
                img = img.transpose(Image.ROTATE_270)
            img.save(dest_dir + "/" + f)
            if showProgress: print(f)

def resize(src_dir, dest_dir, resize=(90, 60), showProgress=False):
    os.makedirs(dest_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f.startswith(".") == False :
            img = Image.open(src_dir + "/" + f)
            img = img.resize(resize)
            img.save(dest_dir + "/" + f)
            if showProgress: print(f)
