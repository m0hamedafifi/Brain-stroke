import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import *
from PIL import Image
from random import shuffle
from sklearn.model_selection import train_test_split



def blocks(net, n_filter_1, n_filter_1_3, n_filter_3, n_filter_1_5, n_filter_5, n_filter1_pool, name=None):
    conv1 = tf.keras.layers.Conv2D(n_filter_1, 1, activation='relu')(net)

    conv2 = tf.keras.layers.Conv2D(n_filter_1_3, 1, activation='relu')(net)
    conv33 = tf.keras.layers.Conv2D(n_filter_3, 3, padding='same', activation='relu')(conv2)

    conv3 = tf.keras.layers.Conv2D(n_filter_1_5, 1, activation='relu')(net)
    conv55 = tf.keras.layers.Conv2D(n_filter_5, 5, padding='same', activation='relu')(conv3)

    pool1 = tf.keras.layers.MaxPool2D(3, strides=1, padding='same')(net)
    convp = tf.keras.layers.Conv2D(n_filter1_pool, 1, activation='relu')(pool1)

    block = tf.keras.layers.concatenate([conv1, conv33, conv55, convp], axis=3)
    return block
def inception ():
    input_net = tf.keras.Input(shape=(400, 400, 1))

    layer1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, activation='relu')(input_net)
    layer2 = tf.keras.layers.MaxPooling2D(3, strides=2)(layer1)
    layer3 = tf.keras.layers.Conv2D(64, 1, activation='relu')(layer2)
    layer4 = tf.keras.layers.Conv2D(192, 3, activation='relu')(layer3)
    layer5 = tf.keras.layers.MaxPooling2D(3, strides=2)(layer4)
    layer5 = blocks(layer5, 64, 96, 128, 16, 32, 32)
    layer5 = blocks(layer5, 128, 128, 192, 32, 96, 64)
    layer5 = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(layer5)
    layer5 = blocks(layer5, 192, 96, 208, 16, 48, 64)

    result1 = tf.keras.layers.AveragePooling2D(5, strides=3)(layer5)
    result1 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', activation='relu')(result1)
    result1 = tf.keras.layers.Flatten()(result1)
    result1 = tf.keras.layers.Dense(1024, activation='relu')(result1)
    result1 = tf.keras.layers.Dropout(0.5)(result1)
    result1 = tf.keras.layers.Dense(2, activation='softmax', name='left_1')(result1)

    layer5 = blocks(layer5, 160, 112, 224, 24, 64, 64)
    layer5 = blocks(layer5, 128, 128, 256, 24, 64, 64)
    layer5 = blocks(layer5, 112, 144, 288, 32, 64, 64)

    result2 = tf.keras.layers.AveragePooling2D(5, strides=3)(layer5)
    result2 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', activation='relu')(result2)
    result2 = tf.keras.layers.Flatten()(result2)
    result2 = tf.keras.layers.Dense(1024, activation='relu')(result2)
    result2 = tf.keras.layers.Dropout(0.5)(result2)
    result2 = tf.keras.layers.Dense(2, activation='softmax', name='left_2')(result2)

    layer5 = blocks(layer5, 256, 160, 320, 32, 128, 128)
    layer5 = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(layer5)
    layer5 = blocks(layer5, 256, 160, 320, 32, 128, 128)
    layer5 = blocks(layer5, 384, 192, 384, 48, 128, 128)

    result3 = tf.keras.layers.AveragePooling2D(5, strides=3)(layer5)
    result3 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', activation='relu')(result3)
    result3 = tf.keras.layers.Flatten()(result3)
    result3 = tf.keras.layers.Dense(1024, activation='relu')(result3)
    result3 = tf.keras.layers.Dropout(0.5)(result3)
    result3 = tf.keras.layers.Dense(2, activation='softmax', name='final_output')(result3)

    model2 = tf.keras.models.Model(input_net, [result1, result2, result3])
    LR = 0.001
    model2.compile(optimizer=tf.keras.optimizers.Adam(lr=LR),
                   loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                   metrics=['acc'])
    return model2
def load_(img_path):
    model2=inception()
    load_dir = './inception/model.tfl'
    model2.load_weights(str(load_dir))
    fimg = resize_img(img_path)
    prediction = model2.predict([fimg])
    return prediction

def resize_img(img_path):
    img = cv2.imread(img_path, cv2.cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(img, (400, 400))
    test_img = test_img.reshape(-1,400, 400, 1)
    print (np.shape(test_img))
    return test_img

def prediction(img_path):
    predec=load_(img_path)
    pre_of_1 = predec[0][0][0]+predec[1][0][0] +predec[2][0][0]
    pre_of_0 = predec[0][0][1] + predec[1][0][1] + predec[2][0][1]
    pre_of_1=pre_of_1/3
    pre_of_0=pre_of_0/3
    print(prediction)
    if pre_of_1 < pre_of_0:
        return [0,1] , pre_of_0
    else:
        return [1,0] , pre_of_1



