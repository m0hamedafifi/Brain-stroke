

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

def to_arr (im):
    img = Image.fromarray(im, 'RGB')
    diff_colors = list(set(img.getdata()))
    if len(diff_colors) == 1 and diff_colors[0] == (0, 0, 0):
        return [0,1]
    else:
        return[1,0]

def create_traintest_data(img_Dir,lable_Dir):
    data = []    
    for img in tqdm(os.listdir(img_Dir)):
            try: 
              lable = img  
              img_path = os.path.join(img_Dir, img)
              image = Image.open(img_path)
              new_image = image.resize((400, 400))
              lable_Path = os.path.join(lable_Dir, lable)
              ##img_data = cv2.imread(img_path)
              lable_data =cv2.imread(lable_Path)
              data.append([np.array(new_image),to_arr(lable_data)])
            
            except:
                print("An exception occurred")
                print(img)
    shuffle(data)
    test= data[:20]
    _train,validation = train_test_split(data[20:], test_size=0.15) 
    return test,_train,validation

#use in fun#test,train,validation=create_traintest_data()


"""### **Save numpy arrays**"""
def sevenp(save_dir,train,test,validation):
    train_path = os.path.join(save_dir, 'train_400.npy')
    test_path = os.path.join(save_dir, 'test_400.npy')
    validation_path = os.path.join(save_dir, 'validation_400.npy')
    np.save(train_path, train)
    np.save(test_path, test)
    np.save(validation_path, validation)

"""### **Load numpy arrays**"""

#from keras.datasets import imdb
#np_load_old = np.load
#np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
def loadnp(T_DIR,V_DIR,TEST_DIR):
    train =np.load(T_DIR)
    validation =np.load(V_DIR)
    test =np.load (TEST_DIR)
    #np.load = np_load_old



"""#new Load"""
def new_load(DIR):
    X_train=np.load(DIR+"X_train_crop1.npy")
    y_train=np.load(DIR+"Y_train_crop1.npy")
    X_val=np.load(DIR+"X_VAL_crop1.npy")
    y_val=np.load(DIR+"Y_VAL_crop1.npy")
    X_test=np.load(DIR+"X_test_crop1.npy")
    y_test=np.load(DIR+"Y_test_crop1.npy")


"""# ***CNN***"""
def CNN():
    LR = 0.001
    tf.reset_default_graph()
    conv_input = input_data(shape=[None, 400, 400, 1], name='input')
    conv1 = conv_2d(conv_input, 32, 5, activation='relu')
    pool1 = max_pool_2d(conv1, 5)

    conv2 = conv_2d(pool1, 64, 5, activation='relu')
    pool2 = max_pool_2d(conv2, 5)

    conv3 = conv_2d(pool2, 128, 5, activation='relu')
    pool3 = max_pool_2d(conv3, 5)

    conv4 = conv_2d(pool3, 64, 5, activation='relu')
    pool4 = max_pool_2d(conv4, 5)

    conv5 = conv_2d(pool4, 32, 5, activation='relu')
    pool5 = max_pool_2d(conv5, 5)

    fully_layer = fully_connected(pool5, 1024, activation='relu')
    fully_layer = dropout(fully_layer, 0.5)

    cnn_layers = fully_connected(fully_layer, 2, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                            name='targets')
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    return model
    """## **Train weights and Save**"""

def fitCNN(X_train, X_val, y_train, y_val):
    X_train=X_train.reshape(-1,224,224,1)
    X_val=X_val.reshape(-1,224,224,1)
    model=CNN()
    MODEL_NAME = 'Brain stroke'
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=5,
              validation_set=({'input': X_val}, {'targets': y_val}), snapshot_step=500, show_metric=True,
              run_id=MODEL_NAME)

    model.save('model.tfl')

def load():
    model=CNN()
    Load_dir = 'Model_weights 5_5'
    model.load(Load_dir+'./model.tfl')
    return model

def prediction(img_path):
    model=load()
    fimg=resize_img(img_path)
    prediction = model.predict([fimg])[0]
    if prediction[0] <= prediction[1]:
        return [0,1] , prediction[1]
    else:
        return [1,0] , prediction[0]


def resize_img(img_path):
    _img = cv2.imread(img_path, cv2.cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(_img, (400, 400))
    test_img = test_img.reshape(400, 400, 1)
    return test_img


"""# **Test the prediction**"""
'''
testdata=[]
act_test_data=[]
h=0
for img in tqdm(test):
        h+=1
        prediction = model.predict([img[0].reshape(400,400,1)])[0]
        if prediction[0] <= prediction[1]:
          testdata.append([0,1])
          act_test_data.append(img[1])
        else:
          testdata.append([1,0])
          act_test_data.append(img[1])

"""## **The result from prediction**"""

for i in range(len(testdata)):
  print("predict from pic "+str(i) +": "+str(testdata[i]) +"   *||||||*  and act from pic "+str(i)+": "+str(act_test_data[i]))

save_dir='/content/gdrive/My Drive/computed-tomography-ct-images/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/Model_weights 5*5_5/'

model.save(str(save_dir)+'model.tfl')
'''

