# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:25:51 2020

@author: AZEST_ANALYTICS
"""


import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import model_from_json, load_model,Model,Sequential
from tensorflow.keras import metrics
from tensorflow.keras.layers import InputLayer, Conv2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Input, add, Dropout, Conv2DTranspose, UpSampling2D, concatenate, MaxPooling2D
#from tensorflow.keras.layers import GaussianDropout
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.manifold import TSNE
import seaborn as sns
sns.set_style('darkgrid')

(x_tr,y_tr),(x_tst,y_tst) =  mnist.load_data()

def reorganizeMNIST(x, y):
    assert x.shape[0] == y.shape[0]
    
    dataset = {i: [] for i in range(10)}
    l = {i: 0 for i in range(10)}
    
    for i in range(x.shape[0]):
        dataset[y[i]].append(x[i])
        l[y[i]]+=1
    
    return dataset,l


dataset,l = reorganizeMNIST(x_tr, y_tr)
# tsne from raw data
def scatter(x, labels):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    
    plt.show()
    
def create_model(inp_size):
    
    model = Sequential()
    model.add(InputLayer(inp_size))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(96))
    model.add(Dense(36))
    
    return model



def triplet_loss():
    
    def all_diffs(a, b):
        # Returns a tensor of all combinations of a - b
        return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

    def euclidean_dist(embed1, embed2):
        # Measures the euclidean dist between all samples in embed1 and embed2
        
        diffs = all_diffs(embed1, embed2) # get a square matrix of all diffs
        return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)


    def loss(y_true,y_pred):
        
        return None

#model = create_model((28,28,3))

def train(model,xt,yt,epochs,batch_size):
    
    
    
    
    
    
    
    None
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

