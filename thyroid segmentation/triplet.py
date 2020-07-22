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
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Input, add, Dropout, Conv2DTranspose, UpSampling2D, concatenate, Maxpooling2D
#from tensorflow.keras.layers import GaussianDropout
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.regularizers import l2


(x_tr,y_tr),(x_tst,y_tst) =  mnist.load_data()


def create_model(inp_size):
    
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(96))
    model.add(Dense(36))
    
    return model

def triplet_loss():
    
    def loss(y_true,y_pred):
        
        return None

model = create_model((28,28,1))







