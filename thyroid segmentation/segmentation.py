# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:14:46 2020

@author: AZEST-2019-07
"""

import os 
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, jaccard_score
from tensorflow.keras.callbacks import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.regularizers import l2
import cv2
#from dataprep import *

#Reference libraries 
#sklearn.model_selection -> Gridsearch, train_test_split
#sklearn.preprocessing -> StandardScalar, One hot encoder, .. 
#sklearn.linear_model -> lasso, linear regression, stochastic gradient descent,
#sklearn.ensemble -> RandomForestClassifier, RandomForestRegressor,
#sklearn.metrics -> max_error, mean_absolute_error, f1_score, jaccard_score, roc_curve, auc

#keras.models -> model_from_json, model_from_yaml, load_model, save_model, Input, Sequential, Model
#keras.metrics -> accuracy, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy
#keras.layers -> Dense, Cropping2D, Maxpooling2D, Dropout, Conv2D, concatenate, Conv2DTranspose...


input_shape = (320, 512, 1)
n_filters = 32
cf_size = (3, 3)   # convolution filter size
mpf_size = (2, 2)  # max pooling size
s_size = (2, 2)    # stride size
batch_size = 32

fpath = 'C:\\Users\\AZEST-2019-07\Desktop\\Ito\\Selected'
fpath2 = 'C:\\Users\\AZEST-2019-07\Desktop\\Ito\\SelectedP'
fxml = 'C:\\Users\\AZEST-2019-07\Desktop\\Ito\\annotations\\final2.xml'
rjson = 'C:\\Users\\AZEST-2019-07\Desktop\\Ito\\orientation_json.txt'
#imgdict = read_json(rjson)
xt,yt = [],[]

# load the data from path
def load(fpath):
    global xt,yt,imgdict
    for path,subdir,files in os.walk(fpath):
        for file in files:
            full_path = path+ '\\' + file
            if not 'capture' in file.lower(): 
                img = cv2.imread(full_path)
                img = cutpr(img)/255
                
                y = imgdict[file]
                l = img_resize(img)
                [xt.append(i) for i in l]
                [xt.append(flip(i)) for i in l]
                [yt.append(y) for _ in range(2*len(l))]


#load(fpath)
#load(fpath2)

# self defined F1 metric
def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    def recall(y_true, y_pred):
        
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def save_model(model):    
    model.save('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\tl.h5')
    json_string = model.to_json()
    open('models.json', 'w').write(json_string)
    
def load_model2():
    model = model_from_json(open('models.json').read())
    #model = load_model('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\mymodel.h5')
    model.load_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weightss.hdf5')
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=sm.losses.bce_jaccard_loss,
                  optimizer=opt,
                  metrics=[sm.metrics.iou_score,f1])
    return model

# resnet block
def resunit(inp,nf=32,Ksize=3,padding='same',strides=1,BN='True',BN_first=True,activation='relu',sno='0'):
    
    conv = Conv2D(nf,
                  kernel_size=Ksize,
                  strides=strides,
                  padding=padding,
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  name='C'+sno)
    
    x = inp
    
    if BN_first:
        if BN:
            x = BatchNormalization(name='BN'+sno)(x)
        if activation is not None:
            x = Activation(activation,name='Act'+sno)(x)
        x = conv(x)
    else:
        x = conv(x)
        if BN:
            x = BatchNormalization(name='BN'+sno)(x)
        if activation is not None:
            x = Activation(activation,name='Act'+sno)(x)
    return x

# resnet or unet is created based on the 'net' parameter
def resnet(input_shape,n_classes=5,nf=32,nb=4,net='unet'):
    # nb defines the number of resnet blocks
    Ksize=3
    padding='same'
    strides=1
    ld = {}   # stores intermediate layers for concatenation in unet
    nf = 32   # number of filters at the starting
    
    
    inputs = Input(input_shape,name='inp')
    x = resunit(inputs,nf=nf,Ksize=3,padding='same',strides=1,BN_first=False,activation='relu',sno='00')
    x = Conv2D(nf,kernel_size=3,strides=1,padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='C01')(x)
    x = add([inputs, x],name='A0')
    ld['A0'] = x
    
    for i in range(nb):
        nf = 2*nf
        y = resunit(x,nf,strides=2,sno=str((i+1)*10))     # strides = 2 reduces the output dimension by 2
        y = resunit(y,nf,sno=str((i+1)*10+1))
        x = Conv2D(nf,kernel_size=1,padding='same',strides=2,name='1C'+str((i+1)*10))(x)  # matches the shape for addition
        x = add([x,y],name='A0'+str(i+1))
        ld['A'+str(i+1)] = x
    
    if net=='resnet':
        x1 = AveragePooling2D(pool_size=2)(x)
        y1 = Flatten()(x1)
        outputs = Dense(1,
                        activation='sigmoid',
                        kernel_initializer='he_normal',name='out')(y1)
    
    elif net == 'unet':
        for i in range(nb):
            nf = int(nf/2)
            if i<2:
                x = Conv2DTranspose(nf, kernel_size=Ksize, strides=2, padding=padding, kernel_initializer='he_normal')(x)
            else:
                x = UpSampling2D((2,2))(x)
            x = concatenate([ld['A'+str(nb-i-1)],x],axis=3,name='CT'+str(i))
            y = resunit(x,nf,sno=str((i+5)*10))
            y = resunit(y,nf,sno=str((i+5)*10+1))
            x = Conv2D(nf,kernel_size=1,padding='same',strides=1,name='1C'+str((i+5)*10))(x)
            x = add([x,y],name='A0'+str(i+5))
        
        
        outputs = Conv2D(n_classes,kernel_size=1,padding='same',strides=1,name='out')(x)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs, name=net)
    return model

# integrate the learned resnet encoder into unet
def transfer(model,input_shape,n_classes=5,nb=4):
    
    x = model.layers[nb*8+5].output
    nf = 512
    Ksize=3
    padding='same'
    strides=1
    
    for i in range(nb):
        nf = int(nf/2)
        if i<2:
            x = Conv2DTranspose(nf, kernel_size=Ksize, strides=2, padding=padding, kernel_initializer='he_normal')(x)
        else:
            x = UpSampling2D((2,2))(x)
        x = concatenate([model.layers[28-8*i].output,x],axis=3,name='CT'+str(i))
        y = resunit(x,nf,sno=str((i+5)*10))
        y = resunit(y,nf,sno=str((i+5)*10+1))
        x = Conv2D(nf,kernel_size=1,padding='same',strides=1,name='1C'+str((i+5)*10))(x)
        x = add([x,y],name='A0'+str(i+5))
  
    m2 = Model(inputs=model.input, outputs=x,name='unet')
    print(m2.summary())
    return m2
    

def train():
    #tensorboard = TensorBoard(log_dir='C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\logs\\tb1')
    
    model = resnet(input_shape=input_shape,net = 'resnet')
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    
    # loss=sm.losses.JaccardLoss()
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy',f1])
    print(model.summary())
    
    
    # checkpointer = ModelCheckpoint(filepath="C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5", 
    #                                monitor = 'val_accuracy',
    #                                verbose=1, 
    #                                save_best_only=True)
    
    # model.save('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\tl.h5')

train()




