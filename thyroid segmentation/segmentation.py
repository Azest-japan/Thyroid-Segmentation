# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:14:46 2020
@author: AZEST-2019-07
"""

import os 
#import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Input, add, Dropout, Conv2DTranspose, UpSampling2D, concatenate
#from tensorflow.keras.layers import GaussianDropout
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.regularizers import l2
import cv2
from dataprep import cutpr, img_resize, read_json, flip, displt, readxml, onehot, decode, enhanceQ, quality, disp_decode, save_res, fusion, cut, cfname
import gc
import segmentation_models as sm
#from tensorflow.keras.utils import multi_gpu_model

#Reference libraries 
#sklearn.model_selection -> Gridsearch, train_test_split
#sklearn.preprocessing -> StandardScalar, One hot encoder, .. 
#sklearn.linear_model -> lasso, linear regression, stochastic gradient descent,
#sklearn.ensemble -> RandomForestClassifier, RandomForestRegressor,
#sklearn.metrics -> max_error, mean_absolute_error, f1_score, jaccard_score, roc_curve, auc

#keras.models -> model_from_json, model_from_yaml, load_model, save_model, Input, Sequential, Model
#keras.metrics -> accuracy, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy
#keras.layers -> Dense, Cropping2D, Maxpooling2D, Dropout, Conv2D, concatenate, Conv2DTranspose...

tf.compat.v1.enable_eager_execution() 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#config =  tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

input_shape = (320, 512, 1)
n_filters = 32
cf_size = (3, 3)   # convolution filter size
mpf_size = (2, 2)  # max pooling size
s_size = (2, 2)    # stride size
batch_size = 32

fpath1 = '/test/Ito/Selected1'
fpath2 = '/test/Ito/SelectedP'
fxml1 = '/test/Ito/annotations/final.xml'
fxml2 = '/test/Ito/annotations/final2.xml'
rjson = '/test/Ito/orientation_json.txt'


# load the data from path
def load_data1(xt=[],yt=[]):
    imgdict = read_json(rjson)
    imgd = {}
    imgd = readxml(fxml1,imgd)
    imgd = readxml(fxml2,imgd)
    
    for fp in [fpath1,fpath2]:
        for path,subdir,files in os.walk(fp):
            for file in files:
                full_path = path+ '/' + file
                if not 'capture' in file.lower() and file in imgdict.keys() and file in imgd.keys(): 
                    i3 = cv2.imread(full_path)
                    i3 = cutpr(i3)
                    q = quality(i3)
                    l = img_resize(i3) + img_resize(flip(i3))
                    if q<2000:
                        i32 = enhanceQ(i3,q)
                        l = l + img_resize(i32) + img_resize(flip(i32)) 
                        
                    y = [imgdict[file]] + imgd[file]
                    
                    [xt.append(i) for i in l]
                    [yt.append(y) for _ in range(len(l))]
    return xt,yt


def load_data2(fxml,xt=[],yt=[],lb=[]):
    
    imgd = readxml(fxml)
    for k,v in imgd.items():
        no = int(k.split('_')[0])
        if no<=60:
            continue
        if no<201:
            i3 = cv2.imread('/test/Ito/Selected1/' + k)
        else:
            i3 = cv2.imread('/test/Ito/SelectedP/' + k)
        if type(i3) == type(None):
            continue
        i3 = cutpr(i3)
        q = quality(i3)
        l = img_resize(i3) + img_resize(flip(i3))
        if q<2000:
            i32 = enhanceQ(i3,q)
            l = l + img_resize(i32) + img_resize(flip(i32)) 
           
        
        [xt.append(i) for i in l]
        [lb.append(k) for _ in range(len(l))]
        
        l = img_resize(v) + img_resize(flip(v))
        gc.collect()
        if q<2000:
            l = l + l
        [yt.append(onehot(i)) for i in l]
    
    return xt,yt,lb

def addtotalex(xt,yt,lb):
    for path,subdir,files in os.walk('/test/Ito/Selected1/Normal/total extirpation'):
        for file in files:
            full_path = path+ '/' + file
            if not 'capture' in file.lower(): 
                i3 = cutpr(cv2.imread(full_path))
                q = quality(i3)
                i32 = []
                l = img_resize(i3) + img_resize(flip(i3))
                if q<2000:
                    i32 = enhanceQ(i3,q)
                    l = l + img_resize(i32) + img_resize(flip(i32)) 
        
                [xt.append(i) for i in l]
                [lb.append(file) for _ in range(len(l))]
                v = np.uint8(np.zeros((320,512)))
                [yt.append(onehot(v)) for i in range(len(l))]
                
    return np.uint8(xt),np.uint8(yt),np.array(lb)

def test(fxml,xt,yt):
    imgd = readxml(fxml)
    for path,subdir,files in os.walk('/test/Ito/test/'):
        for file in files:
            full_path = path+ '/' + file
            if not 'capture' in file.lower() and file in imgd.keys():
                no = int(file.split('_')[0])
                i3 = cutpr(cv2.imread(full_path))
                q = quality(i3)
                l = img_resize(i3) + img_resize(flip(i3))
            
                [xt.append(i) for i in l]
                
                l = img_resize(imgd[file]) + img_resize(flip(imgd[file]))
                gc.collect()
                [yt.append(onehot(i)) for i in l]
    return xt,yt

def std(x):
    p,q,r,_ = x.shape
    m = np.mean(x)
    s = np.sqrt(np.sum(np.square(x - m))/(p*q*r-1))
    return (x-m)/s


def save_model(model,net):    
    model.save('/test/Ito/'+net+'.h5')
    json_string = model.to_json()
    open('models.json', 'w').write(json_string)
    
def load_model2(fm, fw, lr = 0.0001):
    #model = model_from_json(open('models.json').read())
    model = load_model(fm)
    model.load_weights(fw)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss= keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy',f1])
    
    return model

# resnet block
def resunit(inp,nf=32,Ksize=3,padding='same',strides=1,BN='True',BN_first=True,activation='relu',sno='0'):
    # sno = serial number of the unit
    if strides == 1 and int(sno)<50:
        def conv(x):
            c1 = Conv2D(int(nf/8),
                          kernel_size=1,
                          strides=strides,
                          padding=padding,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4),
                          name='C'+sno+'_1')(x)
            c2 = Conv2D(int(3*nf/4),
                          kernel_size=3,
                          strides=strides,
                          padding=padding,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4),
                          name='C'+sno+'_2')(x)
            c04 = Conv2D(int(nf/32),
                          kernel_size=1,
                          strides=strides,
                          padding=padding,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4),
                          name='C'+sno+'_04')(x)
            c4 = Conv2D(int(nf/8),
                          kernel_size=5,
                          strides=strides,
                          padding=padding,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4),
                          name='C'+sno+'_4')(c04)
            c = concatenate([c1,c2,c4],axis=3,name='CT_conv'+str(sno))
            return c
        
    else:
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

# resnet encoder or unet is created based on the 'net' parameter
def resUnet(input_shape,n_classes=4,nf=32,nb=4,net='unet',dropout=0.25):
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
        if i ==nb-1:
            x = Dropout(dropout)(x)
    
        ld['A'+str(i+1)] = x
    
    if net=='resnet':
        x1 = AveragePooling2D(pool_size=2)(x)
        x1 = Flatten()(x1)
        x1 = Dropout(dropout)(x1)
        outputs = Dense(n_classes,
                        activation='sigmoid',kernel_initializer='he_normal', name='out')(x1)
    
    elif net == 'unet':
        for i in range(nb):
            nf = int(nf/2)
            if i<2:
                x = Conv2DTranspose(nf, kernel_size=Ksize, strides=2, padding=padding, kernel_initializer='he_normal', name='C2DT'+str(i))(x)
            else:
                x = UpSampling2D((2,2), name='up'+str(i))(x)
            x = concatenate([ld['A'+str(nb-i-1)],x],axis=3,name='CT'+str(i))
            y = resunit(x,nf,sno=str((i+5)*10))
            y = resunit(y,nf,sno=str((i+5)*10+1))
            x = Conv2D(nf,kernel_size=1,padding='same',strides=1,name='1C'+str((i+5)*10))(x)
            x = add([x,y],name='A0'+str(i+5))
        
        x = Conv2D(n_classes,kernel_size=1,padding='same',strides=1,name='out0')(x)
        outputs = Activation('sigmoid',name='out')(x)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs, name=net)
    return model,ld

# integrate the learned resnet encoder into unet
def transfer(model,input_shape,ld,n_classes=4,nb=4,dropout=0.2):
    
    x = model.layers[[l.name for l in model.layers].index('A0'+str(nb))].output
    nf = 512
    Ksize=3
    padding='same'
    strides=1
    
    for i in range(nb):
        nf = int(nf/2)
        if i<2:
            x = Conv2DTranspose(nf, kernel_size=Ksize, strides=2, padding=padding, kernel_initializer='he_normal', name='C2DT'+str(i))(x)
        else:
            x = UpSampling2D((2,2), name='up'+str(i))(x)
        x = concatenate([ld['A'+str(nb-i-1)],x],axis=3,name='CT'+str(i))
        y = resunit(x,nf,sno=str((i+5)*10))
        y = resunit(y,nf,sno=str((i+5)*10+1))
        x = Conv2D(nf,kernel_size=1,padding='same',strides=1,name='1C'+str((i+5)*10))(x)
        x = add([x,y],name='A0'+str(i+5))
        if i ==nb-1:
            x = Dropout(dropout)(x)
      
    x = Conv2D(n_classes,kernel_size=1,padding='same',strides=1,name='out0')(x)
    outputs = Activation('sigmoid',name='out')(x)
    m2 = Model(inputs=model.input, outputs=outputs,name='unet')
    #print(m2.summary())
    return m2
    
# Initialize the Resnet encoder
def initialize(input_shape=(320,512,1),n_classes=4,net = 'resnet',dropout = 0.25):
    
    tf.compat.v1.reset_default_graph()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    
    model,ld = resUnet(input_shape=input_shape,n_classes=n_classes,net = net,dropout = dropout)
    #print(model.summary())
    return model,ld
   

def train_resnet(model,lr = 0.0001,epochs = 12, batch_size = 16):
    #tensorboard = TensorBoard(log_dir='C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\logs\\tb1')
    
    gc.collect()
    opt = keras.optimizers.Adam(learning_rate=lr)
    
    # loss=sm.losses.JaccardLoss()
    model.compile(loss= keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy',f1])
    
    
    
    checkpointer = ModelCheckpoint(filepath="/test/Ito/resnet_weights2_"+str(lr)+".hdf5", 
                                    monitor = 'val_accuracy',
                                    verbose=1, 
                                    save_best_only=True)
    
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    gc.collect()
    class_weights = {0:1, 1:2, 2:4}
    
    model.fit(x_tr, y_tr,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_tst, y_tst), callbacks=[checkpointer,earlystopping],class_weight=class_weights)
    
    score = model.evaluate(x_tst, y_tst, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    return model



def weighted_binary_crossentropy(w,batch_size):
    a = K.ones((batch_size,320,512,4))
    b = K.ones((batch_size,320,512,4))
    #c = K.ones((batch_size,320,512,4))
    #d = K.ones((batch_size,320,512,4))
    
    def loss(y_true, y_pred):

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)
        alpha = 0.5
        b_ce = b_ce * a[:,:,:,1].assign(b[:,:,:,1] + alpha*K.cast(((y_true>[0.5,1,1,1])[:,:,:,0]  == (y_pred>[1,0.5,1,1])[:,:,:,1]),'float32'))
        b_ce = b_ce * a[:,:,:,3].assign(b[:,:,:,3] + alpha*K.cast(((y_true<[1,1,1,0.5])[:,:,:,3]  == (y_pred>[1,1,1,0.5])[:,:,:,3]),'float32'))
        
        
        # Apply the weights
        weighted_b_ce = w * b_ce
        jloss = sm.losses.JaccardLoss(class_weights=w)
        dl = sm.losses.DiceLoss()
        
        l_0 = dl(K.flatten(y_true[:,:,:,0]),K.flatten(y_pred[:,:,:,0]))
        l_1 = dl(K.flatten(y_true[:,:,:,1]),K.flatten(y_pred[:,:,:,1]))
        l_2 = dl(K.flatten(y_true[:,:,:,2]),K.flatten(y_pred[:,:,:,2]))
        l_3 = dl(K.flatten(y_true[:,:,:,3]),K.flatten(y_pred[:,:,:,3]))
        
        l = (w[0]*l_0 + w[1]*l_1 + w[2]*l_2*K.cast(l_2 > 0.06,'float32') + w[3]*l_3*K.cast(l_3 > 0.06,'float32') )/(w[0] + w[1] + w[2]*K.cast(l_2 > 0.1,'float32') + w[3]*K.cast(l_3>0.1,'float32'))
        
        """ 
        tf.config.experimental_run_functions_eagerly(True)
       
        @tf.function
        def f(x):
          if x > 1:
            return 1
          else:
            return 0
        
        ch1 = f(K.sum(y_true[:,:,:,2]))
        ch2 = f(K.sum(y_true[:,:,:,3]))
        
        if (ch1<1 and ch2>0) or (ch1>0 and ch2<1):
            l = 1 - (1-l)*4/3
        elif ch1 and ch2:
            l = 1 - (1-l)*2
        """
    
        # Return the mean error
        return K.mean(weighted_b_ce) + l

    return loss 


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


def train_unet(model,lr = 0.0001,epochs = 12, batch_size = 12):
    #tensorboard = TensorBoard(log_dir='C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\logs\\tb1')
    
    #for i in range(1,[l.name for l in model.layers].index('BN10')):
        #print(i,model.layers[i])
        #model.layers[i].trainable = False
    #sess = tf.compat.v1.keras.backend.get_session()
    #K.set_session(sess)
    opt = keras.optimizers.Adam(learning_rate=lr)
    
    #model = multi_gpu_model(model, gpus=2)
    class_weights = np.array([1,1,2,3])
    class_w = np.zeros((320,512,4))
    class_w[:,:,0] = 1
    class_w[:,:,1] = 1
    class_w[:,:,2] = 1.5
    class_w[:,:,0] = 2
    
    jloss= sm.losses.JaccardLoss(class_weights=class_weights)
    iou = sm.metrics.IOUScore()
    bloss = tf.keras.losses.BinaryCrossentropy()
    dl = sm.losses.DiceLoss(class_weights=class_weights)
    
    model.compile(loss= weighted_binary_crossentropy(K.variable(class_weights),batch_size), optimizer=opt, metrics=['accuracy',iou,jloss])
    
    checkpointer = ModelCheckpoint(filepath="/test/Ito/unet_weights2_"+str(lr)+".hdf5", 
                                    monitor = 'val_iou_score',
                                    verbose=1, 
                                    save_best_only=True,save_weights_only=True,mode='max')
    
    earlystopping = EarlyStopping(monitor='val_iou_score', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    gc.collect()
    
    datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=8,
    zoom_range=1/16)
    
    model.fit(datagen.flow(x_tr, y_tr.astype('float32'), batch_size=batch_size),
                    steps_per_epoch=len(x_tr) / batch_size, epochs=epochs,validation_data=(xtst, ytst.astype('float32')),callbacks=[checkpointer])
    
    
    return model

def load_m(fw1='/test/Ito/unet_weights2_0.0001.hdf5',dropout=0.2):
    
    gc.collect()

    fm = '/test/Ito/unet.h5'
    fw0 = '/test/Ito/resnet_weights2_0.0001.hdf5'

    model,ld = initialize(n_classes=3,dropout=dropout)
    #tf.compat.v1.enable_eager_execution() 
    model = transfer(model,input_shape,ld,n_classes=4,nb=4,dropout=dropout)
    model.load_weights(fw1)
    return model

model = load_m()


'''
gc.collect()
xt,yt,lb = [],[],[]
print(len(xt),len(yt),len(lb),0)
xt,yt,lb = load_data2(fxml1,xt,yt,lb)
print(len(xt),len(yt),len(lb),1)
xt,yt,lb = load_data2(fxml2,xt,yt,lb)
print(len(xt),len(yt),len(lb),2)
xt,yt,lb = addtotalex(xt,yt,lb)
xt,yt,lb = np.uint8(xt),np.uint8(yt),np.array(lb)
print(xt.shape,yt.shape,lb.shape,3)

for i in range(yt.shape[0]):
    s = np.sum(yt[i,:,:,2])
    if s < 500 and s>0:
        yt[i,:,:,2][yt[i,:,:,2]>0] = 0

p = np.random.permutation(len(xt))
xt = xt[p[:]]
yt = yt[p[:]]
lb = lb[p[:]]


gc.collect()
xtst = xt[-500:]
ytst = yt[-500:]
xtst = xtst.reshape(xtst.shape+(1,))
xtst = xtst.astype('float32')
xtst = xtst/255

p = np.random.permutation(len(xt)-500)
x_tr = xt[:-500][p[:800]]
y_tr = yt[:-500][p[:800]]
x_tr = x_tr.reshape(x_tr.shape+(1,))
x_tr = x_tr.astype('float32')
x_tr = x_tr/255


model = train_unet(model,lr = 0.0002,epochs = 6,batch_size = 8)
model.save('/test/Ito/unet_2.h5')

score = model.evaluate(xtst.reshape(xtst.shape+(1,)), ytst, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


model,ld = initialize(n_classes=3,dropout=0.125)
model.load_weights(fw0)
model = transfer(model,input_shape,ld,n_classes=4,nb=4)
pred = model.predict(x_tst)
save_res(x_tst,pred)



'''





