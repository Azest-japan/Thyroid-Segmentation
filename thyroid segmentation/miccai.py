import os 
#import time
import sys
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
sys.path.append('/test/Ito/Code')
from dataprep import cutpr, img_resize, read_json, flip, displt, readxml, onehot, decode, enhanceQ, quality, disp_decode, save_res, fusion, cut, cfname
import gc
import segmentation_models as sm


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
        
        
                     
input_shape = (320, 512, 1)
n_filters = 32
cf_size = (3, 3)   # convolution filter size
mpf_size = (2, 2)  # max pooling size
s_size = (2, 2)    # stride size
batch_size = 32

xpath = '/test/Downloads/TNSCUI2020_train/TNSCUI2020_train/image'
ypath = '/test/Downloads/TNSCUI2020_train/TNSCUI2020_train/mask'

xpath2 = '/test/Downloads/tnscui2020_testset/image'

def load_data(xpath):
    xt = []
    yt = []
    lb = []
    for path,subdir,files in os.walk(xpath):
        for file in files:
            xfull = xpath+ '/' + file
            yfull = ypath+ '/' + file
            ix = cv2.imread(xfull,0)
            iy = cv2.imread(yfull,0)
            l = img_resize(ix) #+ img_resize(flip(ix))
            
            [xt.append(i) for i in l]
            [lb.append(file) for _ in range(len(l))]
            
            l = img_resize(iy) #+ img_resize(flip(iy))
            [yt.append(i) for i in l]
            #print(file,ix.shape,iy.shape,len(l),l[0].shape)
            #break
    return xt,yt,lb


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

# Building the model
# resnet block
def resunit(inp,nf=32,Ksize=3,padding='same',strides=1,BN='True',BN_first=True,activation='relu',sno='0'):
    # sno = serial number of the unit
    
    if strides == 1 and int(sno)<60 and int(sno)>25:
        def conv(x):
            c1 = Conv2D(int(nf/8),
                          kernel_size=1,
                          strides=strides,
                          padding=padding,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4),
                          name='C'+sno+'_1')(x)
            c2 = Conv2D(int(5*nf/8),
                          kernel_size=3,
                          strides=strides,
                          padding=padding,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4),
                          name='C'+sno+'_2')(x)
            c3 = Conv2D(int(nf/8),
                          kernel_size=3,
                          strides=strides,
                          dilation_rate=2,
                          padding=padding,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4),
                          name='C'+sno+'_3')(x)
            c4 = Conv2D(int(nf/8),
                          kernel_size=3,
                          strides=strides,
                          dilation_rate=3,
                          padding=padding,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4),
                          name='C'+sno+'_4')(x)
            c = concatenate([c1,c2,c3,c4],axis=3,name='CT_conv'+str(sno))
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
    nf = 32   # number of filters at the start
    
    
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
            x = Dropout(dropout,name='DD'+str(i+1))(x)
    
        ld['A'+str(i+1)] = x
    
    if net=='resnet':
        x1 = AveragePooling2D(pool_size=2)(x)
        x1 = Flatten()(x1)
        x1 = Dropout(dropout)(x1)
        outputs = Dense(n_classes,
                        activation='sigmoid',kernel_initializer='he_normal', name='out')(x1)
    
    # nf = 512
    elif net == 'unet':
        for i in range(nb):
            nf = int(nf/2)
            if i<2:
                x = Conv2DTranspose(nf, kernel_size=Ksize, strides=2, padding=padding, kernel_initializer='he_normal', name='C2DT'+str(i))(x)
            else:
                x = UpSampling2D((2,2), name='up'+str(i))(x)
            
            if i == nb-1:
                xp = ld['A'+str(nb-i-1)]
                x1 = Conv2D(int(nf/2),kernel_size=1,padding='same',strides=1,name='CI1'+str((i+5)*10))(xp)
                x2 = Conv2D(int(nf/2),kernel_size=3,padding='same',strides=1,name='CI2'+str((i+5)*10))(xp)
                x3 = concatenate([x1,x2],axis=3,name='CTI'+str(i))
                x3 = BatchNormalization(name='BNI'+str(i))(x3)
                x3 = Activation('relu',name='ActI'+str(i))(x3)
                x3 = add([xp,x3],name='add'+str(i))
                x = concatenate([x3,x],axis=3,name='CT'+str(i))
            else:
                x = concatenate([ld['A'+str(nb-i-1)],x],axis=3,name='CT'+str(i))
            y = resunit(x,nf,sno=str((i+5)*10))
            y = resunit(y,nf,sno=str((i+5)*10+1))
            x = Conv2D(nf,kernel_size=1,padding='same',strides=1,name='1C'+str((i+5)*10))(x)
            x = add([x,y],name='A0'+str(i+5))
            if i ==nb-1:
                x = Dropout(dropout,name='DU'+str(i+5))(x)
                
        x = Conv2D(n_classes,kernel_size=1,padding='same',strides=1,name='out0')(x)
        outputs = Activation('sigmoid',name='out')(x)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs, name=net)
    return model,ld


# integrate the learned resnet encoder into unet
def transfer(model,input_shape,ld,n_classes=1,nb=4,dropout=0.2):
    # nb defines the number of resnet blocks
    
    x = model.layers[[l.name for l in model.layers].index('DU8')].output
    x = Conv2D(n_classes,kernel_size=1,padding='same',strides=1,name='out0')(x)
    outputs = Activation('sigmoid',name='out')(x)
    m2 = Model(inputs=model.input, outputs=outputs,name='unet')
    
    #print(m2.summary())
    return m2



def train_unet(model,x_tr,y_tr,x_tst,y_tst,lr = 0.0001,epochs = 8, batch_size = 8):
    #tensorboard = TensorBoard(log_dir='C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\logs\\tb1')
    
    #for i in range(1,[l.name for l in model.layers].index('BN10')):
        #print(i,model.layers[i])
        #model.layers[i].trainable = False
    #sess = tf.compat.v1.keras.backend.get_session()
    #K.set_session(sess)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    
    opt = keras.optimizers.Adam(learning_rate=lr)
    
    
    jloss= sm.losses.JaccardLoss()
    iou = sm.metrics.IOUScore()
    dl = sm.losses.DiceLoss()
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[iou,jloss])
    
    checkpointer = ModelCheckpoint(filepath="/test/Ito/miccai.hdf5", 
                                    monitor = 'val_iou_score',
                                    verbose=1, 
                                    save_best_only=True,save_weights_only=True,mode='max')
    
    earlystopping = EarlyStopping(monitor='val_iou_score', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    gc.collect()
    
    datagen = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=8,
    zoom_range=1/16)

   
    model.fit(x_tr,y_tr,batch_size=batch_size,epochs=epochs,validation_data=(x_tst, y_tst),callbacks=[checkpointer])
          
    return model

model,_ = resUnet(input_shape,dropout=0.25)
model.load_weights('/test/Ito/unet_weights2_temp2.hdf5')
m2 = transfer(model,input_shape,ld=None,n_classes=1,nb=4,dropout=0.25)
m2.load_weights('/test/Ito/miccai.hdf5')


xt1,yt1,lb1 = load_data(xpath)
np.random.seed(0) 
p = np.random.permutation(len(xt1))
t = 2000
trt = 1500

xtst,ytst,lbst = np.array([xt1[p[-i]] for i in range(1000)]),np.array([yt1[p[-i]] for i in range(1000)]),np.array([lb1[p[-i]] for i in range(1000)])
xtst = xtst.reshape(xtst.shape+(1,))
xtst = xtst.astype('float32')
ytst = ytst.reshape(ytst.shape+(1,))
ytst = ytst.astype('float32')
xtst = xtst/255
ytst = ytst/255

xt,yt,lb = np.array([xt1[p[i]] for i in range(t)]),np.array([yt1[p[i]] for i in range(t)]),np.array([lb1[p[i]] for i in range(t)])
xt = xt.reshape(xt.shape+(1,))
xt = xt.astype('float32')
yt = yt.reshape(yt.shape+(1,))
yt = yt.astype('float32')
xt = xt/255
yt = yt/255

m2 = train_unet(m2,xt,yt,xtst, ytst,lr = 0.0001,epochs = 6,batch_size = 8)


