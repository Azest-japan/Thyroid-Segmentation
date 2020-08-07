import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, add, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


# Building the model
# resnet block
def resunit(inp,nf=32,Ksize=3,padding='same',strides=1,BN='True',BN_first=True,activation='relu',sno='0'):
    # sno = serial number of the unit
    if strides == 1 and int(sno) < 50:
        def conv(x):
            c = Conv2D(nf,
                        kernel_size=3,
                        strides=strides,
                        padding=padding,
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4),
                        name='C'+sno)(x)
            return c

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
def resnet(input_shape, embedding_dim = 32, nf=32, dropout=0.25):
    # nb defines the number of resnet blocks
    #Ksize=3
    nf = 32   # number of filters at the start


    inputs = Input(input_shape,name='inp')
    x = inputs

    for i in range(3):
        y = resunit(x,nf=nf,Ksize=5,padding='same',strides=1,BN_first=False,activation='relu',sno='0_'+str(i))
        y = Conv2D(nf,kernel_size=5,strides=1,padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='C01_'+str(i))(y)
        x = add([x, y],name='A01_'+str(i))    
        x = Activation('relu',name='Act01_'+str(i))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    nf = 2*nf
    y = resunit(x,nf=nf,Ksize=3,padding='same',strides=1,BN_first=False,activation='relu',sno='1_0')
    y = Conv2D(nf,kernel_size=3,strides=1,padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='C11_'+str(0))(y)
    x = Conv2D(nf,kernel_size=1,padding='same',strides=1,name='1C')(x)
    x = add([x, y],name='A11_'+str(0))    
    x = Activation('relu',name='Act11_'+str(0))(x)

    for i in range(2):
        y = resunit(x,nf=nf,Ksize=3,padding='same',strides=1,BN_first=False,activation='relu',sno='1_'+str(i+1))
        y = Conv2D(nf,kernel_size=3,strides=1,padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='C11_'+str(i+1))(y)
        x = add([x, y],name='AA1_'+str(i+1))    
        x = Activation('relu',name='Act11_'+str(i+1))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)


    for i in range(3):
        y = resunit(x,nf=nf,Ksize=3,padding='same',strides=1,BN_first=False,activation='relu',sno='2_'+str(i+1))
        y = Conv2D(nf,kernel_size=3,strides=1,padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='C21_'+str(i))(y)
        x = add([x, y],name='A21_'+str(i+1))    
        x = Activation('relu',name='Act21_'+str(i+1))(x)


        
    x1 = Flatten()(x)
    x1 = Dropout(dropout)(x1)
    outputs = Dense(embedding_dim, activation='sigmoid',kernel_initializer='he_normal', name='out')(x1)
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = resnet((28,28,1))
model.summary()

