import keras
from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2

# mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train  = x_train.astype('float32')
x_test   = x_test.astype('float32')
x_train /= 255
x_test  /= 255


# loss
def triplet_loss(y_true, y_pred):
    del y_true
    
    loss =  K.variable(0, dtype='float32')
    g = K.constant(1.0, shape=[1], dtype='float32')
    
    batch_size = len(y_pred)

    for i in range(0, batch_size, 3):
        q_embedding = y_pred[i+0]
        p_embedding =  y_pred[i+1]
        n_embedding = y_pred[i+2]
        D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
        D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
        loss = (loss + g + D_q_p - D_q_n )

    loss = loss/(batch_size/3)
    zero = K.constant(0.0, shape=[1], dtype='float32')
    return K.maximum(loss,zero)



