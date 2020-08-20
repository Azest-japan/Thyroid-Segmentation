import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Input, add, Dropout, Conv2DTranspose, UpSampling2D, concatenate
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

from resnet import resnet


# loss
def triplet_loss(y_true, y_pred):
    del y_true
    
    loss =  K.variable(0, dtype='float32')
    alpha = K.constant(0.2, shape=[1], dtype='float32')
    
    batch_size = len(y_pred)

    for i in range(0, batch_size, 3):
        q_embedding = y_pred[i+0]
        p_embedding =  y_pred[i+1]
        n_embedding = y_pred[i+2]
        D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
        D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
        loss = (loss + alpha + D_q_p - D_q_n )

    loss = loss/(batch_size/3)
    zero = K.constant(0.0, shape=[1], dtype='float32')
    return K.maximum(loss,zero)


def train(n_dims):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255


    model = resnet((28,28,1))
    model.compile('Adagrad', loss = triplet_loss)

    batch_size = 200
    model.fit_generator(generator(X_train, y_train, batch_size),
                        steps_per_epoch=, callbacks=[embed_cb, History()],
                        epochs=25)




#triplet sample
"""
class Select_triplet(Sequence):

    def __init__(self, data, data_classes, batch_size = 900, num_of_class = 10):
        self.data_classes = data_classes
        self.length = len(data)
        self.batch_size = batch_size
        self.num_of_class = num_of_class
        self.num_batches_per_epoch = int((self.length - 1) / batch_size) + 1


    def __getitem__(self, idx):
        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.length:
            end_pos = self.length
        item_paths = self.data_paths[start_pos : end_pos]
        item_classes = self.data_classes[start_pos : end_pos]
        imgs = np.empty((len(item_paths), self.height, self.width, self.ch), dtype=np.float32)
        labels = np.empty((len(item_paths), num_of_class), dtype=np.float32)

        for i, (item_path, item_class) in enumerate(zip(item_paths, item_classes)):
            img, label = _load_data(item_path, item_class, self.width, self.height, self.ch)
            imgs[i, :] = img
            labels[i] = label

        return imgs, labels


    def __len__(self):
        return self.num_batches_per_epoch

    def on_epoch_end(self):
        pass
"""