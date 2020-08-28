import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import metrics

from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Input, add, Dropout, Conv2DTranspose, UpSampling2D, concatenate
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from resnet import resnet

from collections import Counter 
import random
import itertools

def pairwise_distances(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)
    # print(square_norm)
    # Compute the pairwise distance matrix
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    distances = tf.maximum(distances, 0.0)
    return distances

"""
def make_digitlist(classes):
    digit_list = []
    unique_digit, idx = tf.unique(classes)
    print('unique_digit')
    # print(type(unique_digit.get_shape()))
    print(tf.shape(unique_digit).numpy())
    # print(int(unique_digit.get_shape()[0]))
    # print(type(int(tf.shape(unique_digit)[0])))
    #for digit in range(tf.unstack(tf.shape(unique_digit))[0]):
    # print(type(tf.where(tf.equal(classes, digit))))
"""


def judge_semihard_negative(D_a_p , D_a_n, alpha=0.2):    
    alpha = K.constant(alpha, shape=[1], dtype='float32')
    zero = K.constant(0.0, shape=[1], dtype='float32')
    # loss = alpha + loss + D_a_p - D_a_n
    if D_a_n > D_a_p and D_a_n < alpha + D_a_p:
        return True
    else:
        return False

def calc_loss(D_a_p , D_a_n, alpha=0.2):    
    alpha = K.constant(alpha, shape=[1], dtype='float32')
    zero = K.constant(0.0, shape=[1], dtype='float32')
    loss = alpha + loss + D_a_p - D_a_n
    return loss


# loss
# @tf.function
def triplet_loss(y_true, y_pred):
    # tf.print(y_true)
    # print(type(y_true))
    # tf.config.run_functions_eagerly(True)
    print('Eager:',tf.executing_eagerly())
    classes = tf.squeeze(y_true)
    distances = pairwise_distances(y_pred)

    loss =  K.variable(0, dtype='float32')


    #anchor-positive 作成
    # make_digitlist(classes)
    idx_list = []
    a_p_list = []
    for i in range(10):
        idxs = [i*20 + j for j in range(20)]
        idx_list.append(idxs)
        a_p_list += list(itertools.permutations(idxs,2))

    idx_list = np.array(idx_list)
    batch_size = len(a_p_list)

    for a_p in a_p_list:
        # tf.print(classes[a_p[0]],classes[a_p[1]])
        D_a_p = distances[a_p[0], a_p[1]]
        a_class = int(classes[a_p[0]].numpy())

        n_list = np.where(classes.numpy() == a_class)
        
        semi_hard_list = []
        for neg in n_list:
            D_a_n = distances[a_p[0], neg]
            if judge_semihard_negative(D_a_p , D_a_n) == True:
                semi_hard_list.append(neg)
        semi_hard = random.choice(semi_hard_list)
        
        D_a_n = distances[a_p[0], semi_hard]
        loss += calc_loss(D_a_p, D_a_n)

    return loss / batch_size


class batch_generator(Sequence):
    def __init__(self, data, data_classes, batch_size = 200, num_of_class = 10):
        self.data = data
        self.data_classes = data_classes
        self.length = len(data)
        self.batch_size = batch_size
        self.num_of_class = num_of_class
        self.num_batches_per_epoch = int((self.length) / batch_size)
        self.num_pic_per_class = int(batch_size / num_of_class)

    def __getitem__(self, idx):

        X = []
        Y = []
        start_pos = self.num_pic_per_class * idx
        end_pos = start_pos + self.num_pic_per_class

        for digit in range(self.num_of_class):
            digit_loc = list(np.where(self.data_classes == digit)[0])[start_pos : end_pos]
            x = [self.data[i] for i in digit_loc]
            y = [digit] * self.num_pic_per_class
            # tf.print(x)
            X += x
            Y += y

        X = np.array(X)
        Y = np.array(Y)
        # tf.print(X)
        return X, Y

    def __len__(self):
        return self.num_batches_per_epoch
    def on_epoch_end(self):
        pass

def train(n_dims):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255


    model = resnet((28,28),embedding_dim = n_dims)
    model.compile('Adagrad', loss = triplet_loss)

    batch_size = 200
    num_of_class = 10
    gen = batch_generator(x_train, y_train, batch_size, num_of_class)
    model.fit_generator(gen, steps_per_epoch = 1, epochs=1)


train(32)


"""
def train(n_dims):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255


    model = resnet((28,28),embedding_dim = n_dims)
    model.compile('Adagrad', loss = triplet_loss)

    batch_size = 200
    num_of_class = 10
    gen = batch_generator(x_train, y_train, batch_size, num_of_class)
    model.fit_generator(gen, steps_per_epoch = 1, epochs=1)

def train(n_dims):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    model = resnet((28,28),embedding_dim = n_dims)
    model.compile('Adagrad', loss = triplet_loss)

    batch_size = 200
    num_of_class = 10
    pic_class_batch = batch_size / num_of_class

    pic_per_class = int(min(Counter(y_train).values())//pic_class_batch*pic_class_batch)

    X = []
    Y = []

    for digit in range(num_of_class):
        digit_loc = list(np.where(y_train == digit)[0])[:pic_per_class]
        x = [x_train[i] for i in digit_loc]
        y = [digit] * pic_per_class
        X += x
        Y += y
    print(X)
    print(Y)
    gen = batch_generator(X, Y, batch_size, num_of_class)
    model.fit_generator(gen, steps_per_epoch = 1, epochs=1)

"""