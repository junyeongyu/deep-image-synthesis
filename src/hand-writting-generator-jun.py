#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 0. Import  
import numpy as np

from IPython.core.debugger import Tracer

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

import matplotlib.pyplot as plt
plt.switch_backend('agg')


class GAN(object):
    def __init__(self, width=28, height=28, channels=1):
        self.width = width
        self.height = height
        self.channels = channels
        self.shape = (self.width, self.height, self.channels)
        self.size = self.width * self.height * self.channels;
        self.optimiser = Adam(lr=0.0002, decay=8e-9)
        self.noise_gen = np.random.normal(0,1,(100,))
            
    def generator(self):    
        model = Sequential()
        model.add(Dense(256, activation="relu", input_shape=(100,)))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(self.width * self.height * self.channels, activation='tanh'))
        model.add(Reshape((self.width, self.height, self.channels)))
        model.summary()
        return model
    
    def discriminator(self):        
        model = Sequential()
        model.add(Flatten(input_shape=self.shape)) #model.add(Reshape(((self.size), input_shape=self.shape))
        model.add(Dense(self.size    , activation="relu", input_shape=self.shape))
        model.add(Dense(int(self.size / 2), activation="relu"))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model
        

if __name__ == '__main__':
    gan = GAN()
    gan.generator();
    gan.discriminator()



'''
def basic_model_2(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="relu", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(20, activation="relu"))
    t_model.add(Dense(y_size))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)
'''