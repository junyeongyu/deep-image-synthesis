#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Simple implementation of Generative Adversarial Neural Network """

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
    """ Generative Adversarial Network """
    def __init__(self, width=28, height=28, channels=1):
        self.width = width
        self.height = height
        self.channels = channels
        
        self.shape = (self.width, self.height, self.channels)
        self.optimizer = Adam(lr=0.0002, decay=8e-9)
        self.noise_gen = np.random.nromal(0,1,(100,))
        
        self.G = self.generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
        self.D = self.discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
        
    
    def generator(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(100,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.width * self.height * self.channels, activation='tanh'))
        model.add(Reshape((self.width, self.height, self.channels,)))
        model.summary()
        
        return model
    
    def descriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense((self.width * self.height * self.channels), input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int((self.width * self.height * self.channels)/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model
    

if __name__ == '__main__':
    gan = GAN()
    gan.descriminator()
    print("working")
