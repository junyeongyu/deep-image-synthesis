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
        self.noise_gen = np.random.normal(0,1,(100,))
        
        self.size = self.height * self.width * self.channels
        
        self.G = self.generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
        self.D = self.descriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
        self.stacked_G_D = self.stacked_g_d()
        self.stacked_G_D.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        
    
    def generator(self):
        model = Sequential()
        model.add(Dense(256, activation="relu", input_shape=(100,)))
        model.add(Dense(256, input_shape=(100,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.size, activation='tanh'))
        model.add(Reshape(self.shape))
        
        model.summary()
        
        return model
    
    def descriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense(self.size, input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int(self.size/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model
    
    def stacked_g_d(self):
        self.D.trainable = False
        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        model.summary()
        return model
    
#    def train(self, X_train, epochs=20000, batch=32, save_interval=200):
#        for cnt in range(epochs):
#            half_batch = int(batch/2)
#            random_index = np.random.randint(0, len(X_train) - half_batch)
#            legit_images = X_train[random_index : random_index + half_batch].reshape(half_batch, self.width, self.height, self.channels)
#            gen_noise = np.random.normal(0, 1, (half_batch,100))
#            syntetic_images = self.G.predict(gen_noise)
#            x_combined_batch = np.concatenate((legit_images, syntetic_images))
#            y_combined_batch = np.concatenate((np.ones((half_batch, 1)), np.zeros((half_batch, 1))))
#            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
#            
#            # train generator
#            noise = np.random.normal(0, 1, (batch, 100))
#            y_mislabled = np.ones((batch, 1))
#            g_loss = self.stacked_G_D.train_on_batch(noise, y_mislabled)
#            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss, g_loss))
#            if cnt % save_interval == 0:
#                self.plot_images(save2file=True, step=cnt)

    
    def train(self, X_train, epochs=20000, batch=32, save_interval=200):
        half_batch = int(batch/2)
        
        for cnt in range(epochs):
            # training for discriminiator
            random_index = np.random.randint(0, len(X_train) - half_batch)
            legit_images = X_train[random_index : random_index + half_batch].reshape(half_batch, self.width, self.height, self.channels)
            gen_noise = np.random.normal(0, 1, (half_batch,100))
            synthetic_images = self.G.predict(gen_noise)
            x_combined_batch = np.concatenate((legit_images, synthetic_images))
            y_combined_batch = np.concatenate((np.ones((half_batch, 1)), np.zeros((half_batch, 1))))
            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
            
            # training for the generator
            noise = np.random.normal(0, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))
            g_loss = self.stacked_G_D.train_on_batch(noise, y_mislabled)
            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss, g_loss))
            
            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=cnt)

    def plot_images(self, save2file=False, samples=16, step=0):
        filename = "./images/mnist_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))
        
        images = self.G.predict(noise)
        plt.figure(figsize=(10,10))
        
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    (X_train, _), (_, _) = mnist.load_data()
    
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train= np.expand_dims(X_train, axis=3)

    gan = GAN()
    gan.train(X_train)
