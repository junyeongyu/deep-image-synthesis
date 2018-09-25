#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 0. Import  
import numpy as np
import time
from IPython.core.debugger import Tracer
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.optimizers import Adam, RMSprop
from keras.layers import BatchNormalization

import matplotlib.pyplot as plt
plt.switch_backend('agg')

class GAN(object):
    def __init__(self, width=28, height=28, channels=1):
        self.width = width
        self.height = height
        self.channels = channels
        self.shape = (self.width, self.height, self.channels)
        self.size = self.width * self.height * self.channels;
        self.noise_gen = np.random.normal(0,1,(100,))
        
        self.G = self.generator()
        #self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.D = None
        self.D = self.discriminator_model()
        self.D.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0002, decay=6e-8))
        self.A = None
        self.A = self.adversarial_model()
        self.A.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001, decay=3e-8))
        
    def generator(self):
        depth = 64+64+64+64
        dim = 7
        model = Sequential()
        # In: 100
        # Out: dim x dim x depth
        model.add(Dense(dim*dim*depth, input_dim=100, activation='relu'))
        model.add(Reshape((dim, dim, depth)))
        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(depth/2), 5, padding='same', activation='relu'))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(depth/4), 5, padding='same', activation='relu'))
        model.add(Conv2DTranspose(int(depth/8), 5, padding='same', activation='relu'))
        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        model.add(Conv2DTranspose(1, 5, padding='same', activation='sigmoid'))
        model.summary()
        return model;
    
    def discriminator(self):
        depth = 64
        model = Sequential()
        model.add(Conv2D(depth*1, 5, strides=2, input_shape=self.shape, padding='same', activation=LeakyReLU(alpha=0.2)))
        model.add(Conv2D(depth*2, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
        model.add(Conv2D(depth*4, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
        model.add(Conv2D(depth*8, 5, strides=1, padding='same', activation=LeakyReLU(alpha=0.2)))
        
        # Out: 1-dim probability
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        return model;

    def discriminator_model(self):
        if self.D:
            return self.D
        model = Sequential()
        model.add(self.discriminator())        
        return model

    def adversarial_model(self):
        if self.A:
            return self.A
        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        return model
    
    def train(self, X_train, epochs=200000, batch=256, save_interval = 20):
        for cnt in range(epochs):
            half_batch = int(batch/2)
            random_index = np.random.randint(0, len(X_train) - half_batch)
            legit_images = X_train[random_index : random_index + half_batch].reshape(half_batch, self.width, self.height, self.channels)
            gen_noise = np.random.normal(0, 1, (half_batch,100))
            syntetic_images = self.G.predict(gen_noise)
            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((half_batch, 1)), np.zeros((half_batch, 1))))
            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
            
            # train generator
            noise = np.random.normal(0, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))
            g_loss = self.A.train_on_batch(noise, y_mislabled)
            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss, g_loss))
            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=cnt)

    def plot_images(self, save2file=False, samples=16, step=0):
        filename = "./images/mnist_cnn_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))
        images = self.G.predict(noise)
        plt.figure(figsize=(10, 10))
        
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
    
    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    
    gan = GAN()
    #gan.generator()
    #gan.discriminator()
    
    gan.train(X_train)
    
'''
    def discriminator(self):        
        model = Sequential()
        model.add(Flatten(input_shape=self.shape)) #model.add(Reshape(((self.size), input_shape=self.shape))
        model.add(Dense(self.size    , activation="relu", input_shape=self.shape))
        model.add(Dense(int(self.size / 2), activation="relu"))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model
'''