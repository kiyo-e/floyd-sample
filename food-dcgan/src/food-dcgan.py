
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import argparse
import math

import os
from scipy import ndimage, misc


# In[2]:


def generator_model():
    model = Sequential()
    model.add(Dense(units=1024, input_dim=100))
    model.add(Activation('tanh'))
    model.add(Dense(128*16*16))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((16, 16, 128), input_shape=(128*16*16,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (5, 5), padding="same"))
    model.add(Activation('tanh'))
    return model


# In[3]:


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(
                        64, (5, 5),
                        padding="same",
                        input_shape=(64, 64, 3)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


# In[4]:


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


# In[33]:


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] =             img[ :, :, :]
    return image


# In[34]:


BATCH_SIZE=128


# In[35]:


discriminator = discriminator_model()
generator = generator_model()


discriminator_on_generator =     generator_containing_discriminator(generator, discriminator)
    
    
    
d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)


generator.compile(loss='binary_crossentropy', optimizer="SGD")

discriminator_on_generator.compile(
    loss='binary_crossentropy', optimizer=g_optim)

discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

noise = np.zeros((BATCH_SIZE, 100))


# In[36]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[39]:


for epoch in range(100):
    index = 0
    print("Epoch is", epoch)
    for X_train in test_datagen.flow_from_directory('../input/images', target_size=(64, 64), 
                                                             color_mode="rgb", batch_size=BATCH_SIZE, class_mode=None):

        print("Number of batches", index)
        index += 1
        # BATCH_SIZEは文字通りのbatch size
        #for index in range(int(X_train.shape[0]/BATCH_SIZE)):
        for i in range(BATCH_SIZE):
              noise[i, :] = np.random.uniform(-1, 1, 100)
            #image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
        
            #ランダムに生成したnosizeで予測
        generated_images = generator.predict(noise, verbose=0)
        
            # 生成したものを20個ごとに保存しているだけ
        if index % 20 == 0:
            image = combine_images(generated_images)
            image = image*127.5+127.5
            Image.fromarray(image.astype(np.uint8)).save(
                "../output/"+str(epoch)+"_"+str(index)+".png")

                
            # 参考にした画像を保存
        #image_batch=np.transpose(X_train,(0,2,3,1))

        image_batch=X_train
        if index % 20 == 0:
            image_real = combine_images(image_batch)
            image_real = image_real*127.5+127.5
            Image.fromarray(image_real.astype(np.uint8)).save(
                "../output/target"+str(epoch)+"_"+str(index)+".png")
        
        X = np.concatenate((image_batch, generated_images))
        y = [1] * int(len(X) / 2) + [0] * int(len(X) / 2)
        
        
            #discriminatorで学習
        d_loss = discriminator.train_on_batch(X, y)
        print("batch %d d_loss : %f" % (index, d_loss))
        
        
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
            
            # discriminatorの学習が進まないように修正
        discriminator.trainable = False
        
        
        g_loss = discriminator_on_generator.train_on_batch(
            noise, [1] * BATCH_SIZE)
        
            # discriminatorが再度学習するように
        discriminator.trainable = True
        
        print("batch %d g_loss : %f" % (index, g_loss))
        if index % 10 == 9:
            generator.save_weights('../output/generator', True)
            discriminator.save_weights('../output/discriminator', True)                    


# In[ ]:





# In[ ]:





# In[ ]:




