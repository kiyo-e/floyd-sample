
# coding: utf-8

# In[1]:

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator



# In[2]:

img_rows, img_cols, channels = 64, 64, 3
input_shape = (img_rows, img_cols, channels)
classes = 2
batch_size=32
samples= 170
epochs=50


# In[3]:

model = Sequential()


# In[4]:

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))


# In[5]:

model.add(Dense(classes))
model.add(Activation("softmax"))


# In[6]:

model.compile(loss="categorical_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])


# In[7]:

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)


# In[8]:

train_generator = train_datagen.flow_from_directory(
    directory="../input/train",
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        color_mode = 'rgb',
        class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
    directory="../input/valid",
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        color_mode = 'rgb',
        class_mode='categorical')


# In[9]:

early_stopping = EarlyStopping(monitor="val_loss", patience=10)


# In[10]:

model.fit_generator(train_generator,
                   steps_per_epoch=2000,
                   epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=120,
                    callbacks=[early_stopping]
                   )


# In[ ]:

model.save("../output/cnn-dog-cat.h5")


# In[ ]:



