
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import tensorflow as tf


# In[3]:


import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Flatten, Lambda, Input, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
import keras.backend as K

# In[4]:


input_shape = (105,105,3)
left = Input(input_shape)
right = Input(input_shape)

model = Sequential()
model.add(Conv2D(64,input_shape=input_shape,kernel_size=(10,10),activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(128,kernel_size=(7,7),activation='relu',))
model.add(MaxPool2D())

model.add(Conv2D(128,kernel_size=(4,4),activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(128,kernel_size=(4,4),activation='relu'))

model.add(Flatten())
model.add(Dense(4096))

model.summary()

left_out = model(left)
right_out = model(right)

layer_lambda = Lambda(lambda inputs:K.abs(inputs[0]-inputs[1]))
layer_second_last = layer_lambda([left_out,right_out])

out = Dense(1,activation='sigmoid')(layer_second_last)

network = Model(inputs=[left,right],outputs=out)

network.summary()