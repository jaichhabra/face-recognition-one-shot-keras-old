
# coding: utf-8

# In[6]:


import numpy as np


# In[7]:


import tensorflow as tf


# In[8]:


import keras


# In[9]:


training_dir = "./clean_data"
testing_dir = "./test"
batch_size = 400


# In[21]:


# import matplotlib.pyplot as plt
# %matplotlib inline


# In[27]:



from keras.preprocessing.image import ImageDataGenerator
from random import randint
from collections import Counter


def data(num_of_images=40,width=105,height=105):
#     image_size = width,height

    train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,        zoom_range=0.2,horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(training_dir,        target_size=(width,height),batch_size=batch_size,        class_mode='binary',shuffle=False)

    train = train_generator.next()

    train_x,train_y = train[0],train[1]
    
    if(num_of_images>batch_size):
        exit()

    train_left_input = np.zeros((num_of_images//2,width,height,3))
    train_right_input = np.zeros((num_of_images//2,width,height,3))
    train_output = np.zeros(num_of_images//2,)
    
    for i in range(num_of_images//2):
        train_left_input[i] = train_x[(((i%40)*10)+randint(0,9))%400]
        
        if randint(0,1):
            train_right_input[i] = train_x[(((i%40)*10)+20)%400]
            train_output[i] = 0
        else:
            train_right_input[i] = train_x[(((i%40)*10)+randint(0,9))%400]
            train_output[i] = 1

    print(Counter(train_output))
    return train_left_input,train_right_input,train_output


# In[23]:


# train_left_input,train_right_input,train_output=data(400)


# In[24]:


from collections import Counter
# for i in range(200):
#     print(int(train_output[i]),end="")
# Counter(train_output)


# In[26]:


# for i in range(100,120):
#     plt.imshow(train_left_input[i])
#     plt.show()
#     plt.imshow(train_right_input[i])
#     plt.show()
#     print(train_output[i])

