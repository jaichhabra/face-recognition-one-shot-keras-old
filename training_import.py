from load_data import data
from scipy import ndimage
from model_original import network

from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
from keras.callbacks import Callback
from keras.models import load_model

import tensorflow as tf
import time
def default(entity,value,typeclass):
    if entity.strip()=="":
        return value
    return typeclass(entity)

lr = input("Learning rate(leave blank for default - 0.000001):")
lr=default(lr,0.000001,float)

epochs = input("Epochs(leave blank for default - 2000):")
epochs = default(epochs,2000,int)

validation_split = input("validation_split(leave blank for default - 0.2):")
validation_split = default(validation_split,0.2,float)


timestr = time.strftime("%Y%m%d__%H%M%S")
path = "models/"+timestr+"/"

import os
if not os.path.exists(path):
    os.makedirs(path)

filepath = path + "model.h5"

checkpoint_loss = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint_loss,TQDMCallback()]

train_left_input,train_right_input,train_output = data(50)

from keras.optimizers import Adam
network.compile(optimizer=Adam(lr=lr),loss='binary_crossentropy',metrics=['mae','mse','cosine','accuracy'])

train = network.fit([train_left_input,train_right_input],train_output,epochs=epochs,callbacks=callbacks_list,validation_split=0.20)

accuracy = train.history['acc']
val_accuracy = train.history['val_acc']
loss = train.history['loss']
val_loss = train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

import pickle

if not os.path.exists('traininghistory/'):
    os.makedirs('traininghistory/')

with open('traininghistory/trainHistoryCheckpointFinal+timestr', 'wb') as file_pi:
        pickle.dump(train.history, file_pi)
