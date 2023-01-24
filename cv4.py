from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D,MaxPooling2D
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_train(path):
  datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)
  train_datagen_flow = datagen.flow_from_directory(path,target_size=(150, 150),batch_size=16,class_mode='sparse',seed=12345)
  
  return  train_datagen_flow

def create_model(input_shape):
    model = Sequential()
    optimizer = Adam(lr=0.001)
    model.add(Conv2D(filters=6, kernel_size=(3, 3), padding='same', activation = 'relu', input_shape=(input_shape)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='valid', activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Flatten())
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=12, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',metrics=['acc'])
    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=1,steps_per_epoch=None, validation_steps=None):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)
    model.fit(train_data, validation_data=test_data,batch_size=batch_size, epochs=epochs,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,verbose=2)
    return model