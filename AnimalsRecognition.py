# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:09:16 2021

@author: bastd
"""

from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
import numpy as np


def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    TL avec VGG16 + couche complètement connectée
    """
    
    # VGG16 pré-entrainée
    classifier = Sequential()
 
    # Utilisation du réseau VGG16 pré-entrainé 
    classifier.add(VGG16(weights='imagenet', 
                                     include_top = False, 
                                     input_shape = input_shape,
                                     pooling = 'max'))
    # Rendre les couches du réseau VGG16 non-entrainable 
    for layer in classifier.layers:
        layer.trainable = False
        
    # Ajouter le réseaU pout rendre le modèle specifique à la reconnaissance souhaitée
    classifier.add(Dense(units = 256, activation = "relu"))
    #classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 256, activation = "relu"))
    classifier.add(Dropout(0.5)) 
     
    classifier.add(Dense(units = n_classes, activation = "softmax"))

    # Compilation
    classifier.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return classifier

# Création data supplémentaire
train_generator = ImageDataGenerator(rotation_range=90, 
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input)

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

BATCH_SIZE = 10 # Mini-lots


traingen = train_generator.flow_from_directory('dataset/Training',
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               seed=42)

validgen = train_generator.flow_from_directory('dataset/Training',
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=42)

testgen = test_generator.flow_from_directory('dataset/Testing',
                                             target_size=(224, 224),
                                             class_mode=None,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=42)


input_shape = (224, 224, 3)
optim_1 = Adam(learning_rate=0.001)
n_classes=3

n_steps = traingen.samples // BATCH_SIZE
n_val_steps = validgen.samples // BATCH_SIZE
n_epochs = 5

vgg_model = create_model(input_shape, n_classes, optim_1, fine_tune=0)

from keras.callbacks import Callback

plot_loss_1 = Callback()

tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)


early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')


# Entrainement

vgg_model.fit_generator(
        traingen,
        steps_per_epoch=n_steps,
        epochs=n_epochs,
        validation_data=validgen,
        validation_steps=n_val_steps)
