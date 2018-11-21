#!/usr/bin/env python3

from __future__ import print_function
import os, sys, json, traceback, gzip
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
import keras.backend as K

# SageMaker paths
prefix      = '/opt/ml/'
input_path  = prefix + 'input/data/'
output_path = os.path.join(prefix, 'output')
model_path  = os.path.join(prefix, 'model')
param_path  = os.path.join(prefix, 'input/config/hyperparameters.json')
data_path   = os.path.join(prefix, 'input/config/inputdataconfig.json')

# Load MNIST data copied by SageMaker
def load_data(input_path):
    # Adapted from https://github.com/keras-team/keras/blob/master/keras/datasets/fashion_mnist.py

    # Training and validation files
    files = ['training/train-labels-idx1-ubyte.gz', 'training/train-images-idx3-ubyte.gz',
             'validation/t10k-labels-idx1-ubyte.gz', 'validation/t10k-images-idx3-ubyte.gz']
    # Load training labels
    with gzip.open(input_path+files[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # Load training samples
    with gzip.open(input_path+files[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    # Load validation labels
    with gzip.open(input_path+files[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # Load validation samples
    with gzip.open(input_path+files[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    print("Files loaded")
    return (x_train, y_train), (x_test, y_test)

# Main code
try:
    # Read hyper parameters passed by SageMaker
    with open(param_path, 'r') as params:
        hyperParams = json.load(params)
    print("Hyper parameters: " + str(hyperParams))
    
    lr          = float(hyperParams.get('lr', '0.1'))
    batch_size  = int(hyperParams.get('batch_size', '128'))
    epochs      = int(hyperParams.get('epochs', '10'))
    gpu_count   = int(hyperParams.get('gpu_count', '0'))

    batch_norm  = int(hyperParams.get('batch_norm', '1'))
    filters1    = int(hyperParams.get('filters1', '64'))
    filters2    = int(hyperParams.get('filters2', '64'))
    fc1         = int(hyperParams.get('fc1', '256'))
    fc2         = int(hyperParams.get('fc2', '64'))
    dropout1    = float(hyperParams.get('dropout1', '0.2'))
    dropout2    = float(hyperParams.get('dropout2', '0.2'))
    dropout_fc1 = float(hyperParams.get('dropout_fc1', '0.2'))
    dropout_fc2 = float(hyperParams.get('dropout_fc2', '0.2'))
    
    # Read input data config passed by SageMaker
    with open(data_path, 'r') as params:
        inputParams = json.load(params)
    print("Input parameters: " + str(inputParams))

    # input image dimensions
    img_rows, img_cols = 28, 28

    # Split data between train and test sets
    (x_train, y_train), (x_test, y_test) = load_data(input_path)

    # Depending on backend, put channels first (TF) or last (MXNet)
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # Normalize pixel values
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    # Convert class vectors to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    
    # Build model    
    model = Sequential()
    
    # 1st convolution block
    model.add(Conv2D(filters1, kernel_size=(3,3), padding='same', input_shape=input_shape))
    if batch_norm == 1:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Dropout(dropout1))
    
    # 2nd convolution block
    model.add(Conv2D(filters2, kernel_size=(3,3), padding='valid'))
    if batch_norm == 1:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Dropout(dropout2))

    # 1st fully connected block
    model.add(Flatten())
    model.add(Dense(fc1))
    if batch_norm == 1:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_fc1))

    # 2nd fully connected block
    model.add(Dense(fc2))
    if batch_norm == 1:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_fc2))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
    
    #sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # Define callback for early stopping
    early_stopping = EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=30,
                              verbose=1, mode='auto')
    
    # Define custom callback to log best validation accuracy
    # This is needed for HPO to grab the metric in the training log
    class LogBestValAcc(Callback):
        def on_train_begin(self, logs={}):
            self.val_acc = []
        def on_train_end(self, logs={}):
            print("Best val_acc:", max(self.val_acc))
        def on_epoch_end(self, batch, logs={}):
            self.val_acc.append(logs.get('val_acc'))
            
    best_val_acc = LogBestValAcc()
    
    # Define callback to save best epoch
    checkpointer = ModelCheckpoint(filepath=model_path+'/'+'mnist-cnn.hd5',
                                   monitor='val_acc', verbose=1, save_best_only=True)
    
    datagen = ImageDataGenerator(
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True)

    # compute quantities required for featurewise normalization
    # std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)
    
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test), 
                    epochs=epochs,
                    steps_per_epoch=len(x_train) / batch_size,
                    callbacks=[early_stopping, best_val_acc, checkpointer],
                    verbose=1)
    
    model.fit(x=x_train, y=y_train, batch_size=batch_size, 
              validation_data=(x_test, y_test), epochs=epochs,
              callbacks=[early_stopping, best_val_acc, checkpointer],verbose=1)
    
    print("Saved Keras model to %s" % (model_path+'/'+'mnist-cnn.hd5'))

    sys.exit(0)
except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
