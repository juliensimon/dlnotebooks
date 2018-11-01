#!/usr/bin/env python3

from __future__ import print_function
import os, sys, json, traceback, gzip
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.utils import multi_gpu_model

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
    
    lr = float(hyperParams.get('lr', '0.001'))
    batch_size = int(hyperParams.get('batch_size', '128'))
    epochs = int(hyperParams.get('epochs', '10'))
    gpu_count = int(hyperParams.get('gpu_count', '0'))

    filter1 = int(hyperParams.get('filter1', '64'))
    filter2 = int(hyperParams.get('filter2', '64'))
    dropout1 = float(hyperParams.get('dropout1', '0.3'))
    dropout2 = float(hyperParams.get('dropout2', '0.3'))

    # Read input data config passed by SageMaker
    with open(data_path, 'r') as params:
        inputParams = json.load(params)
    print("Input parameters: " + str(inputParams))

    num_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_data(input_path)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    
    model.add(Conv2D(filter1, kernel_size=(3, 3), 
                     padding='same',
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(dropout1))

    model.add(Conv2D(filter2, (3, 3),
                     padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(dropout2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    
    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model_name='mnist-cnn-'+str(epochs)
    model.save(model_path+'/'+model_name+'.hd5') # Keras model
    print("Saved Keras model")

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
