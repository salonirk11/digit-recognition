from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.metrics import categorical_accuracy as accuracy

import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('train.csv')
labels_train = train.ix[:,0].values.astype('int32')
features_train = (train.ix[:,1:].values).astype('float32')
features_test = (pd.read_csv('test.csv').values).astype('float32')


features_train = features_train.reshape(features_train.shape[0], 28, 28, 1)
features_test = features_test.reshape(features_test.shape[0], 28, 28, 1)


# convert list of labels to binary class matrix-one-hot encoding
y_train = np_utils.to_categorical(labels_train)


# pre-processing: divide by max and subtract mean
scale = np.max(features_train)
features_train /= scale
features_test /= scale

mean = np.std(features_train)
features_train -= mean
features_test -= mean

input_dim = features_train.shape[1]
nb_classes = y_train.shape[1]


# Creating MLP
model = Sequential()
model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# training the model
model.fit(features_train, y_train, nb_epoch=10, batch_size=16, validation_split=0.1, verbose=2)

# predicting test results
preds = model.predict_classes(features_test, verbose=0)


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "result-1.csv")