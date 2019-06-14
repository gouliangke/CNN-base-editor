
import os
import numpy as np
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import keras.backend as K


def build_model(sequ_len=30):
	input = Input(shape=(sequ_len, 4))
	conv1 = Convolution1D(filters=32, kernel_size=3, padding="valid",
		activation="relu",
		strides=1,
		kernel_initializer='glorot_normal', name='conv1')(input)
	conv2 = Convolution1D(filters=32, kernel_size=3, padding="valid",
		activation="relu",
		strides=1,
		kernel_initializer='glorot_normal', name='conv2')(conv1)
	pool1 = AveragePooling1D(pool_size=2, strides=2)(conv2)
# 	conv3 = Convolution1D(filters=64, kernel_size=3, padding="valid",
# 		activation="relu",
# 		strides=1,
# 		kernel_initializer='glorot_normal', name='conv3')(pool1)
# 	conv4 = Convolution1D(filters=64, kernel_size=3, padding="valid",
# 		activation="relu",
# 		strides=1,
# 		kernel_initializer='glorot_normal', name='conv4')(conv3)
# 	pool2 = AveragePooling1D(pool_size=1, strides=1)(conv4)

	mlp = Dense(32, activation='relu')(Flatten()(pool1))
	output = Dense(1, activation='sigmoid')(mlp)

	model = Model(input=input, output=output)
	model.compile(optimizer='adam', loss='binary_crossentropy',
		metrics=['acc'])
	return model
	