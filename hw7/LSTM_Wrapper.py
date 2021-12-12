from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from LSTM_Layer import *


class LSTM_Wrapper(tf.keras.Model):
    """ final model """

    def __init__(self): 
        super(LSTM_Wrapper, self).__init__()

        self.input_layer = tf.keras.layers.Dense(units=16, activation='sigmoid')
        self.lstm_layer = LSTM_Layer(units=16)
        self.output_layer = tf.keras.layers.Dense(units=2, activation='softmax')


    @tf.function
    def call(self, x):
        """
        init states as zero
        """
        x = self.input_layer(x)
        states = self.lstm_layer.zero_states(x.shape[0])
        x = self.lstm_layer(x, states)
        x = self.output_layer(x)
        return x[-1] # we only want last time step prediction



