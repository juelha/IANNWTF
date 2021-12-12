from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

class LSTM_Cell(tf.keras.Model):
         
    def __init__(self, units):
        """
        """

        super(LSTM_Cell, self).__init__()
        self.units = units


        # gates
        self.input_gate = tf.keras.layers.Dense(units, activation=tf.sigmoid)
        self.cell_state_candidates = tf.keras.layers.Dense(units, activation=tf.tanh)
        self.forget_gate = tf.keras.layers.Dense(units, activation=tf.sigmoid, bias_initializer='ones')
        self.output_gate = tf.keras.layers.Dense(units, activation=tf.sigmoid)


    @tf.function
    # states : (hidden_state, cell_state)   
    def call(self, x, states):
        """
        which is called by providing the input for a
        single times step and a tuple containing (hidden state, cell state)
        
        
        forward propagating the inputs through the network

        input: x, the dataset
        returns: final output
        """

        hidden_state, cell_state = states 

        # concatenate previous hidden state and input
      #  concat_inputs = tf.keras.layers.concatenate([hidden_state, x])

        concat_inputs = tf.concat((x, hidden_state), axis=1) #. Axis 1 = seq_len

        # applying the forget filter to the old cell state Ct−1 via point wise multiplication 
        ft = self.forget_gate(concat_inputs)
        cell_state_update = ft * cell_state

        #do the same with our input filter and the candidate cell state C^t selecting new candidates.
        it = self.input_gate(concat_inputs)
        ct = self.cell_state_candidates(concat_inputs)
        new_candidate = it * ct

        #  We now combine that to form the new cell state Ct:
        new_ct = tf.add(cell_state_update, new_candidate)

        # Determining the hidden state/output
        output = self.output_gate(concat_inputs)
        new_hidden = output *tf.math.tanh(new_ct)

        return  new_hidden, new_ct
