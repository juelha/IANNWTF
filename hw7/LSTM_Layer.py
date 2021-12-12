from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from LSTM_Cell import *

class LSTM_Layer(tf.keras.Model): 

    def __init__(self, units=16):   
        """
        """
        super(LSTM_Layer, self).__init__()
        self.cell = LSTM_Cell(units)


    @tf.function
    def call(self, x, states): 
        """
        input over multiple time steps
        input shape = [batch size, seq len, input size]
                        (32, 3, 16)
        output shape [batch size, seq len, output size]

         # states = self.zero_states()

        output_list = []
        #states = tf.TensorArray(tf.float32, size=x.shape[1])

        for index in range(x.shape[1]):
            cell, hidden = self.cell(tf.squeeze(x[:,index,:]), states)
            output_list = np.stack(hidden)
            states = cell, hidden

           # states = states.write(cell, hidden)

       # return states.stack()
        print(output_list.shape)
        return output_list
        """
      
        # input is expected to be of shape [batch size, seq len, input size]
        length = x.shape[1]

        print(x)
        
        print("length", length)

        # initialize state of the simple rnn cell
        (hstate, cstate) = states
        out = []
        states = tf.TensorArray(tf.float32, size=length)

        # for each time step
        for t in tf.range(length):
            input_t = x[:, t, :]
            hstate, cstate = self.cell(input_t, states=(hstate, cstate))
            #out.append(hstate)
            states = states.write(t, hstate)

       # print(out.stack.shape)
        stack = states.stack()

        return tf.reshape(stack,[x.shape[0],3,16])



    def zero_states(self, batch_size=32): 
        """
        resets the states of the LSTM
        returns a tuple of states of the appropriate size filled with zeros
        """

        return np.zeros((batch_size, self.cell.units), np.float32), tf.zeros((batch_size, self.cell.units), np.float32)
