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
        """
      
        length = x.shape[1]

        # initialize state 
        hidden_state, cell_state = states
        states = tf.TensorArray(tf.float32, size=length)

        # for each time step
        for t in tf.range(length):
            hidden_state, cell_state = self.cell(x[:, t, :], states=(hidden_state, cell_state))
            states = states.write(t, hidden_state)

       # print(out.stack.shape)
        stack = states.stack()

        return stack
       #return tf.reshape(stack,[x.shape[0],x.shape[1],16])



    def zero_states(self, batch_size=32): 
        """
        resets the states of the LSTM
        returns a tuple of states of the appropriate size filled with zeros
        """

        return np.zeros((batch_size, self.cell.units), np.float32), np.zeros((batch_size, self.cell.units), np.float32)
