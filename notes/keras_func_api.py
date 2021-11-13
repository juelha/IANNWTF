
"""implementing a NN w tensorflow: tf.keras"""
# tf.keras as it is the recommended higher level API.
# makes code more concise and flexible
# s: https://jaredwinick.github.io/what_is_tf_keras/
# s: https://www.tensorflow.org/guide/keras/functional 
# tut s: https://www.youtube.com/watch?v=EDQK2JCbE9M


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# TODO: specify shape of input
inputs = tf.keras.Input(shape=(3,)) # e.g. one dim tensor w 3 elem

outputs = layers.Dense(2)( layers.Dense(4)( layers.Dense(8) (inputs))) # applying Dense layers to input
                                                  # order of application = scope of brackets
                                                  # first 8 neuron layer, then 4 ,..
                                                  # Dense = neuron is connected to every neuron in prev layer

model = tf.keras.Model(inputs=inputs, outputs = outputs)

model.compile()

# model.fit(...)

# model.predict(...)




"""implementing a NN w tensorflow: tf.keras w Custom Model"""
# Custom Model
class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(2, activation=tf.nn.softmax)

    def call(self, inputs): # forward pass
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x


model = MyModel()

x = tf.constant([1.0,2.5, 3.8]) # some made up input

model(x)

# Custom Layer
class SimpleDense(tf.keras.layers.Layer):

    def __init__(self, units=8):
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = tf.nn.softmax


    def build(self, input_shape): 
        """need build func bc it builds network struct based on input shape"""
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs): 
        x = tf.matmul(inputs, self.w) + self.b
        x = self.activation(x)
        return x


"""end """
