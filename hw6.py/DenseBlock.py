
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Activation, Concatenate

from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *



class TransitionLayer(tf.keras.Model):

  def __init__(self, n_filters):
    """ 
    reduce the number of channels to existing channels/2

    n_filters : number of filters for each convolutional layer
    """

    super(TransitionLayer, self).__init__()

    self.conv1 = Conv2D(filters=n_filters, kernel_size=(1,1), padding="same")
    self.bn1 = BatchNormalization()
    self.act1 = Activation("relu")
    self.pool1 = AveragePooling2D()

    
  @tf.function
  def call(self, inputs):
    """ 
    """

    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.pool1(x)
    return x




class DenseBlock(tf.keras.Model):

  def __init__(self, n_filters, new_channels, n_conv=2):
    """ 
    n_conv (int): how many conv layers per block 
    """

    super(DenseBlock, self).__init__()  

    self.list_layers = []
    for _ in range(n_conv):
        self.list_layers.append(BatchNormalization())
        self.list_layers.append(Activation("relu"))
        self.list_layers.append(Conv2D(filters=n_filters, kernel_size=(1,1), padding="valid"))


    self.concate = Concatenate(axis=-1)

    
  @tf.function
  def call(self, inputs):
    """ 
    """
    x = inputs
    for layer in self.list_layers:
        x = layer(x)
    

    x = self.concate([x, inputs])

    return x

## testing 
