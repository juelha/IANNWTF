
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dense, Flatten, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Concatenate

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np




class TransitionLayer(tf.keras.Model):
  """
  TransitionLayer class
  Defines a transition layer, inheriting from tf.keras.Model.
  """

  def __init__(self, n_filters):
    """ 
    Initializes a transition layer.
    In the transition layer we reduce the number of channels to half of the existing channels.
    Parameters
    ----------
    n_filters : int
        number of filters for each convolutional layer
    """

    super(TransitionLayer, self).__init__()

    self.conv1 = Conv2D(filters=n_filters, kernel_size=(1,1), padding="same")
    self.bn1 = BatchNormalization()
    self.act1 = Activation("relu")
    self.pool1 = AveragePooling2D()

    
  @tf.function
  def call(self, inputs, training=None):
    """ 
    Computes a forward step with the given data
    Parameters
    ----------
    inputs : tf.Tensor
        the input for the model
    training : bool
        true if call has been made from train_step, which tells the batch normalizing layer how to normalize
   
    Returns
    -------
    x : tf.Tensor
        the output of the block
    """

    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = self.act1(x)
    x = self.pool1(x)
    return x




class DenseBlock(tf.keras.Model):
  """
  DenseBlock class
  Defines a dense block, inheriting from tf.keras.Model.
  """

  def __init__(self, n_filters, new_channels):
    """ 
    Initializes a dense block.
    It is made up of convolutions where the original input is concatenated with the output of the convolutions.
    Parameters
    ----------
    n_filters : int
        number of filters for each convolutional layer
    new_channels : int
        number of new channels
    """

    super(DenseBlock, self).__init__()

    self.bn1 = BatchNormalization()
    self.act1 = Activation("relu")
    self.conv1 = Conv2D(filters=n_filters, kernel_size=(1,1), padding="valid")

    self.bn2 = BatchNormalization()
    self.act2 = Activation("relu")
    self.conv2 = Conv2D(filters=new_channels, kernel_size=(3,3), padding="same")

    self.concat = Concatenate(axis=-1)

    
  @tf.function
  def call(self, inputs, training=None):
    """ 
    Computes a forward step with the given data
    Parameters
    ----------
    inputs : tf.Tensor
        the input for the model
    training : bool
        true if call has been made from train_step, which tells the batch normalizing layer how to normalize
   
    Returns
    -------
    x : tf.Tensor
        the output of the model
    """

    x = self.bn1(inputs, training=training)
    x = self.act1(x)
    x = self.conv1(x)

    x = self.bn2(x, training=training)
    x = self.act2(x)
    x = self.conv2(x)

    x = self.concat([x, inputs])

    return x

## testing 
