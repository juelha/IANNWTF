import math
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dense, Flatten, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Concatenate

train_download, test_download = tfds.load("cifar10", split=["train", "test"], as_supervised=True)


def pipeline(data):
  """ 
  Prepares the data for being handled in our model
  Parameters
  ----------
  data : tf.Dataset
      dataset returned by the tfds.load function
  
  Returns
  -------
  data : tf.Dataset
      preprocessed dataset
  """

  #convert data from uint8 to float32
  data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))
  #normalize data
  data = data.map(lambda img, target: ((img/255.), target))
  #create one-hot targets
  data = data.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
  #cache
  data = data.cache()
  #shuffle, batch, prefetch
  data = data.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)
  return data



# pipeline and simplefying target vector to a boolean vector
train_ds = train_download.apply(pipeline)
test_ds = test_download.apply(pipeline)

ds = train_ds.take(1)  # Only take a single example
for seq, label in ds:
    print("seq")
    # tf.Tensor([64 32 32  3], shape=(4,), dtype=int32)
    print(tf.shape(seq)) # tf.Tensor([64 32 32  3], shape=(4,), dtype=int32)
    print("label")
    #tf.Tensor([64 10], shape=(2,), dtype=int32)
    print(tf.shape(label))



class ResidualBlock(tf.keras.Model):
  """
  ResidualBlock class
  Defines a residual block, inheriting from tf.keras.Model.
  """

  def __init__(self, n_filters=32):
    """ 
    Initializes a residual block.
    It is made up of convolutional layers and computations to create a residual block.
    Parameters
    ----------
    n_filters : int
        number of filters for each convolutional layer
    """

    super(ResidualBlock, self).__init__()

    self.bn1 = BatchNormalization()
    self.act1 = Activation("relu")
    self.conv1 = Conv2D(filters=n_filters, kernel_size=(1,1))

    self.bn2 = BatchNormalization()
    self.act2 = Activation("relu")
    self.conv2 = Conv2D(filters=n_filters, kernel_size=(3,3), padding="same")

    self.bn3 = BatchNormalization()
    self.act3 = Activation("relu")
    self.conv3 = Conv2D(filters=n_filters, kernel_size=(1,1))

    self.add = Add()

    
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

    x = self.bn3(x, training=training)
    x = self.act3(x)
    x = self.conv3(x)

    x = self.add([x, inputs])
    return x



# testing
class MyModel(tf.keras.Model): 
  def __init__(self ):
    super(MyModel, self).__init__()

    self.block1 = ResidualBlock( )
    self.block2 = ResidualBlock()
  
  def call(self, x):
    x_out = self.block1(x)
    x_out = self.block2(x_out)

    return x_out

image_shape = (1, 32,32,3)
dummy = tf.ones(image_shape)

#resblock_try = ResidualBlock(mode = 'strided', input_shape = (32,32,3))
#out = resblock_try(dummy)

model = MyModel()

output = model(dummy)

print(output)