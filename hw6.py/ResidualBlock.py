import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam


# Custom Layer
class ResidualBlock(tf.keras.layers.Layer):
    
    def __init__(self, mode = "normal", n_filters = 32,input_shape = (32,32,3), out_filters = 64):
        
        """
        Instantiates the layers and computations involved in a residual block from ResNet V2.
        
        Args:
        x (KerasTensor) : Input to the block
        
        n_filters (int) : changes the number of filters used by the first convolutions
        
        out_filters (int) : changes the number of channels of the output
        
        mode (str) : either "normal", "strided" or "constant". See the markdown text above.

        three different kinds of ResBlocks
        When mode is set to 
            "strided", strided convolutions are used and strided 1x1 "pooling" is used to shrink the feature map size. With mode set to 
            "normal", we do not change the size of the feature maps but can control the number of channels that we get in the output. With mode set to
            "constant", we keep both the size and the number of channels constant.
        """
        
        super(ResidualBlock, self).__init__()

        self.batch_normal = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(tf.nn.relu)
        self.conv1 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1,1), padding="same", activation="relu")

        self.list_layers = []
        self.transform_original = []
        self.mode = mode

        if mode == "normal":
          self.list_layers.append(tf.keras.layers.Conv2D(filters=out_filters/2, kernel_size =(3,3), padding="same"))
          self.list_layers.append(tf.keras.layers.Conv2D(filters=out_filters, kernel_size =(3,3), padding="same"))

          # transform original input to also have 256 channels
          self.transform_original.append(tf.keras.layers.Conv2D(filters=out_filters, kernel_size=(1,1)))
    
        # some blocks in ResNetV2 have a MaxPool with 1x1 pool size and strides of 2 instead
        elif mode == "strided":
           
            # do strided convolution (reducing feature map size)
            self.list_layers.append(tf.keras.layers.Conv2D(filters=out_filters/2, kernel_size =(3,3), padding="same"))
            self.list_layers.append(tf.keras.layers.Conv2D(filters=out_filters, kernel_size =(3,3), padding="same", strides=(2,2)))

            self.skip_layer = tf.keras.layers.Conv2D(n_filters, (1,1) , strides = (2,2))

            # transform original input with 1x1 strided max pooling to match output shape
            self.transform_original.append(tf.keras.layers.Conv2D(filters=out_filters, kernel_size=(1,1)))
            self.transform_original.append(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
            
        # other ResNetV2 blocks keep both the size and channel number constant
        elif mode == "constant":
            self.list_layers.append(tf.keras.layers.Conv2D(filters=out_filters, kernel_size =(3,3), padding="same"))
            self.transform_original.append(tf.keras.layers.Conv2D(filters=out_filters, kernel_size =(1,1), kernel_initializer="Ones", padding="same", trainable = False ))



    @tf.function
    def call(self, x):




        ###
        # copy tensor to variable called x_skip
        x_skip = x
        # have an initial Conv layer before the first res block (increasing the n of channels)
        x = self.conv1(x) 

        for layer in self.list_layers:
          x = layer(x)


        if self.mode == "strided":
          # Processing Residue with conv(1,1)
          x_skip = self.skip_layer(x_skip)
        
        
        # Add Residue
        x = tf.keras.layers.Concatenate(axis=-1)([x, x_skip])     
        x = tf.keras.layers.Activation('relu')(x)


        return x




# testing
class MyModel(tf.keras.Model): 
  def __init__(self, image_shape):
    super(MyModel, self).__init__()

    self.block1 = ResidualBlock(mode = 'strided', input_shape = image_shape)
    self.block2 = ResidualBlock(mode = 'normal', input_shape = image_shape)
  
  def call(self, x):
    x_out = self.block1(x)
    x_out = self.block2(x_out)

    return x_out
## testing ###
image_shape = (1, 32,32,3)
dummy = tf.ones(image_shape)

#resblock_try = ResidualBlock(mode = 'strided', input_shape = (32,32,3))
#out = resblock_try(dummy)

model = MyModel(image_shape)

output = model(dummy)

print(output)
