




class TransitionLayer():

    """
    Instantiates the layers and computations involved in a TransitionLayer from DenseNet for the functional API.
    
    A transition layer is used to reduce the size of the feature maps and halve the number of feature maps.
    
    Args:
    x (KerasTensor) : Input to the transition layer
    """

    def __init__(self, input_shape):
        self.batch_normal = tf.keras.layers.BatchNormalization(epsilon=1.001e-05)
        self.activation = tf.keras.layers.Activation(tf.nn.relu)
    
        self.conv = tf.keras.layers.Conv2D(filters = reduce_filters_to, kernel_size=(1,1), padding="valid", use_bias=False)

        # bottleneck, reducing the number of feature maps
        #(floor divide current number of filters by two for the bottleneck)
        self.bottleneck = input_shape//2
        # reduce the height and width of the feature map (not too useful for low-res input)
        
        self.pool = tf.keras.layers.AvgPool2D(pool_size=(2,2), strides = (2,2), padding = 'valid')
    
    def call(self, x):
        x = self.batch_normal(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.bottleneck(x)

        return x





class DenseBlock(): 

  def __init__(self): 
    super(DenseBlock, self).__init__()
    self.batch_normal =  tf.keras.layers.BatchNormalization(epsilon=1.001e-05)
    self.activation = tf.keras.layers.Activation(tf.nn.relu)
    
    # 1x1 convolution with 128 filters (padding "valid" because with 1x1 we don't need padding)
    self.conv1 = tf.keras.layers.Conv2D(n_filters, kernel_size=(1,1), padding="valid", use_bias=False)
    # 3x3 convolution with 32 filters (to be concatenated with the input)
    self.conv2 = tf.keras.layers.Conv2D(new_channels, kernel_size=(3,3), padding="same", use_bias=False)

  
  def block(self, x):
    x_out = self.batch_normal(x) 
    x_out = self.activation(x_out)
    x_out = self.conv1(x_out)
    x_out = self.batch_normal(x_out)
    x_out = self.activation(x_out)
    x_out = self.conv2(x_out)
    return x_out
  
  def call(self, x):
    # Concatenate layer (just a tf.keras.layers.Layer that calls tf.concat)
    x_out = tf.keras.layers.Concatenate(axis=3)([x, block(x)]) # axis 3 for channel dimension
    return x_out


# Custom Layer
class DenseNet(tf.keras.layers.Layer):

    def __init__(self, units=8):
        super(DenseNet, self).__init__()
        self.units = units
        self.activation = tf.nn.softmax

    def build(self, input_shape): 
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




## testing 


def get_DenseNet121():
    """
    Creates a tf.keras.Model with the functional API that matches the official DenseNet121 in detail.
    
    The architecture is as follows:
    
    (stem of the network)
    zero padding (3,3)
    7x7 conv with strides 2 (valid)

    batchnormalization
    
    relu
    
    zero padding (1,1)
    3x3 max pool, strides 2 (valid)
    
    (after reducing the input size, we now use our dense and transition blocks)
    
     6 Dense Blocks 
     
    Transition layer

    12 Dense Blocks
    
    Transition layer

    24 Dense Blocks
    
    Transition layer

    16 Dense Blocks

    batchnormalization
    
    relu

    global pooling 
    
    (having extracted the feature vector from the image with the DenseBlocks, we apply the classification head)
    
    1000 units dense with softmax (because imagenet has 1000 classes)
    
    """
    # (pseudo input, not used for building subclassed models, only in functional api!)
    x_in = tf.keras.layers.Input(shape= (224,224,3)) # shape of imagenet images
    
    # the stem of the network (used to subsample the image, reducing the size of feature maps)
    # note this is not needed for low res images like in cifar10.
    
    # use extra zero padding because otherwise same padding would be asymmetric (we want it to be symmetric)
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), data_format=None)(x_in)
    
    x = tf.keras.layers.Conv2D(filters = 64, 
                               kernel_size=(7,7), 
                               strides=(2,2),
                               padding="valid",
                              use_bias=False)(x)
    
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-05)(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    
    # use extra zero padding because otherwise same padding would be asymmetric (we want it to be symmetric)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)(x)
    
    x = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")(x)
    
    
    # 6 DenseBlocks forming the first Block
    
    for _ in range(6):
        x = dense_block(x, n_filters=128, new_channels=32)
    
    # Transition layer to reduce resolution and halve number of feature maps
    x = transition_layer(x)
    
    # 12 DenseBlocks forming the second Block
    
    for _ in range(12):
        x = dense_block(x, n_filters=128, new_channels=32)
    
    # Transition layer to reduce resolution and halve number of feature maps
    x = transition_layer(x)
    
    # 24 DenseBlocks forming the third Block
    
    for _ in range(24):
        x = dense_block(x, n_filters=128, new_channels=32)
        
    x = transition_layer(x)
    
    for _ in range(16):
        x = dense_block(x, n_filters=128, new_channels=32)
        
    # this was the last block, so we now simply apply bn and relu
    
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-05)(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    
    # By now, the feature maps are only 7x7 in height and width. 
    # We use global average pooling to transform them into feature vectors that work with Dense Layers.
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Classification head (imagenet has 1000 classes, which means 1000 output units with softmax)
    
    x_out = tf.keras.layers.Dense(1000, activation="softmax")(x)
    
    return tf.keras.Model(x_in, x_out)