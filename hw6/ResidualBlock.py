import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    
    def __init__(self, mode = "normal", input_shape = (32,32,3), n_filters = 32, out_filters = 64):
        """
        init a building block that is used in MyResNet(), 
        consisting of multiple alterations of Convolution and Batch Normalization layers
        
        Args:
        x (tensor) : Input 
        
        n_filters (int) : changes the number of filters used by the first convolutions
        
        out_filters (int) : changes the number of channels of the output

        mode (str):
            "strided", strided convolutions are used and strided 1x1 "pooling" is used to shrink the feature map size
            "normal", we do not change the size of the feature maps but can control the number of channels that we get in the output. 
            "constant", we keep both the size and the number of channels constant.
        """
        
        super(ResidualBlock, self).__init__()

        # Your residualblock should consist of multiple alterations of Convolution and Batch Normalization layers
        self.batch_normal = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(tf.nn.relu)
        self.conv1 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1,1), padding="same", activation="relu")

        self.list_layers = [] # all layers depending on mode
        self.mode = mode 

        if mode == "normal":
            self.list_layers.append(tf.keras.layers.Conv2D(filters=out_filters/2, kernel_size =(3,3), padding="same"))
            self.list_layers.append(tf.keras.layers.Conv2D(filters=out_filters, kernel_size =(3,3), padding="same"))
    
        # some blocks in ResNetV2 have a MaxPool with 1x1 pool size and strides of 2 instead
        elif mode == "strided":
           
            # do strided convolution (reducing feature map size)
            self.list_layers.append(tf.keras.layers.Conv2D(filters=out_filters/2, kernel_size =(3,3), padding="same"))
            self.list_layers.append(tf.keras.layers.Conv2D(filters=out_filters, kernel_size =(3,3), padding="same", strides=(2,2)))

            self.skip_layer = tf.keras.layers.Conv2D(n_filters, (1,1) , strides = (2,2))
            
        # other ResNetV2 blocks keep both the size and channel number constant
        elif mode == "constant":
            self.list_layers.append(tf.keras.layers.Conv2D(filters=out_filters, kernel_size =(3,3), padding="same"))
           


    @tf.function
    def call(self, x):
        """
        output of the block has the same dimensions as its input
        """
        
        # copy tensor to variable called x_skip
        x_skip = x

        # have an initial Conv layer 
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



