import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Flatten, BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *

from DenseBlock import *

class MyDenseNet(tf.keras.Model):
    """
    Model: "my_dense_net"
    _________________________________________________________________   
    Layer (type)                Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)             multiple                  896       
                                                                    
    dense_block (DenseBlock)    multiple                  21376     
                                                                    
    transition_layer (Transitio  multiple                 10560     
    nLayer)                                                         
                                                                    
    dense_block_1 (DenseBlock)  multiple                  25600     
                                                                    
    batch_normalization_5 (Batc  multiple                 768       
    hNormalization)                                                 
                                                                    
    global_average_pooling2d (G  multiple                 0         
    lobalAveragePooling2D)                                          
                                                                    
    flatten (Flatten)           multiple                  0         
                                                                    
    dense (Dense)               multiple                  1930      
                                                                    
    =================================================================
    Total params: 94,410
    Trainable params: 93,386
    Non-trainable params: 1,024
    _________________________________________________________________
    """

    def __init__(self, n_denseblocks = 2, n_filters = 128, n_channels = 32, growth_rate = 32 ):
        """ 
        structure taken from pdf instructions 
        """
        super(MyDenseNet, self).__init__()

        # init conv layer 
        self.input_layer = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same",  input_shape=(32, 32, 3))

        self.denseblocks = []

        #alternately append a Dense Block and A Transition Layer
        for _ in range(n_denseblocks-1):
            self.denseblocks.append(DenseBlock(n_filters, n_channels))
            self.denseblocks.append(TransitionLayer(n_filters=growth_rate*2))

        #You will not want to include a Transition Layer after the last Dense Block
        self.denseblocks.append(DenseBlock(n_filters, n_channels))

        # readout layers
        self.bn1 = BatchNormalization()
        self.globalpool = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.out = tf.keras.layers.Dense(10, activation="softmax")
        
        # for visualization of training
        self.test_accuracies = []
        self.test_losses = []
        self.train_losses = []
        
    @tf.function
    def call(self, inputs):
        """ 
        
        """
        
        # init conv layer
        x = self.input_layer(inputs)

        # denseblocks
        for i in range(len(self.denseblocks)):
            x = self.denseblocks[i](x)
        
        # readout layers
        x = self.bn1(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.out(x)
        return x



    ###################################################
    ## 3 Training                                    ##
    ###################################################

    def train_step(self, input, target, loss_function, optimizer):
        """
        implements train step for ONE (1) datasample or batch (of datasamples)
        
        returns: loss of one trainig step
        """
        with tf.GradientTape() as tape:
            prediction = self(input)
            loss = loss_function(target, prediction)
            gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def test(self, test_data, loss_function):
        """
        forward pass of test_data 
        accuracy and loss not tracked during pass, but calculated from 
        final output

        inputs: data to be tested, loss_function defined in training_loop()
        returns: the loss and accuracy of the data
        """
        test_accuracy_aggregator = []
        test_loss_aggregator = []

        for (input, target) in test_data:
            prediction = self(input)


            sample_test_loss = loss_function(target, prediction)
            sample_test_accuracy =  target == np.round(prediction, 0)
            sample_test_accuracy = np.mean(sample_test_accuracy)
            test_loss_aggregator.append(sample_test_loss.numpy())
            test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

        test_loss = tf.reduce_mean(test_loss_aggregator)
        test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

        return test_loss, test_accuracy


    ###################################################
    ## Training Loop                                 ##
    ###################################################

    def training_loop(self, train_dataset, test_dataset, num_epochs, learning_rate, optimizer_func=Adam):
        """
        training of the model 
        initializes the vectors self.test_losses, self.test_accuracies, and self.test_accuracies 
        inputs: train_dataset, test_dataset, num_epochs, learning_rate, loss_function, optimizer_func
        """
        # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
        # cross_entropy_loss = loss_function
        # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
        cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()


        optimizer = optimizer_func(learning_rate)

        #testing once before we begin
        test_loss, test_accuracy = self.test( test_dataset, cross_entropy_loss)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_accuracy)

        #check how model performs on train data once before we begin
        train_loss, _ = self.test(train_dataset, cross_entropy_loss)
        self.train_losses.append(train_loss)

        # We train for num_epochs epochs.
        for epoch in range(num_epochs):
            print(f'Epoch: {str(epoch)} starting with accuracy {self.test_accuracies[-1]}')

            #training (and checking in with training)
            epoch_loss_agg = []
            for input,target in train_dataset:
                train_loss = self.train_step( input, target, cross_entropy_loss, optimizer)
                epoch_loss_agg.append(train_loss)
            
            #track training loss
            self.train_losses.append(tf.reduce_mean(epoch_loss_agg))

            #testing, so we can track accuracy and test loss
            test_loss, test_accuracy = self.test( test_dataset, cross_entropy_loss)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)


    ###################################################
    ## 4 Visualize                                   ##
    ###################################################
    def visualize_learning(self, type_classifier): 
        """
        Visualize accuracy and loss for training and test data.
        """
        plt.figure()
        line1, = plt.plot(self.train_losses)
        line2, = plt.plot(self.test_losses)
        line3, = plt.plot(self.test_accuracies)
        plt.xlabel("Training steps")
        plt.ylabel("Loss/Accuracy")
        plt.legend((line1,line2, line3),("training losses", "test losses", "test accuracy"))
        plt.title(f'{type_classifier}')
        return plt.figure

