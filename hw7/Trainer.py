
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from LSTM_Wrapper import *


   
   
class Trainer():   


    def __init__(self, model):

        self.model = model

        # for visualization of training
        self.test_accuracies = []
        self.test_losses = []
        self.train_losses = []

   
    ###################################################
    ## 3 Training                                    ##
    ###################################################

    def train_step(self, input, target, loss_function, optimizer):
        """
        implements train step for ONE (1) datasample or batch (of datasamples)
        
        returns: loss of one trainig step
        """
        with tf.GradientTape() as tape:
          prediction = self.model(input)
          loss = loss_function(target, prediction)
          gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
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
          
          prediction =  self.model(input)

          print("target") # shape=(32, 1, 2)
          print(target)

          print("prediction") # shape=(32, 2),
          print(prediction)

          sample_test_loss = loss_function(target, prediction)
         # sample_test_accuracy = np.argmax(target, axis=2) == np.argmax(prediction, axis=1)
          sample_test_accuracy = np.argmax(target) == np.argmax(prediction)
          sample_test_accuracy = np.mean(sample_test_accuracy)
          test_loss_aggregator.append(sample_test_loss.numpy())
          test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

        test_loss = tf.reduce_mean(test_loss_aggregator)
        test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

        return test_loss, test_accuracy
      

    ###################################################
    ## Training Loop                                 ##
    ###################################################

    def training_loop(self, train_dataset, test_dataset, num_epochs, learning_rate, loss_function, optimizer_func):
        """
        training of the model 
        initializes the vectors self.test_losses, self.test_accuracies, and self.test_accuracies 
        inputs: train_dataset, test_dataset, num_epochs, learning_rate, loss_function, optimizer_func
        """
        # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
        cross_entropy_loss = loss_function

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
    def visualize_learning(self, type_classifier="so... hows it going?"): 
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