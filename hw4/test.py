import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Dense, Activation
from tensorflow_datasets.core.utils.type_utils import T

import pandas as pd

###################################################
## 1 Data set                                    ##
###################################################



class Classifier:


  def __init__(self, model=None, train_ds=None, test_ds=None):
      """
      
      """
      #self.model = model
      self.treshhold = 0
      
      #self.model = MyModel(dim_hidden=(2,511),perceptrons_out=10)

      # self.train(num_epochs=30, learning_rate=0.01)
        

  def load_data(self):

    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=";")

    # shuffle first so inputs and labels stay on same row
    data_suffled = data.sample(frac=1)

    # separate into input and labels 
    targets = data_suffled.pop('quality')

    # Split the dataset into a train, test and validation split
    # 80:10:10
    train_ds, test_ds, validation_ds = np.split(data_suffled, [int(.8*len(data_suffled)), int(.9*len(data_suffled))])
    train_tar, test_tar, validation_tar = np.split(targets, [int(.8*len(targets)), int(.9*len(targets))])

    # convert to tensor dataset
    training_ds = tf.data.Dataset.from_tensor_slices((train_ds, train_tar))
    testing_ds = tf.data.Dataset.from_tensor_slices((test_ds, test_tar))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_ds, validation_tar))


    self.treshhold = np.median(train_tar)

    return training_ds, testing_ds

    

  def make_binary(self,target):
    # note: casting to int lowers accuracy
    return(tf.expand_dims(target >= self.treshhold, -1))

  def pipeline(self,tensor):


    tensor = tensor.map(lambda features, target: (features, self.make_binary(target)))

  # perfomance is better without converting to one_hot
  #  tensor = tensor.map(lambda inputs, target: (inputs, tf.one_hot(target,1)))
    
    #cache this progress in memory
    tensor = tensor.cache()
    #shuffle, batch, prefetch
    tensor = tensor.shuffle(50)
    tensor = tensor.batch(32)
    tensor = tensor.prefetch(20)
    #return preprocessed dataset
    return tensor



 

  ###################################################
  ## Main Program                                  ##
  ###################################################
  def train(self, num_epochs, learning_rate):


    # loading 100 000 training examples and 1 000 testing examples as recommended
    train_ds, test_ds = self.load_data()

    train_dataset = train_ds.apply(self.pipeline)
    test_dataset = test_ds.apply(self.pipeline)

    tf.keras.backend.clear_session()

    # Initialize the model based on wether we are allow
    model = MyModel(dim_hidden=(4,12),perceptrons_out=1)

    # trainig model
    tr,te,te_acc = model.training_loop(train_dataset,test_dataset, num_epochs, learning_rate)

    # visualize 
    model.visualize_learning(tr,te,te_acc)


###################################################
## 2 Model Class                                 ##
###################################################

class MyModel(tf.keras.Model):
      
    def __init__(self, dim_hidden, perceptrons_out):
      """
      dim_hidden: dimensions of hidden layers (hardcoded as dense layers)
                  1st arg: n_layers
                  2nd arg: n_perceptrons per layers
      perceptrons_out: n of perceptrons in output layer
      """
      super(MyModel, self).__init__()
      n_layers, n_perceptrons = dim_hidden
      self.hidden = [Dense(n_perceptrons, activation=tf.sigmoid)
                            for _ in range(n_layers)]
      self.out = Dense(perceptrons_out, activation=tf.sigmoid)

      # for visualization of training
      self.test_losses = []
      self.test_accuracies = []
      self.training_losses = []

    @tf.function
    def call(self, x):
      """
      forward propagating the inputs through the network
      """
      for layer in self.hidden:
            x = layer(x)
      x = self.out(x)
      return x       


    ###################################################
    ## 3 Training                                    ##
    ###################################################

    def train_step(self, input, target, loss_function, optimizer):
      """
      implements train step for ONE (1) datasample or batch (of datasamples)
      """
      # loss_object and optimizer_object are instances of respective tensorflow classes
      with tf.GradientTape() as tape:
        prediction = self(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, self.trainable_variables)
      optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      return loss

    def test(self, test_data, loss_function):
      """
      test over complete test data
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

    def training_loop(self, train_dataset, test_dataset, num_epochs, learning_rate ):
      # todo loss func, optimizert

      ### Hyperparameters

      
      # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
      cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
      # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    # optimizer = tf.keras.optimizers.SGD(learning_rate)
      optimizer = tf.keras.optimizers.Adam(learning_rate)

      # Initialize lists for later visualization.
      train_losses = []

      test_losses = []
      test_accuracies = []

      #testing once before we begin
      test_loss, test_accuracy = self.test( test_dataset, cross_entropy_loss)
      test_losses.append(test_loss)
      test_accuracies.append(test_accuracy)

      #check how model performs on train data once before we begin
      train_loss, _ = self.test( train_dataset, cross_entropy_loss)
      train_losses.append(train_loss)

      # We train for num_epochs epochs.
      for epoch in range(num_epochs):
          print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

          #training (and checking in with training)
          epoch_loss_agg = []
          for input,target in train_dataset:
              train_loss = self.train_step( input, target, cross_entropy_loss, optimizer)
              epoch_loss_agg.append(train_loss)
          
          #track training loss
          train_losses.append(tf.reduce_mean(epoch_loss_agg))

          #testing, so we can track accuracy and test loss
          test_loss, test_accuracy = self.test( test_dataset, cross_entropy_loss)
          test_losses.append(test_loss)
          test_accuracies.append(test_accuracy)
      return train_losses, test_losses, test_accuracies

     ###################################################
  ## 4 Visualize                                   ##
    ###################################################

    def visualize_learning(self,train_losses,test_losses,test_accuracies): 
      """
      Visualize accuracy and loss for training and test data.
      """
      plt.figure()
      line1, = plt.plot(train_losses)
      line2, = plt.plot(test_losses)
      line3, = plt.plot(test_accuracies)
      plt.xlabel("Training steps")
      plt.ylabel("Loss/Accuracy")
      plt.legend((line1,line2, line3),("training losses", "test losses", "test accuracy"))
      
      return plt.show()



  


## testing ##
myclassifier = Classifier(MyModel(dim_hidden=(2,511),perceptrons_out=10))

myclassifier.train(num_epochs=30, learning_rate=0.01)
