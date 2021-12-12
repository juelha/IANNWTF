import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import *
from tensorflow_datasets.core.utils.type_utils import T

from LSTM_Wrapper import *
from DataGenerator import *


class BinaryClassifier:


  def __init__(self, model=None):
    """
    model: Instance of MyModell()-Class
    """
    self.model = model
        
          
  ###################################################
  ## 1 Data set                                    ##
  ###################################################
  def load_data(self):
    """
    initializes the tf_datasets: train_ds, test_ds, validation_ds,
    which have been split from the original dataset using the ratio 
    = 80:10:10

    initializes the threshhold (mean of the target vector), which
    will be used in make_binary()
    """

    data_generator =  DataGenerator()
    dataset = data_generator.dataset


    num_samples = 80000  # todo make dynamic     
    # Split the dataset into a train, test and validation split
    # ratio is 80:10:10
    train_size = int(0.8 * num_samples)
    valid_size = int(0.1 * num_samples)
    test_size = int(0.1 * num_samples)

    train_ds = dataset.take(train_size)
    remaining = dataset.skip(train_size)  
    
    valid_ds = remaining.take(valid_size)
    remaining = remaining.skip(valid_size)

    test_ds = remaining.take(test_size)



    self.train_ds = train_ds
    self.test_ds = test_ds
    self.validation_ds = valid_ds


  def pipeline(self, ds):
    """
    input: tensorflow dataset
    returns: preprocessed and prepared dataset
    """

    # to match states
    #ds = ds.map(lambda seq, label: (tf.dtypes.cast(seq, tf.float64),label))


   # ds = ds.map(lambda features, target: (features, self.make_binary(target)))
    # note: perfomance is better without converting to one_hot
    # tensor = tensor.map(lambda inputs, target: (inputs, tf.one_hot(target,1)))
    #cache this progress in memory
    ds = ds.cache()
    #shuffle, batch, prefetch
    ds = ds.shuffle(1000)
    ds = ds.batch(32)
    ds = ds.prefetch(20)
    return ds



  ###################################################
  ## 3 Training                                    ##
  ###################################################
  def train(self, num_epochs, learning_rate, optimizer_func=SGD):
    """
    all steps needed to train the model of the classifier
    """

    # loading data and splitting into datasets
    self.load_data()

    # pipeline and simplefying target vector to a boolean vector
    train_ds = self.train_ds.apply(self.pipeline)
    test_ds = self.test_ds.apply(self.pipeline)

    ds = train_ds.take(1)  # Only take a single example
    for seq, label in ds:
      print("seq")
      print(seq)
      print("label")
      print(label)

    tf.keras.backend.clear_session()

    # loss function for binary problems
    loss_func = tf.keras.losses.BinaryCrossentropy()

    # trainig model
    self.model.training_loop(train_ds, test_ds, num_epochs, learning_rate, loss_func, optimizer_func)


  ###################################################
  ## Evaluate perfomance                           ##
  ###################################################
  def evalutate(self):
    """
    testing the model with the validation dataset
    (no training here, just a forward pass)
    """

    validation_ds = self.validation_ds.apply(self.pipeline)
    test_loss, test_accuracy =  self.model.test( validation_ds, tf.keras.losses.BinaryCrossentropy())

    return  test_loss,test_accuracy
    



## testing

baseline = BinaryClassifier(LSTM_Wrapper())
# training the model
baseline.train(num_epochs=10, learning_rate=0.01)

fig = baseline.model.visualize_learning()
plt.show()
