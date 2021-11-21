import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import *
from tensorflow_datasets.core.utils.type_utils import T

from MyModel import *


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

    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", 
        delimiter=";")

    # shuffle first so inputs and targets stay on same row
    data_suffled = data.sample(frac=1)

    # separate into input and targets 
    targets = data_suffled.pop('quality')

    # Split the dataset into a train, test and validation split
    # ratio is 80:10:10
    train_ds, test_ds, validation_ds = np.split(data_suffled, [int(.8*len(data_suffled)), int(.9*len(data_suffled))])
    train_tar, test_tar, validation_tar = np.split(targets, [int(.8*len(targets)), int(.9*len(targets))])

    # convert to tensor dataset
    training_ds = tf.data.Dataset.from_tensor_slices((train_ds, train_tar))
    testing_ds = tf.data.Dataset.from_tensor_slices((test_ds, test_tar))
    validating_ds = tf.data.Dataset.from_tensor_slices((validation_ds, validation_tar))

    self.treshhold = np.median(train_tar)

    self.train_ds = training_ds
    self.test_ds = testing_ds
    self.validation_ds = validating_ds


  def make_binary(self,target):
    """
    is needed to make the non-binary classification problem binary
    input: the target to be simplified 
    returns: boolean 
    """
    # note: casting to integers lowers accuracy
   # return(tf.expand_dims(int(target >= self.treshhold), -1))
    return(tf.expand_dims(target >= self.treshhold, -1))


  def pipeline(self, ds):
    """
    input: tensorflow dataset
    returns: preprocessed and prepared dataset
    """
    ds = ds.map(lambda features, target: (features, self.make_binary(target)))
    # note: perfomance is better without converting to one_hot
    # tensor = tensor.map(lambda inputs, target: (inputs, tf.one_hot(target,1)))
    #cache this progress in memory
    ds = ds.cache()
    #shuffle, batch, prefetch
    ds = ds.shuffle(50)
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
    



  