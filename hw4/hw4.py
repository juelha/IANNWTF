import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Dense, Activation
from tensorflow_datasets.core.utils.type_utils import T

import pandas as pd


from MyModel import *




class BinaryClassifier:


  def __init__(self, model=None):
      """
      
      """

      self.model = model
      
      self.load_data()
      self.treshhold = 0 # needed to make the problem a binary classifiying problem

      self.train_ds = self.train_ds.apply(self.pipeline)
      self.test_ds = self.test_ds.apply(self.pipeline)
      self.validation_ds = self.validation_ds.apply(self.pipeline)



  ###################################################
  ## Train                                         ##
  ###################################################
  def train(self, num_epochs, learning_rate):


    tf.keras.backend.clear_session()


    # trainig model
    self.model.training_loop(self.train_ds, self.test_ds, num_epochs, learning_rate)

    # visualize 
    self.model.visualize_learning()

  ###################################################
  ## 1 Data set                                    ##
  ###################################################
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
    self.train_ds = training_ds
    self.test_ds = testing_ds
    self.validation_ds = validation_ds

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
    return tensor



 





  


## testing ##
# baseline
myclassifier = BinaryClassifier(MyModel(dim_hidden=(1,12),perceptrons_out=1))

# 0.01 does not have any change in accuracy
myclassifier.train(num_epochs=10, learning_rate=0.1)

final_acc = myclassifier.model.test_accuracies[-1]

print(f'Epoch:  ending w  {final_acc}')