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


from MyModel import *


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


    


  