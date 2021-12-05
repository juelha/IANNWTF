import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils.type_utils import T

from keras.datasets import cifar10
from tensorflow.keras.optimizers import *

from MyResNet import *
from MyDenseNet import *


class MyClassifier:


  def __init__(self, model=None):
    """
    model: Instance of MyResNet()-Class or MyDenseNet()-Class
    """
    self.model = model
        
          
  ###################################################
  ## 1 Data set                                    ##
  ###################################################
  def load_data(self):
    """
    initializes the tf_datasets: train_ds, test_ds
    """
    self.train_ds, self.test_ds  = tfds.load("cifar10", split=["train", "test"], as_supervised=True)


  def pipeline(self,tensor):
    # casting from uint to float
    tensor = tensor.map(lambda seq, label: (tf.dtypes.cast(seq, tf.float32),label))
    # normalizing sequence
    tensor = tensor.map(lambda seq, label: ((seq/255.), label))
    # EachNimage corresponds to one of 10 categories -> depth = 10
    tensor = tensor.map(lambda seq, target: (seq, tf.one_hot(target, depth=10)))
    #cache this progress in memory
    tensor = tensor.cache()
    #shuffle, batch, prefetch
    tensor = tensor.shuffle(1000)
    tensor = tensor.batch(64)
    tensor = tensor.prefetch(tf.data.AUTOTUNE)
    #return preprocessed dataset
    return tensor


  ###################################################
  ## 3 Training                                    ##
  ###################################################
  def train(self, num_epochs, learning_rate, optimizer_func=SGD):
    """
    all steps needed to train the model of the classifier
    """

    # loading data and splitting into datasets
    self.load_data()
    
    # pipeline 
    train_ds = self.train_ds.apply(self.pipeline)
    test_ds = self.test_ds.apply(self.pipeline)

    self.image_shape = (32, 32, 3) 


    tf.keras.backend.clear_session()
    
    # trainig model
    self.model.training_loop(train_ds, test_ds, num_epochs, learning_rate)
   





