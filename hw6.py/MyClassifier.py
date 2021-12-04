import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils.type_utils import T

from keras.datasets import cifar10
from tensorflow.keras.optimizers import *

from MyResNet import *


class MyClassifier:


  def __init__(self, model=None):
    """
    model: Instance of MyResNet()-Class
    """
    self.model = model
        
          
  ###################################################
  ## 1 Data set                                    ##
  ###################################################
  def load_data(self):
    """
    initializes the tf_datasets: train_ds, test_ds, validation_ds,
        """


    
    self.train_ds, self.test_ds  = tfds.load("cifar10", split=["train", "test"], as_supervised=True)




  def pipeline(self,tensor):

    tensor = tensor.map(lambda seq, label: (tf.dtypes.cast(seq, tf.float32),label))

    tensor = tensor.map(lambda seq, label: ((seq/255.), label))


    #flatten the images into vectors
    #tensor = tensor.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    #create one-hot targets 

      # EachN image corresponds to one of 10 categories.
    tensor = tensor.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

   # tensor = tensor.map(lambda seq, label: (seq, tf.one_hot(label, depth=10)))
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

    

    # pipeline and simplefying target vector to a boolean vector
    train_ds = self.train_ds.apply(self.pipeline)
    test_ds = self.test_ds.apply(self.pipeline)

    ds = train_ds.take(1)  # Only take a single example
    for seq, label in ds:
      print("seq")
      print(tf.shape(seq))
      print("label")
      print(tf.shape(label))  

      self.image_shape = tf.shape(seq)

    self.image_shape = (32, 32, 3) 


    tf.keras.backend.clear_session()

    # loss function for binary problems
   # loss_func = tf.keras.losses.BinaryCrossentropy()

    # testinh ###

    #image_shape = tf.shape(train_ds)

    #resblock_try = ResidualBlock(mode = 'strided', input_shape = (32,32,3))
    #out = resblock_try(dummy)

    self.model = MyResNet(self.image_shape)

    # trainig model
    self.model.training_loop(train_ds, test_ds, num_epochs, learning_rate)


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
    




# testing
myclassifier = MyClassifier()
myclassifier.train(num_epochs=10,learning_rate=1)

print("okay")