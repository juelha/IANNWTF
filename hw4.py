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

def load_data():


  ds = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=";")

  # shuffle first so inputs and labels stay on same row
  ds = ds.sample(frac=1)

  # separate into input and labels 
  targets = ds.pop('quality')

  # Split the dataset into a train, test and validation split
  # 80:10:10
  train_ds, test_ds, validation_ds = np.split(ds, [int(.8*len(ds)), int(.9*len(ds))])
  train_tar, test_tar, validation_tar = np.split(targets.sample(frac=1), [int(.8*len(targets)), int(.9*len(targets))])

  # convert to tensor dataset
  train_ds = tf.data.Dataset.from_tensor_slices((train_ds, train_tar))
  test_ds = tf.data.Dataset.from_tensor_slices((test_ds, test_tar))
  validation_ds = tf.data.Dataset.from_tensor_slices((validation_ds, validation_tar))


  treshhold = np.median(train_tar)

  return train_ds, test_ds, treshhold



def make_binary(target):
  # note: casting to int lowers accuracy
  return(tf.expand_dims((target >= 6), -1))

def pipeline( tensor):


  tensor = tensor.map(lambda features, target: (features, make_binary(target)))

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
## 2 Model Class                                 ##
###################################################


class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation=tf.sigmoid)
        self.dense2 = tf.keras.layers.Dense(16, activation=tf.sigmoid)
        self.out = tf.keras.layers.Dense(1, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x



###################################################
## 3 Training                                    ##
###################################################

def train_step(model, input, target, loss_function, optimizer):
  """
  implements train step for ONE (1) datasample or batch (of datasamples)
  """
  # loss_object and optimizer_object are instances of respective tensorflow classes
  with tf.GradientTape() as tape:
    prediction = model(input)
    loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def test(model, test_data, loss_function):
  """
  test over complete test data
  """


  test_accuracy_aggregator = []
  test_loss_aggregator = []

  for (input, target) in test_data:
    prediction = model(input)
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

def training_loop(model, train_dataset, test_dataset, num_epochs,learning_rate ):
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
  test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
  test_losses.append(test_loss)
  test_accuracies.append(test_accuracy)

  #check how model performs on train data once before we begin
  train_loss, _ = test(model, train_dataset, cross_entropy_loss)
  train_losses.append(train_loss)

  # We train for num_epochs epochs.
  for epoch in range(num_epochs):
      print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

      #training (and checking in with training)
      epoch_loss_agg = []
      for input,target in train_dataset:
          train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
          epoch_loss_agg.append(train_loss)
      
      #track training loss
      train_losses.append(tf.reduce_mean(epoch_loss_agg))

      #testing, so we can track accuracy and test loss
      test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
      test_losses.append(test_loss)
      test_accuracies.append(test_accuracy)
  return train_losses, test_losses, test_accuracies


###################################################
## 4 Visualize                                   ##
###################################################

def visualize_learning(train_losses,test_losses,test_accuracies): 
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


###################################################
## Main Program                                  ##
###################################################

if __name__ == "__main__":

  # loading 100 000 training examples and 1 000 testing examples as recommended
  train_ds, test_ds,treshhold = load_data()

  print(treshhold)


  train_dataset = train_ds.apply(pipeline)
  test_dataset = test_ds.apply(pipeline)

  tf.keras.backend.clear_session()

  # Initialize the model based on wether we are allow
  model = MyModel()# MyModel(dim_hidden=(4,12),perceptrons_out=1)

  # trainig model
  tr,te,te_acc = training_loop(model,train_dataset,test_dataset, num_epochs=10, learning_rate=0.1)

  # visualize 
  visualize_learning(tr,te,te_acc)

  