import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_datasets.core.utils.type_utils import T


###################################################
## 1 Data set                                    ##
###################################################

def onehotify(seq):
  """
  function that converts the string tensor into a 
  usable tensor that contains the one-hot-encoded 
  sequence
  """
  vocab = {'A':'1', 'C': '2', 'G':'3', 'T':'0'}
  for key in vocab.keys():
    seq = tf.strings.regex_replace(seq, key, vocab[key])
  split = tf.strings.bytes_split(seq)
  labels = tf.cast(tf.strings.to_number(split), tf.uint8)
  onehot = tf.one_hot(labels, 4)  # groups nucleotides together, with on-value, off-value 'matrix' with depth of 4 
  onehot = tf.reshape(onehot, (-1,))   # flattens into 1-D
  return onehot

def pipeline(tensor):
  tensor = tensor.map(lambda seq, label: (onehotify(seq), tf.one_hot(label, 10)))
  #cache this progress in memory
  tensor = tensor.cache()
  #shuffle, batch, prefetch
  tensor = tensor.shuffle(1000)
  tensor = tensor.batch(32)
  tensor = tensor.prefetch(20)
  #return preprocessed dataset
  return tensor


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
      self.hidden = [SimpleDense(n_perceptrons, activation=tf.sigmoid)
                            for _ in range(n_layers)]
      self.out = SimpleDense(perceptrons_out, activation=tf.nn.softmax)

    @tf.function
    def call(self, x):
      """
      forward propagating the inputs through the network
      """
      for layer in self.hidden:
            x = layer(x)
      x = self.out(x)
      return x       

# Custom Layer
class SimpleDense(tf.keras.layers.Layer):

    def __init__(self, units, activation=None, use_bias=True):
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias


    def build(self, input_shape): 
        """need build func bc it builds network struct based on input shape"""
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs): 
      x = tf.matmul(inputs, self.w) + self.b
      x = self.activation(x)
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
    sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
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

  # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
  cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
  # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
  optimizer = tf.keras.optimizers.SGD(learning_rate)

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
  train_ds, test_ds = tfds.load('genomics_ood', split=['train[0:100000]', 'test[0:1000]'], as_supervised=True)

  train_ds = train_ds.apply(pipeline)
  test_ds = test_ds.apply(pipeline)

  tf.keras.backend.clear_session()

  # Initialize the model based on wether we are allowed to change parameters
  boost = False 
  if (boost==True): 
    # change parameters here to boost performance 
    model = MyModel(dim_hidden=(2,511),perceptrons_out=10)
  else: 
    # parameters given in pdf
    model = MyModel(dim_hidden=(2,256),perceptrons_out=10)

  # trainig model
  tr,te,te_acc = training_loop(model,train_ds,test_ds, num_epochs=10, learning_rate=0.1)

  # visualize 
  visualize_learning(tr,te,te_acc)

  