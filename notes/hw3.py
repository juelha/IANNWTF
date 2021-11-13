import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

import numpy as np


###################################################
## 1 Data set                                    ##
###################################################

# loading 100 000 examples for training and 1 000 for testing as recommended
train_ds, test_ds = tfds.load('genomics_ood', split=['train[0:100000]', 'test[0:1000]'], as_supervised=True)

# function that converts the string tensor into a usable tensor that contains the one-hot-encoded sequence
def onehotify(seq):
  vocab = {'A':'1', 'C': '2', 'G':'3', 'T':'0'}
  for key in vocab.keys():
    seq = tf.strings.regex_replace(seq, key, vocab[key])
  split = tf.strings.bytes_split(seq)
  labels = tf.cast(tf.strings.to_number(split), tf.uint8)
  onehot = tf.one_hot(labels, 4)  # groups nucleotides together, with on-value, off-value 'matrix' with depth of 4 
  onehot = tf.reshape(onehot, (-1,))   # flattens into 1-D
  return onehot

"""
ds = train_ds.take(1)  # Only take a single example
for seq, label in ds:

  print(seq)
  print(onehotify(seq))


"""

def pipeline(tensor):
  tensor = tensor.map(lambda seq, label: (onehotify(seq), tf.one_hot(label, 10)))
  #cache this progress in memory, as there is no need to redo it; it is deterministic after all
  tensor = tensor.cache()
  #shuffle, batch, prefetch
  tensor = tensor.shuffle(1000)
  tensor = tensor.batch(32)
  tensor = tensor.prefetch(20)
  #return preprocessed dataset
  return tensor


train_ds = train_ds.apply(pipeline)
test_ds = test_ds.apply(pipeline)



###################################################
## 2 Model Class                                 ##
###################################################

class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = SimpleDense(256)#, activation=tf.sigmoid)
        self.dense2 = SimpleDense(256)#, activation=tf.sigmoid)
        #self.drop = tf.keras.layers.Drop(0.5)
        self.out = SimpleDense(10)#, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
      #  x = self.drop(x)
        x = self.out(x)
        return x


# source keras: https://github.com/keras-team/keras/blob/v2.7.0/keras/layers/core/dense.py#L31-L240 
# Custom Layer
class SimpleDense(tf.keras.layers.Layer):

    def __init__(self, units=8):
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = tf.nn.softmax


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
  """implements train step for ONE (1) datasample or batch (of datasamples)"""
  # loss_object and optimizer_object are instances of respective tensorflow classes
  with tf.GradientTape() as tape: 
    prediction = model(input)
    loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def test(model, test_data, loss_function):
  # test over complete test data

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
## MAIN                                          ##
###################################################

tf.keras.backend.clear_session()

#For showcasing we only use a subset of the training and test data (generally use all of the available data!)
train_dataset = train_ds.take(1000)
test_dataset = test_ds.take(100)

### Hyperparameters
num_epochs = 10
learning_rate = 0.1

# Initialize the model.
model = MyModel()
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

###################################################
## 4 Visualize                                   ##
###################################################

# Visualize accuracy and loss for training and test data.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
line3, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend((line1,line2, line3),("training","test", "test accuracy"))
plt.show()