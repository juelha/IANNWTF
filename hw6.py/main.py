import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_datasets.core.utils.type_utils import T



# data ------------------------------------------------------------------------------------------------

# loading 100 000 training examples and 1 000 testing examples as recommended
train_ds, test_ds = tfds.load('Cifar10', split=['train[0:50000]', 'test[0:1000]'], as_supervised=True)



def normalize(seq):

  return seq/255


def pipeline(tensor):
  tensor = tensor.map(lambda seq, label: (normalize(seq), tf.one_hot(label, 1)))
  #cache this progress in memory
  tensor = tensor.cache()
  #shuffle, batch, prefetch
  tensor = tensor.shuffle(1000)
  tensor = tensor.batch(64)
  tensor = tensor.prefetch(20)
  #return preprocessed dataset
  return tensor

ds = train_ds.take(1)  # Only take a single example
for seq, label in ds:

  print(seq)
  print(label)
  print(normalize(seq))



# pipeline and simplefying target vector to a boolean vector
train_ds = train_ds.apply(pipeline)
test_ds = test_ds.apply(pipeline)










print("okay")