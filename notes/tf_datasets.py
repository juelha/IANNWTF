

###################################################
## data sets                                     ##
###################################################

'self made iterable'

# generating data
Inputs = [(1,1),(1,0),(0,1),(0,0)]

def data_generator_and():
    bools = [1,0]
    while True:
        for elem in Inputs:
            x1, x2 = elem
            target = x1 and x2
            yield ((x1, x2), target)


# from any iterable
import tensorflow as tf
                                             # func that returns python iterable 
dataset_and = tf.data.Dataset.from_generator(data_generator_and, 
                                            output_signature = # tells tf what kind of data to expect
                                                (tf.TensorSpec(shape=(2,), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)))

dataset_and = dataset_and.take(8)
for elem in dataset_and:
    print(elem)



'tensorflow_datasets'
# take away: w tf datasets we only change the data when we actually use it

import tensorflow as tf
import tensorflow_datasets as tfds

# load data

ds_mnist, info = tfds.load('mnist', split='train', shuffle_files=True, with_info=True, as_supervised=True)
fig = tfds.show_examples(ds_mnist, info)

# take a look at data

print(info)
"""
look at info > features > "image":
    Image(shape=(28, 28, 1),  # 28 by 28 pixels, 
                              # colorchannel depth of 1 (greyscale)
    dtype=tf.uint # data comes in unsigned integer 8 format
                  # unsigned: positive
                  # 8: 8 bit (from 0 to 255)
"""
# OR
for elem in ds_mnist.take(1):
    img, target = elem
    print(f'Image shape: {img.shape} || Image datatype: {img.dtype}  Target shape: {target.shape}, Target datatype: {target.dtype}')


# reshape data 

ds_tensorflow = ds_mnist.map(lambda img, target: img)
ds_numpy = [elem[0].numpy() for elem in ds_mnist]

for elem in ds_tensorflow:
  print(elem.shape)
  break
for elem in ds_numpy:
  print(elem.shape)
  break


'experiment: np vs tf dataset'
import time

#time it!
time_before_tf = time.time()
ds_tensorflow_plusone = ds_tensorflow.map(lambda img: img+1)
time_after_tf = time.time()
time_elapsed_tf = time_after_tf - time_before_tf

time_before_np = time.time()
ds_numpy_plusone = [img + 1 for img in ds_numpy]
time_after_np = time.time()
time_elapsed_np = time_after_np - time_before_np

print(f'Time elapsed for TF dataset: {time_elapsed_tf} || Time elapsed for np dataset: {time_elapsed_np}')


def iterate_through_dataset(dataset):
  for elem in dataset:
    a = 1+2
    pass

#time it!
time_before_tf = time.time()
iterate_through_dataset(ds_tensorflow)
time_after_tf = time.time()
time_elapsed_tf = time_after_tf - time_before_tf

time_before_np = time.time()
iterate_through_dataset(ds_numpy)
time_after_np = time.time()
time_elapsed_np = time_after_np - time_before_np



















