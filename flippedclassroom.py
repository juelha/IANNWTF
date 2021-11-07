import tensorflow as tf
import tensorflow_datasets as tfds

ds, info = tfds.load('amist', split='train', with_info=True)
print(info)


for elem in ds.take(1):
    print(elem)