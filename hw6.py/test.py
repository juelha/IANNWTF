
from PIL import Image
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_addons as tfa
import timeit

device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))


# load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# summarize loaded dataset
print(f'Train: X= {x_train.shape}, y= {y_train.shape}')
print(f'Test: X={x_test.shape}, y={y_test.shape}')
print("Some of the Images in the Dataset are displayed below")

figure = plt.figure(figsize=(12,12))
for i in range(12, 21):
# define subplot
    plt.subplot(3,3,i-11)
# plot raw pixel data
    plt.xlabel(y_train[i])
    plt.imshow(x_train[i])
# show the figure
plt.show()


print(x_train.dtype, x_test.dtype)
# Normalizing data 
# As the Data is of type uint8 we will convert it to Float
x_train = x_train.astype(float)
x_test = x_test.astype(float)
x_train = x_train/255.0
x_test = x_test/255.0



print(x_train[0].shape)

print(x_train[0])

print(y_train[0].shape)

print(y_train[0])


# convert to tensor dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))


ds = train_ds.take(1)  # Only take a single example
for seq, label in ds:
    print("seq")
    print(tf.shape(seq))
    print("label")
    print(tf.shape(label))

# tf.Tensor([32 32  3], shape=(3,), dtype=int32)

# tf.Tensor([1], shape=(1,), dtype=int32)