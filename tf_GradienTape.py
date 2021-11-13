"""
TensorFlow provides the tf.GradientTape API for automatic differentiation; 
that is, computing the gradient of a computation with respect to some inputs, usually tf.Variables.
TensorFlow "records" relevant operations executed inside the context of a tf.GradientTape onto a "tape". TensorFlow then uses that tape to compute the gradients of a "recorded" computation using reverse mode differentiation.
s: https://www.tensorflow.org/guide/autodiff 
s script: https://github.com/Spinkk/TeachingTensorflow/blob/main/basics/Automatic%20Differentiation%20with%20tf.GradientTape().ipynb 
"""


import tensorflow as tf
import numpy as np

"""
prob: have a dataset which can be described as f(x)=ax where a is 
the parameter we want to learn

here: a = pi
data set is sampled from the function f(x) = pi*x
"""

# a simple linear univariate model function without bias
def model(x, parameter):
    return x * parameter

# generate data (X) and targets (Y)
X = tf.random.uniform((20,1), minval= 0, maxval = 10)
Y = X * np.pi

# initialize parameter variable to a value far away from pi
parameter_estimate = tf.Variable(7.5, trainable=True, dtype=tf.float32)

# set learning rate
learning_rate = tf.constant(0.005, dtype=tf.float32)


#iterate over epochs
for epoch in range(2):

    # iterate over training examples
    for x,y in zip(X,Y):
        
        # within GradientTape context manager, calculate loss between targets and prediction
        with tf.GradientTape() as tape:

            prediction = model(x, parameter_estimate)

            loss = (prediction - y)**2

        # outside of context manager, obtain gradients with respect to list of trainable variables
        gradients = tape.gradient(loss, [parameter_estimate])

        # apply gradients scaled by learning rate to parameters
        new_parameter_val = parameter_estimate - learning_rate * gradients

        # assign new parameter values
        parameter_estimate.assign(new_parameter_val[0])



tf.print(parameter_estimate)



print(parameter_estimate == np.pi)

