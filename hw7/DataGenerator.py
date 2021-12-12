import scipy.integrate as integrate
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DataGenerator:
    '''
    The input consists of noise over a certain
    amount of time steps, let us say that n t denotes the noise signal at time step t.
    The target is again a binary decision on whether the integral of the white noise
    up to that time step t is positive or negative.

    '''
    def __init__(self) :
        """
        seq_len = 1 #sequence length, thus the number of time steps we are considering

        """
         
        self.seq_len = 5 #
        self.num_samples = 80000
        self.dataset = tf.data.Dataset.from_generator(self.wrapper_generator, output_signature= 
                                                            (tf.TensorSpec(shape=(self.seq_len,1), dtype=tf.float32),
                                                            tf.TensorSpec(shape=(1), dtype=tf.float32)))

    def integration_task(self, seq_len, num_samples): 
        """
        target, namely if the sum of the noise signal is greater or smaller than 1
        """
        for sample in range(num_samples):
            noise_signal=np.array([])
            for signal in range(seq_len):
                noise_signal = np.append(noise_signal, np.random.normal(loc=0, scale=2)) # noise signal ints of length seq_len
            target = np.array(int(np.sum(noise_signal)>0))
            noise_signal = np.expand_dims(noise_signal,-1) 
            target = np.expand_dims(target,-1)
            #print(noise_signal.shape, target.shape)
            yield noise_signal, target

    def wrapper_generator(self): 
        return self.integration_task(self.seq_len,self.num_samples)
