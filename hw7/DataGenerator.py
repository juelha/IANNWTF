import scipy.integrate as integrate
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Dataset Setup

class DataGenerator:
    '''
    The input consists of noise over a certain
    amount of time steps, let us say that n t denotes the noise signal at time step t.
    The target is again a binary decision on whether the integral of the white noise
    up to that time step t is positive or negative.

    '''
    def __init__(self) :
         
        self.seq_len = 3
        self.num_samples = 80000
        self.dataset = tf.data.Dataset.from_generator(self.wrapper_generator, output_signature= 
                                                                    (tf.TensorSpec(shape=(self.seq_len,1), dtype=tf.float32),
                                                                    tf.TensorSpec(shape=(1), dtype=tf.float32)))


    def integration_task(self, seq_len, num_samples): 
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

    #wrapper_generator()

    
    ###################################################
    ## 4 Visualize                                   ##
    ###################################################
    def visualize_learning(dataset): 
        """
        Visualize accuracy and loss for training and test data.
        """
        plt.figure()
        line1, = plt.plot(dataset)
        #line2, = plt.plot(self.test_losses)
        #line3, = plt.plot(self.test_accuracies)
        plt.xlabel("Training steps")
        plt.ylabel("Loss/Accuracy")
        plt.legend((line1),("training losses"))
    #  plt.title(f'{type_classifier}')
        return plt.figure

