import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt



# Implement a function sigmoid(x) and a function sigmoidprime(x) (the derivative)

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoidprime(x):
    return sigmoid(x)*(1-sigmoid(x))

###################################################
## data                                          ##
###################################################

def truthtable(n_col):
    """generates data (truthvalue pairs) to train perceptron on"""
    if n_col < 1:
        return [[]]
    subtable = truthtable(n_col-1)
    return [row + [v] for row in subtable for v in [0, 1]]


inputs = np.asarray(truthtable(2))

print(inputs)


###################################################
## labels                                        ##
###################################################

log_operators = {
    'and': lambda x: x[0] and x[1],
    'or': lambda x: x[0] or x[1],
    'nand': lambda x: not (x[0] and x[1]),
    'nor': lambda  x: not (x[0] or x[1]),
    'xor': lambda  x: (x[0] and not x[1]) or (not x[0] and x[1])
}

labels = {}
for key in log_operators:
    labels[key] = []
    for x in inputs: 
        labels[key].append(int(log_operators[key](x)))
    
print(labels)


###################################################
## Perceptron                                    ##
###################################################

class Perceptron():
    
    def __init__(self, input_units, learning_rate=1):
        
        self._weights = rnd.normal(size=input_units)
        self._bias = 1
        self._alpha = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_step(self,inputs):
        self.inputs = inputs
        mat_vec_prod = np.dot(self._weights, np.asarray(inputs))
        return sigmoid(mat_vec_prod + self._bias)




## TESTING
    
p = Perceptron(2)
assert p._weights.size == 2, "Should have a weight per input and a bias."
assert isinstance(p.forward_step([2, 1]), float), "Should activate as scalar."
assert -1 <= p.forward_step([100, 100]) <= 1, "Should activate using sigmoid."
p._weights = np.array([.5, .5])


###################################################
## Perceptron Layer                              ##
###################################################

class PerceptronLayer:
    """Layer of multiple neurons.
    
    Attributes:
        perceptrons (list): List of perceptron instances in the layer.
    """
    def __init__(self, n_perceptrons, n_inputs):
        """Initialize the layer as a list of individual neurons.

        """
        # Set self.perceptrons to a list of Perceptrons
        ### BEGIN SOLUTION
        self.perceptrons = [Perceptron(n_inputs)
                            for _ in range(n_perceptrons)]
        self.inputs = [] # vector of input values of given layer
        self.output = [] # vector of activations values
        self.weightMAT = np.asarray([p._weights for p in self.perceptrons])

        ### END SOLUTION

    

    def activate(self, x):
        """Activate this layer by activating each individual neuron.

        Args:
            x (ndarray): Vector of input values.

        Retuns:
            ndarray: Vector of output values which can be 
            used as input to another PerceptronLayer instance.
        """
        # return the vector of activation values
        ### BEGIN SOLUTION

        self.inputs = x
        self.outputs = np.array([p.forward_step(x) for p in self.perceptrons])

        return self.outputs
        ### END SOLUTION

    def adapt(self,  delta):
        """Adapt this layer by adapting each individual neuron.

        Args:
            x (ndarray): Vector of input values.
            deltas (ndarray): Vector of delta values.
            rate (float): Learning rate.
        """
        print(self.inputs)

        gradients = self.inputs * delta

        print(self.weight_matrix)
        print(gradients)

        self.weightMAT *= gradients

      
    @property
    def weight_matrix(self):
        """Helper property for getting this layer's weight matrix.

        Returns:
            ndarray: All the weights for this perceptron layer.
        """
        


        return np.asarray([p._weights for p in self.perceptrons])

        
    def get_delta(self,error_term):
        """derivative of sigmoid 
        # sigmoidprime is sigmoid(x)*(1-sigmoid(x)) so to be mathematically correct"""
        
        delta =  sigmoid(self.inputs) * (1-sigmoid(self.inputs))

     #   print(deltas)
    #    print(error_term)

        delta *=   2
        delta *=  error_term 
        return delta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    





l = PerceptronLayer(3, 2)


###################################################
## MultilayerPerceptron                          ##
###################################################

class MultilayerPerceptron:
    """Network of perceptrons, also a set of multiple perceptron layers.
    
    Attributes:
        layers (list): List of perceptron layers in the network.
    """
    def __init__(self, n_hidden, n_perceptrons, n_inputs, n_output):
        """Initialize a new network, madeup of individual PerceptronLayers.

        Args:
            n_hidden, how many hidden layers
            n_perceptrons, how many perceptrons per hidden layer
            n_inputs, how many weights for FIRST HIDDEN LAYER !!TODO MAKE SCALABLE 
            n_output, how many output perceptrons 
        """
        self.hidden_layers = [PerceptronLayer(n_perceptrons, n_inputs) 
                            for _ in range(n_hidden)]
                             # NOT SCALABLE TO MULTIPLE LAYERS YET BC N_INPUT 

        self.output_layer = PerceptronLayer(n_output,n_perceptrons)

        self.accuracy = 0
        self.loss = np.inf
                            


    def forward_step(self, x):
        """Activate network and return the last layer's output.

        Args:
            x (ndarray): Vector of input values.

        Returns:
            predicted output 
        """
        # Propagate activation through the network
        for layer in self.hidden_layers:
            x = layer.activate(x)

        x = self.output_layer.activate(x)
        return x        


    def backprop_step(self, x, targets):
        """Adapt the whole network given an input and expected output.

        Args:
            x (ndarray): Vector of input values.
            t (ndarray): Vector of target values (expected outputs).
            rate (float): Learning rate.
        """

        'forward propagation'

        outputs = self.forward_step(x) 

  #      print("oiut", self.output_layer.outputs)
        

        'Error: OutputLayer'

        error_term  = targets - self.output_layer.outputs

        self.accuracy = (targets-self.output_layer.outputs) < 0.5

        self.loss = error_term**2

        delta = self.output_layer.get_delta(error_term)

        # adapt
        self.output_layer.adapt( delta )


        'Error: hidden Layers'

        
        for layer in self.hidden_layers:
            delta = layer.get_delta(error_term)
        ##    print("d")
        #    print(delta)
         ##   print("e")
        #    print(error_term)


            error_term = delta * layer.weight_matrix
            layer.adapt( delta )



    def train(self,inputs,targets,epochs):
        accuracy_per_epoch = []
        loss_per_epoch = []
        

        for epoch in range(epochs):

            accuracy_per_point = []
            loss_per_point = []

            for sample_index in range(len(targets)):

                self.backprop_step(inputs[sample_index], targets[sample_index])
                
                accuracy_per_point.append(MLP.accuracy)
                loss_per_point.append(MLP.loss)
            
            accuracy_per_epoch.append(np.mean(accuracy_per_point[-len(targets):]) )
            loss_per_epoch.append(np.mean(loss_per_point[-len(targets):]))


        return accuracy_per_epoch, loss_per_epoch






if __name__ == "__main__":

    #  def __init__(self, n_hidden, n_perceptrons, n_inputs, n_output):
    MLP = MultilayerPerceptron(1,4,2,1)




    inputs = np.asarray(truthtable(2))

    targets = labels["nand"]

    # prameter init
    n_epochs = 100

    # training
    acc, loss = MLP.train(inputs,targets,n_epochs)

    ## visu
    x = np.linspace(1, n_epochs, n_epochs)
    y_1 = acc
    y_2 = loss


    plt.plot(x,y_1)
    plt.plot(x, y_2)
    plt.legend(['accuracy', 'loss'])
    plt.show()
