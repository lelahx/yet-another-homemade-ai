import numpy as np
import data_input as dinput

def ReLU(x: np.array):
    """
    Returns the original array, with negative values replaces by zeros.
    """
    
    y = np.copy(x)
    y[y < 0] = 0

    return y

def sigmoid(x: np.array):
    """
    Squishes the real line between 0 and 1 for every value in the given array, and returns a new array with it
    """

    return 1 / (1 + np.exp(-x))

class layer:
    """
    Object defining an individual layer of a neural network.
    """

    def __init__(self, n_inputs: int, n_neurons: int, act=ReLU):
        self.w = np.random.random((n_neurons, n_inputs))*2 - 1 # Creates a weights matrix of shape (n_neurons, n_inputs) with random values between -1 and 1
        self.b = np.zeros(n_neurons) # Creates a weights vector with n_neurons values which are initialized at 0
        self.f_act = act # Defines the activation function of the layer

    def activate(self, inputs: dinput.example):
        """
        Activates the layer and returns the values of the neuron.
        """
        
        return self.f_act(self.w @ inputs.values + self.b) # Computes the matrix product of the weights with the inputs, adding the biases and applying the activation function


class network:
    pass


#class neuron:
#    def __init__(self, n, act=ReLU):
#        self.w = np.random.random(n)*2 - 1
#        self.b = 0
#        self.f_act = act
#
#    def activate(self, inputs: dinput.example):
#        return self.f_act(np.dot(self.w, inputs.values) + self.b)