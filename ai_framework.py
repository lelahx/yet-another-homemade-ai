import numpy as np
import data_input as dinput

def ReLU(x: np.ndarray):
    """
    Returns the original array, with negative values replaces by zeros.
    """

    y = np.copy(x)
    y[y < 0] = 0

    return y

def sigmoid(x: np.ndarray):
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

        self.size = n_neurons

    def __len__(self):

        return self.size

    def activate(self, inputs: np.ndarray):
        """
        Activates the layer and returns the values of the neuron.
        """
        
        return self.f_act(self.w @ inputs + self.b) # Computes the matrix product of the weights with the inputs, adding the biases and applying the activation function


class network:
    def __init__(self, *sizes: int, last_act=sigmoid):
        assert sizes != []
        print(sizes)

        self.layers = [layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 2)]

        self.layers.append(layer(sizes[-2], sizes[-1], last_act))


    def run(self, input: np.ndarray):

        values = input
        for current_layer in self.layers:
            values = current_layer.activate(values)
        
        return values







