import numpy as np
import csv
from ast import literal_eval
from types import NoneType
from typing import Callable

def ReLU(x: np.ndarray) -> np.ndarray:
    """
    Returns the original array, with negative values replaces by zeros.
    """

    y = np.copy(x)
    y[y < 0] = 0

    return y

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Squishes the real line between 0 and 1 for every value in the given array, and returns a new array with it
    """

    return 1 / (1 + np.exp(-x))

class layer:
    """
    Object defining an individual layer of a neural network.
    """

    def __init__(self, n_inputs: int = None, n_neurons: int = None, weights: np.ndarray = None, biases: np.ndarray = None, act: Callable[[np.ndarray], np.ndarray] = ReLU) -> None:
        """
        Initializes the layer with given coefficients, or with random ones if they aren't provided 
        """

        self.size = n_neurons
        self.input_space = n_inputs
        self.f_act = act # Defines the activation function of the layer

        if type(weights) == NoneType:
            self.weights = np.random.random((self.size, self.input_space))*2 - 1 # Creates a weights matrix of shape (n_neurons, n_inputs) with random values between -1 and 1
        else:
            assert weights.size == self.size*self.input_space # There must be the right number of coefficients in the given weights array
            self.weights = weights.reshape((self.size, self.input_space)) # Reshapes the weights to the correct shape
        
        if type(biases) == NoneType:
            self.biases = np.zeros(n_neurons) # Creates a biases vector with n_neurons values which are initialized at 0
        else:
            assert biases.size == self.size # There must be the same number of biases and neurons
            self.biases = biases

    def __len__(self) -> int:
        """
        Returns the neuron count of the layer
        """

        return self.size
    
    def __getitem__(self, key: str) -> np.ndarray:
        """
        Enables the use of layer['weights'] and layer['biases'] to access these parameters
        """

        if key == "weights":
            return self.weights
        elif key == "biases":
            return self.biases
        else:
            raise KeyError("Unknown parameter for layer object")

    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Activates the layer and returns the values of the neuron.
        """
        
        return self.f_act(self.weights @ inputs + self.biases) # Computes the matrix product of the weights with the inputs, adding the biases and applying the activation function


class network:
    """
    A collection of interconnected layer objects, which can be initialized either from a file or at random
    """
    
    def __init__(self, *sizes: int, last_act: Callable[[np.ndarray], np.ndarray] = sigmoid, from_file: str = None) -> None:
        """
        Initializes a neural network with given sizes and random parameters, or with fully determined ones from a file
        """
        
        if from_file == None:
            self.size = len(sizes) - 1 # First size doesn't count as a layer, it corresponds to the number of inputs
            assert self.size >= 1 # NN should have at least one layer

            self.layers = [layer(sizes[i], sizes[i + 1]) for i in range(self.size - 1)] # Every layer's numer of inputs is the size of the preceding layer
            self.layers.append(layer(sizes[-2], sizes[-1], act=last_act)) # Last layer has special argument, hence has to be defined separately
        else:
            params = []
            with open(from_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    params.append(row)
            
            self.layers = []
            act_funcs = {'ReLU': ReLU, 'sigmoid': sigmoid}
            for d in params:
                s, inp = literal_eval(d['shape'])
                w = np.fromstring(d['weights'], sep='|')
                b = self.biases = np.fromstring(d['biases'], sep='|')
                f = act_funcs[d["f_act"]]

                self.layers.append(layer(inp, s, w, b, f))

    def __len__(self) -> int:
        """
        Returns number of layers of the NN
        """
        
        return self.size

    def __getitem__(self, layer_index: int) -> layer:
        """
        Enables the use of network[n] to access its different layers by their index
        """

        return self.layers[layer_index]

    def store(self, filepath: str = "neuralnet_params.csv") -> None:
        """
        Writes the parameters of the NN to a file for conservation and reuse
        """

        with open(filepath, "w", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["layer", "shape", "weights", "biases", "f_act"])
            writer.writeheader()
            for i in range(self.size):
                writer.writerow({"layer": str(i), 
                                 "shape": str(self.layers[i]['weights'].shape),
                                 "weights": np.array2string(self.layers[i]['weights'].flatten(), 999999, 8, True, '|', threshold=999999)[1:-1],
                                 "biases": np.array2string(self.layers[i]['biases'], 999999, 8, True, '|', threshold=999999)[1:-1],
                                 "f_act": self.layers[i].f_act.__name__})

    def compute(self, input: np.ndarray) -> np.ndarray:
        """
        Runs the input through the NN and outputs the label it generates
        """

        values = np.copy(input)
        for current_layer in self.layers:
            values = current_layer.activate(values)
        
        return values