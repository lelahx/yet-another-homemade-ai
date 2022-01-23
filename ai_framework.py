import numpy as np
import csv

from typing import Callable
from ast import literal_eval

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

    def __init__(self, n_inputs: int = None, n_neurons: int = None, act: Callable[[np.ndarray], np.ndarray] = ReLU, from_dict: dict = None):
        if from_dict == None:
            self.size = n_neurons
            self.input_space = n_inputs

            self.weights = np.random.random((n_neurons, n_inputs))*2 - 1 # Creates a weights matrix of shape (n_neurons, n_inputs) with random values between -1 and 1
            self.biases = np.zeros(n_neurons) # Creates a biases vector with n_neurons values which are initialized at 0
            self.f_act = act # Defines the activation function of the layer
        else:
            self.size, self.input_space = literal_eval(from_dict['shape'])
            self.weights = np.fromstring(from_dict['weights'], sep='|').reshape((self.size, self.input_space))
            self.biases = np.fromstring(from_dict['biases'], sep='|')

            if from_dict["f_act"] == "sigmoid":
                self.f_act = sigmoid
            else:
                self.f_act = ReLU

    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, key: str) -> np.ndarray:
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
    def __init__(self, *sizes: int, last_act: Callable[[np.ndarray], np.ndarray] = sigmoid, from_file: str = None):
        if from_file == None:
            self.size = len(sizes) - 1
            assert self.size >= 1

            self.layers = [layer(sizes[i], sizes[i + 1]) for i in range(self.size - 1)]
            self.layers.append(layer(sizes[-2], sizes[-1], act=last_act))
        else:
            params = []
            with open(from_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    params.append(row)
            
            self.layers = [layer(from_dict = d) for d in params]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, layer_index: int) -> layer:
        return self.layers[layer_index]

    def store(self, filepath: str = "neuralnet_params.csv"):
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

        values = np.copy(input)
        for current_layer in self.layers:
            values = current_layer.activate(values)
        
        return values