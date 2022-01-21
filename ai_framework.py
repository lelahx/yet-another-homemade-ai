import numpy as np
import data_input as dinput

def ReLU(x: np.array):
    return np.maximum(np.zeros(x.shape), x)

def sigmoid(x: np.array):
    return 1 / (1 + np.exp(-x))

class layer:
    def __init__(self, n_inputs: int, n_neurons: int, act=ReLU):
        self.w = np.random.random((n_neurons, n_inputs))*2 - 1
        self.b = np.zeros(n_neurons)
        self.f_act = act

    def activate(self, inputs: dinput.example):
        return self.f_act(self.w @ inputs.values + self.b)


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