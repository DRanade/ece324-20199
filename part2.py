import numpy as np


class ElementwiseMultiply:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if inp.shape == self.weight.shape:
                return inp * self.weight
        else:
            print("Input is not an array of matching shape")
            return False


class AddBias:
    def __init__(self, bias):
        if np.isscalar(bias) and type(bias) in [int, float]:
            self.bias = bias
        else:
            print("Invalid input")

    def __call__(self, inp):
        if isinstance(inp, np.ndarray) or type(inp) in [int, float]:
            return self.bias + inp
        else:
            return False


class LeakyRelu:
    def __init__(self, alpha):
        if np.isscalar(alpha) and type(alpha) in [int, float]:
            self.alpha = alpha
        else:
            print("Invalid input")

    def __call__(self, inp):
        if isinstance(inp, np.ndarray) or type(inp) in [int, float]:
            return np.where(inp >= 0, inp, self.alpha * inp)
        else:
            print("Invalid input")
            return False


class Compose:
    def __init__(self, layers):
        if type(layers) == list:
            self.layers = layers

    def __call__(self, inp):
        out = inp
        for i in range(len(self.layers)):
            print(self.layers[i](out))
            out = self.layers[i](out)
        return out