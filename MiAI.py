import numpy as np

##################
# Model Definition
##################
class Model(object):
    def __init__(self):
        self.layers = []
    
    # forward pass
    def __call__(self, x):
        for layer in self.layers:
            if layer.type == "Dense": # need input for dense layer only
                layer.input = x
            x = layer(x)
            layer.output = x
        return x
    
    # backpropagation
    def backprop(self, Loss, lr):
        
        delta = Loss.backward()

        for layer in reversed(self.layers):
            if layer.type == "Dense":
                delta, dW, dB = layer.backward(delta)
                layer.weights -= lr * dW
                layer.bias -= lr * dB
            else:
                delta = layer.backward(delta)

    # Summarize the network
    def summarize(self):
        print("Model Summary")
        print("===================================")
        for layer in self.layers:
            if layer.type == "Softmax" or layer.type == "Sigmoid" or layer.type == "ReLU":
                print(layer.get_name())
            else:
                print(layer.get_name())
                print("Weights:", "(" + str(layer.weights.shape[0]) + ", " + str(layer.weights.shape[1]) + ")")
            print("********************")

#######################
# Neural Network Layers
#######################
class Dense():
    def __init__(self, input_dim, output_dim, bias=True):
        self.type = "Dense"
        self.input = None
        self.output = None
        
        self.weights = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim) if bias else None
                
    # forward propagation
    def __call__(self, X):
        self.input = X

        matrix_multiplication = np.einsum("ij, nj -> ni", self.weights, X)
        
        if self.bias is None:
            self.output = matrix_multiplication
        else:
            self.output = matrix_multiplication + self.bias[np.newaxis, :]

        return self.output
        
    # backpropagation
    def backward(self, delta):
        
        dW = np.matmul(delta.T, self.input) / delta.shape[0]
        dB = np.sum(delta, axis=0) / delta.shape[0]
        
        delta = np.einsum("ij, ni -> nj", self.weights, delta)
        
        return delta, dW, dB
    
    def get_name(self):
        return "Dense Layer"

class ReLU():
    def __init__(self):
        self.type = "ReLU"
        self.output = None

    def __call__(self, X):
        self.output = np.maximum(0, X)
        return self.output
    
    def backward(self, delta):
        return np.multiply(delta, np.heaviside(self.output, 0))
        
    def get_name(self):
        return "ReLU Activation Function"

class Softmax():
    def __init__(self) -> None:
        self.type = "Softmax"
        self.output = None

    def __call__(self, X):
        self.output = np.divide(np.exp(X), np.sum(np.exp(X), axis=1)[:, np.newaxis])
        return self.output
    
    def backward(self, delta):
        pass

    def get_name(self):
        return "Softmax Activation Function"
    
class Sigmoid():
    def __init__(self) -> None:
        self.type = "Sigmoid"
        self.output = None

    def __call__(self, X):
        self.output = np.divide(1, 1 + np.exp(-X))
        return self.output
    
    def backward(self, delta):
        return np.multiply(delta, np.multiply(self.output, (1 - self.output)))

    def get_name(self):
        return "Sigmoid Activation Function"

################
# Lost Functions
################
class BCE():
    def __init__(self):
        self.real = None
        self.prediction = None

    def __call__(self, real, prediction):
        self.real = real
        self.prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
        return -np.mean(self.real * np.log(self.prediction) + (1 - self.real) * np.log(1 - self.prediction))
    
    def backward(self):
        return -(self.real / self.prediction) + ((1 - self.real) / (1 - self.prediction))