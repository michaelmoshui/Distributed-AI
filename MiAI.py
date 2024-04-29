import numpy as np

##################
# Model Definition
##################
class Model():
    def __init__(self):
        '''
        Purpose:
        ~ Initialize a neural network by adding layer objects into the self.layers list
        ~ Implementation in inherited classes

        Attributes:
        ~ self.layers: a list of neural network layers
        '''
        self.layers = []
    
    # forward pass
    def __call__(self, x):
        '''
        Purpose:
        ~ make a forward pass through the neural network

        Input:
        ~ x: model input;
        ~ an ndarray of the shape N X D, where N is the size of the sample batch and D is the number of features for one sample

        Output:
        ~ prediction made by the neural network;
        ~ an ndarray of the shape N X O, where N is the size of the sample batch and O is the output dimension of the final layer
        '''
        for layer in self.layers:
            if layer.type == "Dense": # need input for dense layer only
                layer.input = x
            x = layer(x)
            layer.output = x
        return x
    
    # backpropagation
    def backprop(self, Loss, lr):
        '''
        Purpose:
        ~ update weights and biases through backpropagation

        Input:
        ~ lr: learning rate
        ~ Loss: loss function object 

        Output:
        None
        '''
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
        '''
        Purpose:
        ~ Display the shapes of all the layers in the model

        Input:
        None

        Output:
        None
        '''
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
        '''
        Purpose:
        ~ Initialize a Dense layer object

        Input:
        ~ input_dim: input dimension
        ~ output_dim: output dimension
        ~ bias: default to True

        Output:
        None

        Attributes:
        ~ self.type: Dense
        ~ self.input: a (N, I) input; where N is batch size and I is input dimension
        ~ self.output: a (N, O) output; where N is batch size and I is output dimension
        ~ self.weights: a (O, I) weight matrix; where O is output dimension and I is input dimension
        ~ self.bias: an (O,) vector; where O is output dimension
        '''
        self.type = "Dense"
        self.input = None
        self.output = None
        
        self.weights = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim) if bias else None
                
    # forward propagation
    def __call__(self, X):
        '''
        Purpose:
        ~ forward pass of the Dense layer

        Input:
        ~ X: input sample batch;
        ~ an N X I array where I is input dimension

        Output:
        ~ self.output
        ~ N outputs each with dimension O
        '''
        self.input = X

        # compute matrix multiplication Weight X Input for every one of the N input vectors
        matrix_multiplication = np.einsum("ij, nj -> ni", self.weights, X)
        
        # depending on wanting bias or not, add bias to every output vector
        if self.bias is None:
            self.output = matrix_multiplication
        else:
            self.output = matrix_multiplication + self.bias[np.newaxis, :]

        return self.output
        
    # backpropagation
    def backward(self, delta):
        '''
        Purpose:
        ~ back propagation for Dense Layer

        Input:
        ~ delta: error from the previous layer
        ~ has shape (N, O), where O is the output dimension of this Dense layer

        Output:
        ~ delta: error of this layer
        ~ has shape (N, I), where I is the input dimension of this Dense layer
        
        ~ dW: gradient matrix for weights in this layer
        ~ has shape (O, I); same as self.weights

        ~ dB: gradient vector for bias term in this layer
        ~ has shape (O,); same as self.bias
        '''
        
        # compute dW with delta X input averaged over all N matrix multiplications
        dW = np.matmul(delta.T, self.input) / delta.shape[0] 
        dB = np.sum(delta, axis=0) / delta.shape[0]
        
        delta = np.einsum("ij, ni -> nj", self.weights, delta)
        
        return delta, dW, dB
    
    def get_name(self):
        return "Dense Layer"

class ReLU():
    def __init__(self):
        '''
        Purpose:
        ~ Initialize a ReLU layers

        Attributes:
        ~ self.type: ReLU
        ~ self.output: a (N, D) matrix, where D is the dimension of the input and output
        '''
        self.type = "ReLU"
        self.output = None

    def __call__(self, X):
        '''
        Purpose:
        ~ forward pass of the ReLU activation function

        Inputs:
        ~ X: a (N, D) input

        Outputs:
        ~ same dimension as X but transformed under element-wise max(0, X)
        '''
        self.output = np.maximum(0, X)
        return self.output
    
    def backward(self, delta):
        '''
        Purpose:
        ~ perform backward pass of the ReLU activation function

        Inputs:
        ~ delta: the (N, D) error from the higher layers

        Outputs:
        ~ returns an element-wised multiplication of delta and derivative of this layer, which is the error for this layer
        ~ shape is (N, D)
        '''
        return np.multiply(delta, np.heaviside(self.output, 0))
        
    def get_name(self):
        return "ReLU Activation Function"

class Softmax():
    def __init__(self) -> None:
        '''
        Purpose:
        ~ Initialize the softmax activation

        Attributes:
        ~ self.type: Softmax
        ~ self.output: a (N, D) matrix, where D is the dimension of the input and output
        '''
        self.type = "Softmax"
        self.output = None

    def __call__(self, X):
        '''
        Purpose:
        ~ Calls the softmax activation function

        Input and Output:
        ~ (N, D) matrix that gets transformed according to the softmax activation function definition
        '''
        self.output = np.divide(np.exp(X), np.sum(np.exp(X), axis=1)[:, np.newaxis])
        return self.output
    
    # Not implemented yet!
    def backward(self, delta):
        pass

    def get_name(self):
        return "Softmax Activation Function"
    
class Sigmoid():
    def __init__(self) -> None:
        '''
        Purpose:
        ~ Initiate the sigmoid activation function

        Attributes:
        ~ self.type: Sigmoid
        ~ self.output: (N, D) matrix, where the columns are probability values between 0 and 1
        '''
        self.type = "Sigmoid"
        self.output = None

    def __call__(self, X):
        '''
        Purpose:
        ~ calls the sigmoid activation function and transform logits into probability values between 0 and 1

        Input and Output:
        ~ matrix of shape (N, D) for N probability predictions with 3 categories of predictions
        '''
        self.output = np.divide(1, 1 + np.exp(-X))
        return self.output
    
    def backward(self, delta):
        '''
        Purpose:
        ~ Completes the backward pass of the sigmoid activation function

        Input:
        ~ Delta: (N, D) error from the previous layer

        Output:
        ~ returns element-wised multiplication between Delta and derivative of sigmoid activation
        ~ (N, D)
        '''
        return np.multiply(delta, np.multiply(self.output, (1 - self.output)))

    def get_name(self):
        return "Sigmoid Activation Function"

################
# Lost Functions
################
class BCE():
    def __init__(self):
        '''
        Purpose:
        ~ Initialize a binary-crossentropy function object

        Attributes:
        ~ self.real: (N, D) array of the real labels
        ~ self.prediction: (N, D) array of predictions made by the neural network
        '''
        self.real = None
        self.prediction = None

    def __call__(self, real, prediction):
        '''
        Purpose:
        ~ calculate the binary crossentropy loss

        Attributes:
        ~ real: real labels
        ~ predictions: predictions made by the neural network
        '''
        self.real = real
        self.prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
        print(self.real.shape, self.prediction.shape)
        return -np.mean(np.add(np.multiply(self.real, np.log(self.prediction)), np.multiply((1 - self.real), np.log(1 - self.prediction))))
    
    def backward(self):
        '''
        Purpose:
        ~ compute the gradient of the loss function

        Output:
        ~ (N, D) matrix containing N samples of a N-dimensional label
        '''
        return -(self.real / self.prediction) + ((1 - self.real) / (1 - self.prediction))