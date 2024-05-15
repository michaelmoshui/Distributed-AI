import numpy as np
from multiprocessing import Pool
import gc
from functools import partial

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
        ~ self.x: the input (and then transformed) of the function
        ~ self.layers: a list of neural network layers
        ~ self.multi: train on multiple processes?
        ~ self.processes: number of processes for training
        '''
        self.x = None
        self.layers = []
        self.multi = False
        self.num_processes = 0
        self.num_minibatch = 4
    
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
    def backprop(self, Loss, Optimizer):
        '''
        Purpose:
        ~ update weights and biases through backpropagation

        Input:
        ~ Optimizer
        ~ Loss: loss function object 

        Output:
        None
        '''
        delta = Loss.backward()

        for layer in reversed(self.layers):
            # backward prop fro Dense Layer
            if layer.type == "Dense":
                # calculate delta and derivatives through backward pass
                delta, dW, dB = layer.backward(delta)
                
                # update the weights (at the same time update gradient based on optimizer type)
                Optimizer.optimize(dW, 'W', layer)
                
                if layer.bias:
                    Optimizer.optimize(dB, 'B', layer)

            else:
                delta = layer.backward(delta)

    # training function
    def train(self, X, y, Loss, Optimizer):
        '''
        ~ Both single-core and multi-core training
        ~ Data parallelism
        ~ Training loop does not update the input and output of each layer.
        ~ need to do that in the __call__ function
        '''
        if self.multi:
            # partition inputs into different segments
            N = X.shape[0]
            partition_size = N // self.num_minibatch
            partitions = [(X[i:min(N, i + partition_size)], y[i:min(N, i + partition_size)]) for i in range(0, N, partition_size)]

            # establish worker processes and send data to the different workers
            with Pool(processes=self.num_processes) as pool:
                multi_train = partial(self.multi_train, Loss, Optimizer)
                all_results = pool.map(multi_train, partitions) 

            # get the mean of gradients and new weights as necessary
            for (i, layer) in enumerate(self.layers):
                if layer.type == "Dense":

                    layer.params['W'] = np.mean([process[i].params['W'] for process in all_results], axis=0)
                    layer.params['B'] = np.mean([process[i].params['B'] for process in all_results], axis=0)
                    layer.grads['W'] = np.mean([process[i].grads['W'] for process in all_results], axis=0)
                    layer.grads['B'] = np.mean([process[i].grads['B'] for process in all_results], axis=0)
                    
        else:
            self.single_train(X, y, Loss, Optimizer)

    # single processor training; forward and backward pass
    def single_train(self, X, y, loss_fn, Optimizer):
        
        for layer in self.layers:
            if layer.type == "Dense": # need input for dense layer only
                layer.input = X
            X = layer(X)
            layer.output = X

        loss_fn(y, X)

        delta = loss_fn.backward()

        for layer in reversed(self.layers):
            # backward prop fro Dense Layer
            if layer.type == "Dense":
                # calculate delta and derivatives through backward pass
                delta, dW, dB = layer.backward(delta)
                
                # update the weights (at the same time update gradient based on optimizer type)
                Optimizer.optimize(dW, 'W', layer)
                if layer.bias:
                    Optimizer.optimize(dB, 'B', layer)
            else:
                delta = layer.backward(delta)
        
    # multiprocessor training; forward and backward pass on a minibatch
    def multi_train(self, loss_fn, Optimizer, partition):
        x, y = partition
        # forward pass
        for layer in self.layers:
            if layer.type == "Dense": # need input for dense layer only
                layer.input = x
            x = layer(x)
            layer.output = x

        # backward pass
        loss_fn(y, x)

        delta = loss_fn.backward()
        
        for layer in reversed(self.layers):
            if layer.type == "Dense":
                # calculate delta and derivatives through backward pass
                delta, dW, dB = layer.backward(delta)
                
                # update the weights (at the same time update gradient based on optimizer type)
                Optimizer.optimize(dW, 'W', layer)
                if layer.bias:
                    Optimizer.optimize(dB, 'B', layer)
            else:
                delta = layer.backward(delta)

        return self.layers
    
    # clear gradients
    def clear_grad(self):
        for layer in self.layers:
            if layer.type == "Dense":
                layer.grads = {'W': np.zeros((layer.output_dim, layer.input_dim)),
                               'B': np.zeros(layer.output_dim) if layer.bias else None}
    
    # clear paramenters
    def clear_params(self):
        for layer in self.layers:
            if layer.type == "Dense":
                layer.params = {'W': np.random.randn(layer.output_dim, layer.input_dim),
                                'B': np.random.randn(layer.output_dim) if layer.params['B'] else None}

    # initiate training with multiple cores
    def multiprocess(self, num_processes=None, num_minibatch=4):
        self.multi = True
        self.num_processes = num_processes
        self.num_minibatch = num_minibatch

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
                print("Weights:", "(" + str(layer.params['W'].shape[0]) + ", " + str(layer.params['B'].shape[1]) + ")")
            print("********************")

#######################
# Neural Network Layers
#######################
class Layer():
    def __init__(self):
        self.type = None
        self.otuput = None

    def __call__(self, X):
        pass

    def backward(self, X):
        pass

    def get_name(self):
        pass

class Dense(Layer):
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
        ~ self.bias: boolean; include bias term or not
        ~ self.input_dim: input dimension
        ~ self.output_dim: output dimension
        ~ self.input: a (N, I) input; where N is batch size and I is input dimension
        ~ self.output: a (N, O) output; where N is batch size and I is output dimension
        ~ self.params['W']: a (O, I) weight matrix; where O is output dimension and I is input dimension
        ~ self.params['B']: an (O,) vector; where O is output dimension
        '''
        super().__init__()
        self.type = "Dense"
        
        self.bias = bias

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = None
        
        self.params = {'W': np.random.randn(self.output_dim, self.input_dim),
                       'B': np.random.randn(self.output_dim) if self.bias else None}

        self.grads = {'W': None,
                      'B': None}
                
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
        matrix_multiplication = np.einsum("ij, nj -> ni", self.params['W'], X)
        
        # depending on wanting bias or not, add bias to every output vector
        if not self.bias:
            self.output = matrix_multiplication
        else:
            self.output = matrix_multiplication + self.params['B'][np.newaxis, :]

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
        ~ has shape (O, I); same as self.params['W']

        ~ dB: gradient vector for bias term in this layer
        ~ has shape (O,); same as self.params['B']
        '''
        
        # compute dW with delta X input averaged over all N matrix multiplications
        dW = np.matmul(delta.T, self.input) / delta.shape[0]
        dB = np.sum(delta, axis=0) / delta.shape[0] if self.bias else None
        
        delta = np.dot(delta, self.params['W'])
        
        return delta, dW, dB

    def get_name(self):
        return "Dense Layer"

class ReLU(Layer):
    def __init__(self):
        '''
        Purpose:
        ~ Initialize a ReLU layers

        Attributes:
        ~ self.type: ReLU
        ~ self.output: a (N, D) matrix, where D is the dimension of the input and output
        '''
        super().__init__()
        self.type = "ReLU"

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

class Softmax(Layer):
    def __init__(self) -> None:
        '''
        Purpose:
        ~ Initialize the softmax activation

        Attributes:
        ~ self.type: Softmax
        ~ self.output: a (N, D) matrix, where D is the dimension of the input and output
        '''
        super().__init__()
        self.type = "Softmax"

    def __call__(self, X):
        '''
        Purpose:
        ~ Calls the softmax activation function

        Input and Output:
        ~ (N, D) matrix that gets transformed according to the softmax activation function definition
        '''

        X = minmax_scale(X, -1, 1)

        self.output = np.divide(np.exp(X), np.sum(np.exp(X), axis=1)[:, np.newaxis])
        return self.output
    
    def backward(self, delta):
        return delta # this assumes that the gradient of CCE and Softmax is combined together

    def get_name(self):
        return "Softmax Activation Function"
    
class Sigmoid(Layer):
    def __init__(self) -> None:
        '''
        Purpose:
        ~ Initiate the sigmoid activation function

        Attributes:
        ~ self.type: Sigmoid
        ~ self.output: (N, D) matrix, where the columns are probability values between 0 and 1
        '''
        super().__init__()
        self.type = "Sigmoid"

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

# backward pass not finished!
class BatchNorm(Layer):
    def __init__(self, num_feature, eps=1e-5, momentum=0.1):
        '''
        Purpose:
        ~ Normalize layer activations across the batch dimension

        Attributes:
        ~ self.type: BatchNorm
        ~ self.output: (N, D) matrix, where D is the dimension of the input and output matrix
        ~ self.momentum (float): determines the update scheme of the running statistics (mean and variance)
        ~ self.num_feature (float): number of features/channels of the input and output
        ~ self.eps (float): numerical correction for normalization calculation
        ~ self.training (boolean): determines training vs. inference mode 
        ~ self.gamma: (D,) learnable parameter for affine transformation
        ~ self.beta: (D,) learnable parameter for affine transformation
        '''
        super().__init__()
        self.type = "BatchNorm"
        self.momentum = momentum
        self.num_feature = num_feature
        self.eps = eps
        self.training = True
        self.num_feature = num_feature
        self.batch_dim = None
        
        # Trainable parameters; Gamma and Beta
        self.params = {'G': np.random.randn(1, self.num_feature),
                       'B': np.random.randn(1, self.num_feature)}
        
        self.grads = {'G': None,
                      'B': None}

        # Moving statistics
        self.moving_mean = np.zeros((1, self.num_feature))
        self.moving_variance = np.ones((1, self.num_feature))

        self.cache = None

    def __call__(self, X):
        '''
        Purpose:
        ~ forward pass of the Dense layer

        Input:
        ~ X: input sample batch of shape (batch_size, num_feature, spatial)
        ~ spatial dimension is optional, and might have multiple dimensions
        ~ in dense layers, there is no spatial dimension
        ~ in convolution layers, spatial dimension consists of width and height

        Output:
        ~ self.output
        ~ Same shape as input: (batch_size, num_feature, spatial)
        '''

        # Compute mean and variance of the mini-batch for each input feature
        # Assume the input feature dimension is axis=1
        self.batch_dim = tuple(j for j in range(X.ndim) if j != 1) 
        batch_mean = np.mean(X, axis=self.batch_dim, keepdims=True) # (1, num_feature)
        batch_variance = np.var(X, axis=self.batch_dim, keepdims=True) # (1, num_feature)

        # Update moving mean and variance
        self.moving_mean = (1 - self.momentum)*self.moving_mean + self.momentum*batch_mean
        self.moving_variance = (1 - self.momentum)*self.moving_variance + self.momentum*batch_variance
        
        # Forward pass for training vs. inference
        if self.training:
            X_norm = (X - batch_mean)/np.sqrt(batch_variance + self.eps)
            self.cache = (X, X_norm, batch_mean, batch_variance)
        else:
            X_norm = (X - self.moving_mean)/np.sqrt(self.moving_variance + self.eps)
        
        return self.params['G'] * X_norm + self.params['B']

    def backward(self, delta):
        X, X_norm, batch_mean, batch_variance = self.cache

        dG = np.sum(delta * X_norm, axis=self.batch_dim, keepdims=True)
        dB = np.sum(delta, axis=self.batch_dim, keepdims=True)

        dX_norm = delta * self.params['G']
        
    def get_name(self):
        return "Batch Normalization"

################
# Lost Functions
################
class Loss():
    def __init__(self):
        self.real = None
        self.prediction = None

class BCE(Loss):
    def __init__(self):
        '''
        Purpose:
        ~ Initialize a binary-crossentropy function object

        Attributes:
        ~ self.real: (N, D) array of the real labels
        ~ self.prediction: (N, D) array of predictions made by the neural network
        '''
        super().__init__()
        
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

    def calculate_loss(self):
        return -np.mean(np.add(np.multiply(self.real, np.log(self.prediction)), np.multiply((1 - self.real), np.log(1 - self.prediction))))
    
    def backward(self):
        '''
        Purpose:
        ~ compute the gradient of the loss function

        Output:
        ~ (N, D) matrix containing N samples of a N-dimensional label
        '''
        return -(self.real / self.prediction) + ((1 - self.real) / (1 - self.prediction))

class CCE(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, real, prediction):
        self.real = real
        self.prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
    
    def calculate_loss(self):
        return np.mean(-np.sum(self.real * np.log(self.prediction), axis=1))
    
    def backward(self):
        # Assumes softmax layer implemented
        return self.prediction - self.real

############
# Optimizers
############
class Optimizer():
    def __init__(self, lr = 0.01):
        self.lr = lr

# Gradient Descent
class GD(Optimizer):
    def __init__(self, lr = 0.01):
        super().__init__(lr = lr)

    def optimize(self, dW, weight_type, layer):
        layer.grads[weight_type] = dW # update grad
        layer.params[weight_type] -= self.lr * dW # gradient descent
    
    def get_name():
        return "Gradient Descent"
    
class RMSProp(Optimizer):
    def __init__(self, lr = 0.01, beta = 0.9):
        super().__init__(lr = lr)
        self.beta = beta

    def optimize(self, dW, weight_type, layer):
        # update grad
        if layer.grads[weight_type] is not None:
            layer.grads[weight_type] = np.sqrt(self.beta * layer.grads[weight_type] + (1 - self.beta) * dW * dW) # update grad
        else:
            layer.grads[weight_type] = np.sqrt(dW * dW)
        # update weights
        layer.params[weight_type] -= self.lr * dW / np.clip(layer.grads[weight_type], a_min=1e-15, a_max=None)

    def get_name():
        return "RMSProp Optimizer"

class Adam(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def optimize(self, dW, weight_type, layer):
        pass
    
# Miscellaneous Functions
def minmax_scale(data, min_val=0, max_val=1):
    # Calculate the min and max along axis 1, keep the dimensions for broadcasting
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)

    # Avoid division by zero in case of constant rows
    scaled_data = np.where(max_vals - min_vals == 0, min_val,
                           (max_val - min_val) * ((data - min_vals) / (max_vals - min_vals)) + min_val)

    return scaled_data