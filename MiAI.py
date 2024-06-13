import numpy as np
from multiprocessing import Pool, shared_memory
from functools import partial
import time
from TimingAnalysis import TimeAnalysis as TA
from mpi4py import MPI

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

        # multicore process
        self.multi = False
        self.num_processes = 0
        self.num_minibatch = 4
    
        # cluster process
        self.cluster = True
        self.num_devices = 1
        self.X = None
        self.y = None

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
            if layer.type == "Dense" or layer.type == "conv2d":
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
            multi_train_ta = TA()

            multi_train_ta.begin("Training")

            self.y = y

            # partition inputs into different segments
            multi_train_ta.begin("Partition")
            
            N = X.shape[0]
            partition_size = N // self.num_minibatch
            partitions = [(i, min(N, i + partition_size)) for i in range(0, N, partition_size)]
            
            multi_train_ta.end("Partition")

            # Initialize shared memory
            multi_train_ta.begin("Share Memory")

            shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
            X_shared = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
            np.copyto(X_shared, X)
            
            multi_train_ta.end("Share Memory")
            
            # establish worker processes and send data to the different workers
            multi_train_ta.begin("Worker Process")
                     
            with Pool(processes=self.num_processes) as pool:
                multi_train = partial(self.multi_train,
                                      Loss,
                                      shm_name=shm.name,
                                      shape=X.shape,
                                      dtype=X.dtype)
                all_results = pool.map(multi_train, partitions)
                                    
                # clean up shm
                shm.close()
                shm.unlink()
            
            multi_train_ta.end("Worker Process")

            # get the mean of gradients and new weights as necessary
            multi_train_ta.begin("Collection")

            for (i, layer) in enumerate(self.layers):
                if layer.type == "Dense" or layer.type == "conv2d":
                    dW = np.mean([process[0][i]['dW'] for process in all_results], axis=0)
                    Optimizer.optimize(dW, 'W', layer)
                    
                    if layer.bias:
                        dB = np.mean([process[0][i]['dB'] for process in all_results], axis=0)
                        Optimizer.optimize(dB, 'B', layer)
            
            multi_train_ta.end("Collection")

            multi_train_ta.end("Training")

            multi_train_ta.print_analysis()

        else:
            self.single_train(X, y, Loss, Optimizer)

    def cluster_train(self, loss_fn, Optimizer, batch_size=64):
        # generate batches
        mini_batchsize = batch_size // self.num_devices
        random_indices = np.random.randint(0, len(self.X), mini_batchsize)

        batch_x_shape = (mini_batchsize,) + self.X.shape[1:]
        batch_x = np.empty(batch_x_shape, dtype=np.float32)

        batch_y_shape = (mini_batchsize,) + self.y.shape[1:]
        batch_y = np.empty(batch_y_shape, dtype=self.y.dtype)

        for i, index in enumerate(random_indices):
            batch_x[i] = self.X[index]
            batch_y[i] = self.y[index]

        # start forward pass
        for layer in self.layers:
            if layer.type == "Dense" or layer.type == "conv2d": # need input for dense layer only
                layer.input = batch_x
            batch_x = layer(batch_x)
            layer.output = batch_x
        
        # calculate loss
        loss_fn(batch_y, batch_x)

        # start backward pass
        delta = loss_fn.backward()

        for layer in reversed(self.layers):
            if layer.type == "Dense" or layer.type == "conv2d":
                # calculate delta and derivatives through backward pass
                delta, dW, dB = layer.backward(delta)

                # update the weights (at the same time update gradient based on optimizer type)
                Optimizer.optimize(dW, 'W', layer)
                if layer.bias:
                    Optimizer.optimize(dB, 'B', layer)
            else:
                delta = layer.backward(delta)

        # ring all reduce algorithm to communicate gradients with each other
        num_layers = len(self.layers) // self.num_devices

        send_partition = self.rank
        receive_partition = (self.rank - 1 + self.num_devices) % self.num_devices
        
        # can probably optimize with sending numpy arrays later
        # scatter reduce
        for _ in range(self.num_devices - 1):
            
            # send functions
            send_indices = [send_partition * num_layers,
                            min(send_partition * num_layers + num_layers, len(self.layers))]
            
            sent_data = self.comm.isend(self.layers[send_indices[0]:send_indices[1]],
                                   dest=(send_partition + 1) % self.num_devices)
            
            # receive functions
            receive_indices = [receive_partition * num_layers,
                               min(receive_partition * num_layers + num_layers, len(self.layers))]
            
            received_data = self.comm.irecv(source=receive_partition)
            
            received_data = received_data.wait()

            for i, layer in enumerate(self.layers[receive_indices[0]:receive_indices[1]]):
                if layer.type == "Dense" or layer.type == "conv2d":

                    layer.params['W'] += received_data[i].params['W']
                    layer.params['B'] += received_data[i].params['B']

                    layer.grads['W'] += received_data[i].grads['W']
                    layer.grads['B'] += received_data[i].grads['B']

            sent_data.wait() # wait on sent_data in case it's not finished yet

            # update sending and receiving partition
            send_partition = receive_partition
            receive_partition = (send_partition - 1 + self.num_devices) % self.num_devices

        # all gather
        for _ in range(self.num_devices - 1):

            # send functions
            send_indices = [send_partition * num_layers,
                            min(send_partition * num_layers + num_layers, len(self.layers))]
            
            sent_data = self.comm.isend(self.layers[send_indices[0]:send_indices[1]],
                                   dest=(send_partition + 1) % self.num_devices)
            
            # receive functions
            receive_indices = [receive_partition * num_layers,
                               min(receive_partition * num_layers + num_layers, len(self.layers))]
            
            received_data = self.comm.irecv(source=receive_partition)
            
            received_data = received_data.wait()

            for i, layer in enumerate(self.layers[receive_indices[0]:receive_indices[1]]):
                if layer.type == "Dense" or layer.type == "conv2d":

                    layer.params['W'] = received_data[i].params['W']
                    layer.params['B'] = received_data[i].params['B']

                    layer.grads['W'] = received_data[i].grads['W']
                    layer.grads['B'] = received_data[i].grads['B']

            sent_data.wait() # wait on sent_data in case it's not finished yet

            # update sending and receiving partition
            send_partition = receive_partition
            receive_partition = (send_partition - 1 + self.num_devices) % self.num_devices

        # take the mean
        for layer in self.layers:
            if layer.type == "Dense" or layer.type == "conv2d":
                layer.params['W'] /= self.num_devices
                layer.params['B'] /= self.num_devices
                layer.grads['W'] /= self.num_devices
                layer.grads['B'] /= self.num_devices




    # single processor training; forward and backward pass
    def single_train(self, X, y, loss_fn, Optimizer):
        
        start_train = time.time()

        for layer in self.layers:
            if layer.type == "Dense" or layer.type == "conv2d": # need input for dense layer only
                layer.input = X
            X = layer(X)
            layer.output = X

        loss_fn(y, X)

        delta = loss_fn.backward()

        for layer in reversed(self.layers):
            # backward prop fro Dense Layer
            if layer.type == "Dense" or layer.type == "conv2d":
                # calculate delta and derivatives through backward pass
                delta, dW, dB = layer.backward(delta)

                # update the weights (at the same time update gradient based on optimizer type)
                Optimizer.optimize(dW, 'W', layer)
                if layer.bias:
                    Optimizer.optimize(dB, 'B', layer)
            else:
                delta = layer.backward(delta)

        end_train = time.time()

        print("training duration:", end_train - start_train)
        
    # multiprocessor training; forward and backward pass on a minibatch
    # using shared memory is marginally faster than creating separate copies of data
    # 11.3s vs 12.3s over 10 training epochs
    def multi_train(self, loss_fn, partition_indices, shm_name, shape, dtype):
        start_idx, end_idx = partition_indices

        # Access the shared memory
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        X_shared = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        # Extract the partition from the shared memory
        x = X_shared[start_idx:end_idx]
        y = self.y[start_idx:end_idx]  # Assuming `self.y` is available to all processes

        start_train = time.time()
        # forward pass
        for layer in self.layers:
            if layer.type == "Dense" or layer.type == "conv2d": # need input for dense layer only
                layer.input = x
            x = layer(x)
            layer.output = x

        # backward pass
        weight_updates = [None] * len(self.layers)

        loss_fn(y, x)
        delta = loss_fn.backward()
        
        for i, layer in enumerate(reversed(self.layers)):
            if layer.type == "Dense" or layer.type == "conv2d":
                # calculate delta and derivatives through backward pass
                delta, dW, dB = layer.backward(delta)
                
                # update the weights (at the same time update gradient based on optimizer type)
                weight_updates[len(weight_updates) - 1 - i] = {'dW': dW, 'dB': dB}
            else:
                delta = layer.backward(delta)
        
        stop_train = time.time()

        existing_shm.close()

        return weight_updates, start_train, stop_train - start_train
    
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

    def cluster_init(self, X, y, num_devices=4):
        self.cluster = True
        self.X = X
        self.y = y
        self.num_devices = num_devices

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

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
                print("Trainable parameters:", str(layer.params['W'].size + layer.params['B'].size))
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
        
        self.params = {'W': np.random.randn(self.output_dim, self.input_dim).astype(np.float32),
                       'B': np.random.randn(self.output_dim).astype(np.float32) if self.bias else None}

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

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), padding_value=0, use_bias=True):
        super().__init__()
        self.type = "conv2d"
        self.stride = stride
        self.padding = padding
        self.bias = use_bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        
        # Xavier initialization of weights
        fan_in = self.in_channels * kernel_size[0] * kernel_size[1]
        fan_out = self.out_channels * kernel_size[0] * kernel_size[1]
        limit = np.sqrt(6 / (fan_in + fan_out))

        self.params = {'W': np.random.uniform(-limit, limit, (self.out_channels, self.in_channels, kernel_size[0], kernel_size[1])).astype(np.float32),
                       'B': np.ones((self.out_channels)).astype(np.float32) if self.bias else None}
        
        self.grads = {'W': np.zeros_like(self.params['W']),
                      'B': np.zeros_like(self.params['B']) if self.bias else None}

        self.padding_value = padding_value

        self.input = None
        self.output = None

    def pad_input(self, image):
        if self.padding != (0, 0):
            return np.pad(image, ((0,0), (0,0), self.padding, self.padding), mode='constant', constant_values=self.padding_value)
        else:
            return image
        
    def shape_init(self, image):
        new_height = (image.shape[2] - self.params['W'].shape[2]) // self.stride[0] + 1
        new_width = (image.shape[3] - self.params['W'].shape[3]) // self.stride[1] + 1

        res = np.zeros((image.shape[0], self.out_channels, new_height, new_width))
        
        return res, new_height, new_width
    
    def convolve(self, image):
        
        output, new_height, new_width = self.shape_init(image)
        filter_height, filter_width = self.params['W'].shape[2], self.params['W'].shape[3]
        
        for i in range(0, new_height * self.stride[0], self.stride[0]):
            for j in range(0, new_width * self.stride[1], self.stride[1]):
                region = image[:, :, i:i+filter_height, j:j+filter_width]
                if self.bias:    
                    output[:, :, i // self.stride[0], j // self.stride[1]] = np.tensordot(region, self.params['W'], axes=([1, 2, 3], [1, 2, 3])) + self.params['B']
                else:
                    output[:, :, i // self.stride[0], j // self.stride[1]] = np.tensordot(region, self.params['W'], axes=([1, 2, 3], [1, 2, 3]))
        return output
    
    def __call__(self, X):
        self.input = self.pad_input(X)

        self.output = self.convolve(self.input)
        
        return self.output
    
    def dilate_matrix(self, matrix, y_stride, x_stride):
        
        output_shape = (matrix.shape[0],
                        matrix.shape[1],
                        matrix.shape[2] * y_stride - (y_stride - 1),
                        matrix.shape[3] * x_stride - (x_stride - 1))
        
        dilated = np.zeros(output_shape, dtype=matrix.dtype)

        dilated[:, :, ::y_stride, ::x_stride] = matrix
        return dilated, dilated.shape[2], dilated.shape[3]

    def pad_and_dilate(self, matrix):
        dilated_shape = (matrix.shape[0],
                         matrix.shape[1],
                         matrix.shape[2] * self.stride[0] - (self.stride[0] - 1),
                         matrix.shape[3] * self.stride[0] - (self.stride[0] - 1))
        
        dilated = np.zeros(dilated_shape, dtype=matrix.dtype)

        dilated[:, :, ::self.stride[0], ::self.stride[1]] = matrix

        padding_height, padding_width = self.params['W'].shape[2] - 1, self.params['W'].shape[3] - 1
        padded_shape = (dilated_shape[0],
                        dilated_shape[1],
                        dilated_shape[2] + 2 * padding_height,
                        dilated_shape[3] + 2 * padding_width)
        
        padded = np.zeros(padded_shape, dtype=dilated.dtype)

        padded[:, :, padding_height:padding_height+dilated_shape[2], padding_width:padding_width+dilated_shape[3]] = dilated

        return padded

    def backward(self, delta):
        dout = np.zeros_like(self.input)

        dW = np.zeros_like(self.grads['W'])
        dB = np.sum(delta, axis=(0, 2, 3))

        dilated_gradient, dilated_height, dilated_width = self.dilate_matrix(delta, self.stride[0], self.stride[1])
        
        # comparing the two implementations....
        # tensordot with a for loop sum is actually faster than using pure numpy with einsum

        fs = time.time()
        for n in range(self.input.shape[0]):
            for i in range(dW.shape[2]):
                for j in range(dW.shape[3]):
                    region = self.input[n, :, i:i+dilated_height, j:j+dilated_width]
                    conv = np.tensordot(dilated_gradient[n], region, axes=([-2,-1], [-2,-1]))
                    dW[:, :, i, j] += conv
        fe = time.time()
        # temp = np.zeros_like(dW)

        # es = time.time()
        # for i in range(self.params['W'].shape[2]):
        #     for j in range(self.params['W'].shape[3]):
        #         region = self.input[:, :, i:i+dilated_height, j:j+dilated_width]
        #         temp[:, :, i, j] = np.einsum('nfij,ncij->nfc', dilated_gradient, region).sum(axis=0)
        # ee = time.time()

        padded_delta = self.pad_and_dilate(delta)

        flipped_transposed_params = np.transpose(np.rot90(self.params['W'], 2, axes=(-2,-1)), axes=(1, 0, 2, 3))
        
        for n in range(delta.shape[0]):
            for i in range(delta.shape[2] * self.stride[0]):
                for j in range(delta.shape[3] * self.stride[1]):
                    region = padded_delta[n, :, i:i+self.params['W'].shape[2], j:j+self.params['W'].shape[3]]
                    dout[n, :, i, j] = np.tensordot(flipped_transposed_params, region, axes=([-3, -2, -1], [-3, -2, -1]))
                    
        return dout, dW, dB

    def get_name(self):
        return "Convolution 2D Layer"


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

class Flatten(Layer):
    def __init__(self):
        super().__init__()
    
    def __call__(self, X):
        self.input = X
        self.output = np.reshape(X, (X.shape[0], -1))
        return self.output
    
    def backward(self, delta):
        return np.reshape(delta, self.input.shape)

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