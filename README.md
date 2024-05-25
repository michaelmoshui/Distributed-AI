# Distributed AI

The purpose of this project is to explore distributed neural network training on multiple computers and/or laptop cores. Training performance is not meant to surpass GPUs in any sense; the main intention is to explore computing clusters and learning about the inner working of deep learning with self-implemented neural network layers.

## How It Works

MiAI is a very basic and continually expanding deep learning framework created using Python (primarily usin NumPy APIs). It consists of one Model class, various Layer classes, Optimizer classes, Loss function classes, and expanding set of miscellaneous functions. Look at NNExample.py to see how to instantiate and train a deep learning model using MiAI.

In addition to deep learning framework development, this project also focuses on exploring different parallel computing methods. The latest parallel computing method in model training is doing data parallelism using multiprocessing with shared memory. Unfortunately the initiation overhead is a great bottleneck, and parallel computing only surpasses efficiency in the case of a 256-sample batch size for convolutional neural networks. Still exploring different methods of improving training performance.

There is an additional Timing Analysis module that helps organize duration of function runs. Its primary goal is to reduce random print statements related to timing analysis throughout debugging.

## Dependencies (as of May 25th, 2024):
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Completion (as of May 25th, 2024):
* Basic deep learning framework (Dense, Conv2D, ReLU, Sigmoid, Binary Crossentropy, Batch Norm, Categorical Crossentropy)
* Sample implementation
* RMSProp Optimizer
* Multiprocessing training

## To be completed (very general to be expanded):
* Computing cluster setup
* Distriuted learning Pipeline
* More advanced deep learning components (optimizers, normalization layers, etc...)

## How you can contribute
* adding more deep learning layers
* Suggest ways to improve parallel computing (traditional laptops and CPU only!)
