import MiAI as ma
import numpy as np
import time
from multiprocessing import Pool

class ExampleMA(ma.Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            ma.Dense(784, 1024),
            ma.ReLU(),
            ma.Dense(1024, 1024),
            ma.ReLU(),
            ma.Dense(1024, 512),
            ma.ReLU(),
            ma.Dense(512, 10),
            ma.Softmax()
        ]

if __name__ == "__main__":
    X_train = np.ones((10000, 784))
    y_train = np.random.randn(10000, 10)

    # Multi processor
    model = ExampleMA()
    loss = ma.CCE()
    optim = ma.RMSProp(lr=0.05)
    model.multiprocess(num_minibatch=12)

    s = time.time()
    model.train(X_train, y_train, loss, optim)
    e = time.time()


    e = time.time()

    print("Duration of multi processor:", e - s)
