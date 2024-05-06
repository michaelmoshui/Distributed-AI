import MiAI as ma
import numpy as np
import time

class ExampleMA(ma.Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            ma.Dense(784, 1024),
            ma.ReLU(),
            ma.Dense(1024, 1024),
            ma.ReLU(),
            ma.Dense(1024, 1024),
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

    model = ExampleMA()

    # Single processor
    s = time.time()
    pred_1 = model(X_train)
    e = time.time()

    print("Duration of single processor:", e - s)

    # Multi processor
    model = ExampleMA()
    
    model.multiprocess(num_processes=12, num_minibatch=12)

    s = time.time()
    pred_2 = model(X_train)
    e = time.time()

    print("Duration of multi processor:", e - s)