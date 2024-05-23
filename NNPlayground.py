import MiAI as ma
import numpy as np
import time

class ExampleMA(ma.Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            ma.Conv2D(1, 32, (3, 3), (2,2)),
            ma.Conv2D(32, 64, (5, 5)),
            ma.Conv2D(64, 128, (7, 7)),
            ma.Flatten(),
            ma.Dense(1152, 1024),
            ma.ReLU(),
            ma.Dense(1024, 10),
            ma.Softmax()
        ]

convnet = ExampleMA()



x = np.random.randn(10, 1, 28, 28)
res = convnet(x)