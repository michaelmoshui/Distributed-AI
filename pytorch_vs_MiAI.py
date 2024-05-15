import torch
import torch.nn as nn
import time
import MiAI as ma
import numpy as np

class TorchModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.linear1 = nn.Linear(784, 1024)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 2048)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(2048, 2048)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(2048, 2048)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(2048, 10)
        self.relu5 = nn.ReLU()

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.softmax(x)
        return x
    
class MAModel(ma.Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            ma.Dense(784, 1024),
            ma.ReLU(),
            ma.Dense(1024, 2048),
            ma.ReLU(),
            ma.Dense(2048, 2048),
            ma.ReLU(),
            ma.Dense(2048, 2048),
            ma.ReLU(),
            ma.Dense(2048, 10),
            ma.ReLU(),
            ma.Softmax()
        ]

'''
Somewhat sad results, pytorch is at least 50 times faster for the same computation lol
'''
X = torch.randn(1000, 784)

model = TorchModel()

s = time.time()
res = model(X)
e = time.time()

print("PyTorch Time:", e - s)

X = np.random.randn(1000, 784)

model = MAModel()

s = time.time()
res = model(X)
e = time.time()

print("MiAI Time:", e - s)