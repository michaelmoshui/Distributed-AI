import numpy as np
import MiAI as ma

class ExampleMA(ma.Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            ma.Dense(10, 15),
            ma.ReLU(),
            ma.Dense(15, 12),
            ma.ReLU(),
            ma.Dense(12, 1),
            ma.Sigmoid()
        ]

model = ExampleMA()

input = np.array([[0,.1,.2,.3,.4,.5,.6,.7,.8,.9], [0,.1,.9,.3,.2,.4,.6,.7,.8,.9]])
label = np.reshape(np.array([0, 1]), (-1, 1))

BCELoss = ma.BCE()

for i in range(100):    
    output = model(input)

    loss = BCELoss(label, output)

    model.backprop(BCELoss, 0.01)
    
    print(f'Epoch {i} Loss: {loss}')

# make a prediction
print(model(input))