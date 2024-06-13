import MiAI as ma
import numpy as np
from TimingAnalysis import TimeAnalysis as TA

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

time = TA()
# time.begin("Init 1")
# x = np.zeros((500, 500, 500), dtype=np.float16)
# time.end("Init 1")

# time.begin("Init 2")
# x = np.zeros((500, 500, 500))
# time.end("Init 2")

time.begin("Init 3")
x = np.random.randn(100, 500, 500)
time.end("Init 3")

time.begin("Init 4")
y = np.random.randn(100, 500, 500).astype(np.float32)
time.end("Init 4")

time.begin("Init 5")
z = np.random.randn(100, 500, 500).astype(np.float64)
time.end("Init 5")

time.begin("calc 1")
for i in range(10):
    x = x + x
time.end("calc 1")

time.begin("calc 2")
for i in range(10):
    y = y + y
time.end("calc 2")

time.begin("calc 3")
for i in range(10):
    z = z + z
time.end("calc 3")

time.print_analysis()