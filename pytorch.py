import torch.nn as nn
import torch

conv = nn.Conv2d(2, 4, (3, 3), 2)

x = torch.tile(torch.arange(1.0, 11.0).reshape(10, 1, 1, 1), (1, 2, 8, 8))
print(x.shape)
print(conv(x).shape)