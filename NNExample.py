# %%
'''Importing dependencies'''
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_moons, make_circles
import numpy as np
import MiAI as ma

# %%
'''Generate data and put it to shape compatible with training'''

# Define and register a custom colormap
colormap = mcolors.ListedColormap(['skyblue', 'salmon'])
matplotlib.colormaps.register(name='custom_cmap', cmap=colormap)

# Generate Data function
def generate_data(samples, shape_type='circles', noise=0.05):
    if shape_type == 'moons':
        X, Y = make_moons(n_samples=samples, noise=noise)
    elif shape_type == 'circles':
        X, Y = make_circles(n_samples=samples, noise=noise)
    else:
        raise ValueError(f"The introduced shape {shape_type} is not valid. Please use 'moons' or 'circles'")
    
    data = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
    return data

# Plot data function
def plot_generated_data(data):
    ax = data.plot.scatter(x='x', y='y', figsize=(16, 12), c=data['label'], cmap='custom_cmap', grid=True)
    return ax

# Generate data and plot
data = generate_data(samples=5000, shape_type='circles', noise=0.04)
plot_generated_data(data)

# Reshape data
X = data[['x', 'y']].values
Y = np.reshape(data['label'].T.values, (-1, 1))

# %%
'''Preparing for model training'''

# Define a model
class ExampleMA(ma.Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            ma.Dense(2, 5),
            ma.ReLU(),
            ma.Dense(5, 16),
            ma.ReLU(),
            ma.Dense(16, 1),
            ma.Sigmoid()
        ]

# Initialize model object
model = ExampleMA()

# Define binary crossentropy loss
BCELoss = ma.BCE()

# define training loop
def train(X, Y, model, loss_fn, epochs=1000):

    for i in range(epochs):
        
        output = model(X)
        loss = loss_fn(Y, output)

        model.backprop(loss_fn, 0.01)

        if i % 50:
            print(f'Epoch {i} Loss: {loss}')

# %%
'''Train the model'''

train(X, Y, model, BCELoss, 10000)

# %%

'''Visualize the result of training'''
# make a prediction and plot the result
def plot_decision_boundary(X, y, model, steps=1000, cmap='Paired'):
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model(np.c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels = model(X)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)

    return fig, ax

plot_decision_boundary(X, Y, model, cmap='RdBu')
# %%
