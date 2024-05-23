# %%
'''Importing dependencies'''
import matplotlib.pyplot as plt
import numpy as np
import MiAI as ma
from sklearn.datasets import fetch_openml
import time

# %%
'''Preparing for model training'''
# Define a model
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
            ma.Dense(1024, 1024),
            ma.ReLU(),
            ma.Dense(1024, 1024),
            ma.ReLU(),
            ma.Dense(1024, 10),
            ma.Softmax()
        ]

# accuracy
def calculate_accuracy(real, prediction):
    total = len(real)

    correct = np.sum([1 if np.argmax(real[i]) == np.argmax(prediction[i]) else 0 for i in range(len(real))])
    
    return correct / total

# define training loop
def train(X, y, model, loss_fn, optimizer, batch_size, epochs=100):

    training_losses = []
    training_accuracies = []

    model.clear_grad()
    model.multiprocess(num_minibatch=4)

    for i in range(epochs):        
        random_indices = np.random.randint(0, len(X), batch_size)
  
        batch_x = []
        batch_y = []

        for j in random_indices:
            batch_x.append(X[j])
            batch_y.append(y[j])

        batch_x = np.array(batch_x) / 255.0
        batch_y = np.eye(10)[np.array(batch_y).flatten().astype(int)]

        start = time.time()
        model.train(batch_x, batch_y, loss_fn, optimizer)
        end = time.time()
        output = model(batch_x)

        loss_fn(batch_y, output)

        training_loss = loss_fn.calculate_loss()

        training_accuracy = calculate_accuracy(batch_y, output)

        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)

        print(f'Epoch {i} Training Loss: {training_loss} Training Duration: {end - start}')

    return training_losses, training_accuracies

# %%
if __name__ == "__main__":
    '''Data processing'''
    # fetch data
    mnist = fetch_openml('mnist_784', version=1)

    # Data and targets
    X, y = np.array(mnist['data']).reshape(-1, 1, 28, 28), np.array(mnist['target']).reshape(-1, 1)

    # # random data inputs
    # X, y = np.random.rand(500, 1, 28, 28), np.random.randint(0, 9, (500, 1))
    # # print shapes
    # print("X shape:", X.shape)
    # print("y shape:", y.shape)

    # split into training and testing data
    X_train, X_test = X[:2 * len(X) // 3], X[2 * len(X) // 3:]
    y_train, y_test = y[:2 * len(y) // 3], y[2 * len(y) // 3:]

    # %%
    '''Visualize the data'''
    # Generate random indices
    rand_indices = np.random.randint(0, len(X_train), 9)

    # put number images into a collection and include their respective labels
    number_collection = [[np.reshape(X_train[i], (28, 28)), y_train[i]] for i in rand_indices]

    # plot the collection of number images
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through the data to plot each image with its label
    for i, (image, label) in enumerate(number_collection):
        axes[i].imshow(image, cmap='gray')  # Assume images are grayscale
        axes[i].title.set_text(f'Label: {label}')
        axes[i].axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()

    # %%
    '''
    initiate training
    '''
    # Define model
    model = ExampleMA()

    # Define binary crossentropy loss
    CCELoss = ma.CCE()

    # Define optimizer
    optimizer = ma.RMSProp(0.001, 0.9)

    training_losses, training_accuracies = train(X_train, y_train, model, CCELoss, optimizer, 128, 100)

    # %%
    '''Visualize the result of training'''

    plt.plot(training_losses, label="Training Loss")
    # plt.plot(testing_losses, label="Testing Loss")
    plt.title("Losses over time")
    plt.show()

    plt.plot(training_accuracies, label="Training Accuracy")
    # plt.plot(testing_accuracies, label="Testing Accuracy")
    plt.title("Accuracies over time")
    plt.show()

    # %%
    '''Visualize the result'''
    # Generate random indices
    rand_indices = np.random.randint(0, len(X_test), 25)

    # put number images into a collection and include their respective labels
    number_collection = np.array([X_test[i] for i in rand_indices])

    predictions = np.argmax(model(number_collection), axis=1)

    # plot the collection of number images
    fig, axes = plt.subplots(5, 5, figsize=(9, 9))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through the data to plot each image with its label
    for i in range(len(number_collection)):
        axes[i].imshow(number_collection[i].reshape(28, 28), cmap='gray')  # Assume images are grayscale
        axes[i].title.set_text(f'Label: {predictions[i]}')
        axes[i].axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()

# %%
