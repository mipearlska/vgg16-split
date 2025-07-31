import numpy as np
from tensorflow import keras
import pickle

# Download and preprocess the MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Save to local files
with open("mnist_data.pkl", "wb") as f:
    pickle.dump(((x_train, y_train), (x_test, y_test)), f)

print("Dataset saved to mnist_data.pkl")
