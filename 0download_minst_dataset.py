import numpy as np
from tensorflow import keras
import pickle

# Download and preprocess the MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Save to local files
with open("mnist_data.pkl", "wb") as f:
    pickle.dump(((x_train, y_train), (x_test, y_test)), f)

print("Dataset saved to mnist_data.pkl")
