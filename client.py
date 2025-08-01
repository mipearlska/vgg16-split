# client_service.py
import numpy as np
import requests
import pickle

# Prepare input
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# The data, split between train and test sets
with open("mnist_data.pkl", "rb") as f:
    (x_train, y_train), (x_test, y_test) = pickle.load(f)


# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

tst = np.expand_dims(x_test[0], 0)

# Serialize the input
data = pickle.dumps(tst)

# Send POST request with binary data
response = requests.post('http://localhost:5001/predict', data=data, headers={'Content-Type': 'application/octet-stream'})

# Deserialize the prediction
if response.ok:
    pred = pickle.loads(response.content)
    print("Prediction shape:", pred.shape)

    with open("slice2_input.pkl", "wb") as f:
        pickle.dump(pred, f)
        print("Prediction saved to 'slice2_input.pkl'")

else:
    print("Error:", response.text)
