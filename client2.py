# client2_service.py
import numpy as np
import requests
import pickle

import numpy as np
from tensorflow import keras


# The data, split between train and test sets
with open("slice2_input.pkl", "rb") as f:
    input = pickle.load(f)

# Serialize the input
data = pickle.dumps(input)

# Send POST request with binary data
response = requests.post('http://localhost:5002/predict', data=data, headers={'Content-Type': 'application/octet-stream'})

# Deserialize the prediction
if response.ok:
    pred = pickle.loads(response.content)
    print("Prediction shape:", pred.shape)
    print("Predicted Label:", np.argmax(pred))

else:
    print("Error:", response.text)
