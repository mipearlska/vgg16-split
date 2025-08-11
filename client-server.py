# head_service.py
from flask import Flask, request, Response
import numpy as np
import pickle
# from tensorflow.keras.models import load_model

app = Flask(__name__)
# head = load_model('split2.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive binary data and deserialize
        input_data = pickle.loads(request.data)
        # pred = head.predict(input_data)

        print("Prediction shape:", input_data.shape)
        print("Predicted Label:", np.argmax(input_data))

        # Serialize prediction and return as binary
        return Response(status=200)
    except Exception as e:
        print("Error during prediction:", str(e))
        return Response(str(e), status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
