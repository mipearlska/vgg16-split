# head_service.py
from flask import Flask, request, Response
import numpy as np
import pickle
import requests
from tensorflow.keras.models import load_model

app = Flask(__name__)
head = load_model('split1.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive binary data and deserialize
        input_data = pickle.loads(request.data)
        pred = head.predict(input_data)

        # Serialize prediction and forward to next service with client info
        # payload = {
        #     'prediction': pred,
        #     'client_ip': request.remote_addr,
        #     'client_headers': dict(request.headers)
        # }
        requests.post('http://192.168.122.110:30002/predict',
                     data=pickle.dumps(pred), 
                     headers={'Content-Type': 'application/octet-stream'})
        
        # Don't return anything - let the second entity handle the response
        return '', 202  # 202 Accepted - request received but processing continues elsewhere
    except Exception as e:
        return Response(str(e), status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
