import base64
from io import BytesIO
from flask import Flask, request, jsonify

from model import load_model, predict_age

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    img_bytes = None
    if 'image' in request.files:
        img_bytes = request.files['image'].read()
    else:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'image is required'}), 400
        img_bytes = base64.b64decode(data['image'])

    age = predict_age(img_bytes, model)
    return jsonify({'age': int(age)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
