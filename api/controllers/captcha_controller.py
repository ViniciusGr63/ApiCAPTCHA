from flask import request, jsonify
from ..models import MLModels
from ..utils.image_preprocessing import preprocess_image

ml_models = MLModels()

def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    
    features = preprocess_image(image_bytes)  # já retorna vetor (1, -1)
    
    shape = ml_models.predict_shape(features)
    
    return jsonify({'shape': shape})
