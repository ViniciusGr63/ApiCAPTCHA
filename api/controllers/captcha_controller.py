from flask import request, jsonify
from ..models.captcha_model import MLModels

from ..utils.image_preprocessing import preprocess_image

ml_models = MLModels()

def recognize():
    # Recebe imagem (exemplo via base64 ou multipart/form-data)
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image = image_file.read()
    features = preprocess_image(image)# Pré-processa a imagem para extrair features compatíveis com ML
    shape = ml_models.predict_shape(features) # Prediz a forma

    return jsonify({'shape': shape})
