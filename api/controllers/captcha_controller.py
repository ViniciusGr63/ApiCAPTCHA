from flask import request, jsonify
from ..models.captcha_model import MLModels
from ..utils.image_preprocessing import preprocess_image

ml_models = MLModels()

def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    if 'shape' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()

    shape_exp = request.form.get('shape')
    
    
    features = preprocess_image(image_bytes) 
    shape = ml_models.predict_shape(features,shape_exp)
    
    return jsonify({'result': bool(shape)})

