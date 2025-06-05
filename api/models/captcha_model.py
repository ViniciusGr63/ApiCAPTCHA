
from ..modelsML.shapes import Shapes

import numpy as np




class MLModels:
    def __init__(self):
        self.shape = Shapes()
        self.shape.load_model()
        self.class_map = {0: 'triangle', 1: 'square', 2: 'circle'}

    def predict_shape(self, features):
        pred_class_idx = self.shape.predict(features)
        return self.class_map.get(pred_class_idx, "unknown")
