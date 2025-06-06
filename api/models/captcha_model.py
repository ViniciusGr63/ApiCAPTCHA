
from ..modelsML.shapes import Shapes

import numpy as np


class MLModels:
    def __init__(self):
        self.shape = Shapes()
        self.shape.load_model()

    def predict_shape(self, features):
        return self.shape.predict(features)
