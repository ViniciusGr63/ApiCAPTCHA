
from ..modelsML.triangle_model import TriangleModel
# from ..modelsML.square_model import SquareModel
# from ..modelsML.circle_model import CircleModel
# from ..modelsML.x_model import XModel
import numpy as np

class MLModels:
    def __init__(self):
        self.triangle = TriangleModel()
        self.triangle.load()
        
        # self.square = SquareModel()
        # self.square.load()
        # self.circle = CircleModel()
        # self.circle.load()
        # self.x = XModel()
        # self.x.load()

    def predict_shape(self, features):
        results = {
            'triangle': self.triangle.predict(features)[0],  # pegar o primeiro e único valor
            # 'square': self.square.predict(features)[0],
            # 'circle': self.circle.predict(features)[0],
            # 'x': self.x.predict(features)[0]
        }
        
        for shape, pred in results.items():
            
            if pred == 0:  
                return shape
        return "unknown"
