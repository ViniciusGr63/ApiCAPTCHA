
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
        # Exemplo simples: roda todos e retorna qual modelo tem maior confiança
        # ou simplesmente roda cada um como classificador binário
        # Aqui vamos supor que cada modelo prevê 1 (não) ou 0 (sim)
        results = {
            'triangle': self.triangle.predict(features)[1],
            # 'square': self.square.predict(features)[1],
            # 'circle': self.circle.predict(features)[1],
            # 'x': self.x.predict(features)[1]
        }
        
        # Seleciona o modelo que retornou 1, ou nenhum
        for shape, pred in results.items():
            if pred == 0:
                return shape
        return "unknown"
        
