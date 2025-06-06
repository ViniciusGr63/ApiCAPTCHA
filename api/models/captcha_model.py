
from ..modelsML.shapes import Shapes

class MLModels:
    def __init__(self):
        self.shape = Shapes()
        self.shape.load_model()

    def predict_shape(self, features, shape_exp):
        response = self.shape.predict(features)  # deve retornar uma string, ex: "circle"

        # Garante que ambos estão em lowercase e sem espaços extras
        return response.strip().lower() == shape_exp.strip().lower()
