import os
import joblib
from sklearn.neural_network import MLPClassifier

class TriangleModel:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(base_dir, 'triangle_model.pkl')
        self.model = MLPClassifier(hidden_layer_sizes=(50,))

    def train(self, X, y):
        self.model.fit(X, y)
        self.save()

    def save(self):
        joblib.dump(self.model, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_path}")

    def predict(self, X):
        return self.model.predict(X)


# Exemplo simples de uso
if __name__ == "__main__":
    # Exemplo de dados dummy para treino
    X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = [0, 1, 0, 1]

    model = TriangleModel()
    model.train(X, y)
    model.load()
    print(model.predict([[0, 0]]))  # Deve imprimir a predição para esse input
