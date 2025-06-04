import os
import joblib
from sklearn.neural_network import MLPClassifier
import cv2
import numpy as np

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


# Função para extrair características da imagem
def extract_features(image_path):
    # Carregar imagem
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Erro ao carregar a imagem: {image_path}")

    # Redimensionar a imagem
    image = cv2.resize(image, (420, 620))  # Tamanho padrão

    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar threshold (binarizar a imagem)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Características geométricas
    features = []
    
    for contour in contours:
        # Contorno convexo
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)  # Perímetro
        area = cv2.contourArea(hull)  # Área

        # Aproximação de polígonos (simplificar o contorno)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)

        # Contagem de vértices (dependendo da forma, podemos usar esse número para distinguir formas)
        num_vertices = len(approx)

        # Adicionar as características
        features.append([area, perimeter, num_vertices])

    # Se não encontrar contornos, retornar uma lista vazia
    if len(features) == 0:
        return []

    return features


# Função para treinar o modelo
def train_model():
    # Dados fictícios para treino (essas são as características das imagens já extraídas)
    X = [
        [500, 100, 3],  # Exemplo de um triângulo
        [900, 120, 4],  # Exemplo de um quadrado
        [700, 150, 5]   # Exemplo de um pentágono
    ]
    y = [0, 1, 2]  # 0 = Triângulo, 1 = Quadrado, 2 = Pentágono

    # Criar e treinar o modelo
    model = TriangleModel()
    model.train(X, y)
    return model


# Função para usar o modelo treinado e prever uma nova imagem
def predict_shape(image_path, model):
    new_features = extract_features(image_path)

    if len(new_features) == 0:
        print(f"Não foi possível extrair características da imagem: {image_path}")
        return

    # O modelo espera uma lista de características. Podemos passar as características de uma imagem
    predictions = model.predict(new_features)
    print("Previsão da forma:", predictions)


# Função principal para executar
if __name__ == "__main__":
    model = train_model()  # Treinar o modelo

    # Testar com uma nova imagem (garanta que a imagem está no mesmo diretório do script)
    new_image_path = 'trianguloML.png'  # Caminho correto para a imagem
    predict_shape(new_image_path, model)  # Prever a forma com a imagem
