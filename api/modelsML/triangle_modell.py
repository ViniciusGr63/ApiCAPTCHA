import os
import joblib
from sklearn.neural_network import MLPClassifier
import cv2
import numpy as np



# Caminho absoluto para a pasta com as imagens de triângulos
base_path = os.path.dirname(os.path.realpath(__file__))  # Obtém o caminho do script atual

# Corrigir a criação do caminho para a pasta de imagens
image_folder = os.path.join(base_path, 'triangles_images')  # Não adicionar 'modelsML' duas vezes

# Para a imagem de teste
new_image_path = os.path.join(base_path, 'triangles_images', 'trianguloML.png')


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


# Função para extrair características de imagens
def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Erro ao carregar a imagem: {image_path}")

    # Redimensionar a imagem
    image = cv2.resize(image, (420, 620))

    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binaria a imagem
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)  # Perímetro
        area = cv2.contourArea(hull)  # Área

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)

        # Número de vértices
        num_vertices = len(approx)

        # Adicionar as características
        features.append([area, perimeter, num_vertices])

    # Se não encontrar contornos, retornar vazio
    if len(features) == 0:
        return []

    # Retorna apenas o primeiro conjunto de características para cada imagem
    return features[0]


# Função para treinar o modelo com imagens reais de triângulos
def train_model_with_images(image_folder):
    X = []
    y = []

    # Verifica se o diretório de imagens existe
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"O diretório {image_folder} não foi encontrado")

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        # Extrai as características da imagem
        features = extract_features(image_path)

        # Se a imagem tiver características extraídas, adicione ao conjunto de dados
        if features:
            X.append(features)  # Adiciona o vetor de características
            y.append(0)  # Rótulo "0" para triângulo

    # Converte X e y para arrays do numpy
    X = np.array(X)
    y = np.array(y)

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

    # O modelo espera uma lista de características
    prediction = model.predict([new_features])  # Passa uma lista de características
    print("Previsão da forma:", "Triângulo" if prediction[0] == 0 else "Desconhecido")


# Função principal para executar
if __name__ == "__main__":

    
    # Treinar o modelo com imagens de triângulos
    model = train_model_with_images(image_folder)

    # Testar com uma nova imagem (garanta que a imagem está no mesmo diretório do script)

    predict_shape(new_image_path, model)  # Prever a forma com a imagem
