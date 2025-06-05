# import os
# import joblib
# import cv2
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# #https://www.kaggle.com/datasets/cactus3/basicshapes/data


# class Shapes:
#     def __init__(self, dataset_path='shapes', img_size=(28,28)):
#         self.dataset_path = dataset_path
#         self.img_size = img_size
#         self.class_names = ['circles', 'triangles', 'squares']
#         self.label_map = {name: idx for idx, name in enumerate(self.class_names)}
#         self.model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

#     def load_data(self):
#         X, y = [], []
#         for label_name in self.class_names:
#             folder = os.path.join(self.dataset_path, label_name)
#             for filename in os.listdir(folder):
#                 filepath = os.path.join(folder, filename)
#                 img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#                 if img is None:
#                     continue
#                 img = cv2.resize(img, self.img_size)
#                 img = img.astype(np.float32) / 255.0  # Normaliza para [0,1]
#                 X.append(img.flatten())
#                 y.append(self.label_map[label_name])
#         return np.array(X), np.array(y)

#     def train(self):
#         X, y = self.load_data()
#         X_train, X_val, y_train, y_val = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y)
        
#         self.model.fit(X_train, y_train)
#         y_pred = self.model.predict(X_val)

#         print("Relatório de classificação na validação:")
#         print(classification_report(y_val, y_pred, target_names=self.class_names))

#     def predict(self, img_path):
#         # Carrega imagem e converte para escala de cinza
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise ValueError("Imagem não encontrada ou inválida.")
        
#         # Redimensiona para tamanho esperado
#         img = cv2.resize(img, self.img_size)
        
#         # Normaliza pixels para [0,1]
#         img = img.astype(np.float32) / 255.0
        
#         # Achata imagem para vetor e prepara para predição
#         img_flat = img.flatten().reshape(1, -1)
        
#         pred_idx = self.model.predict(img_flat)[0]
#         return self.class_names[pred_idx]

#     def save_model(self, path='shapes_model.joblib'):
#         joblib.dump(self.model, path)

#     def load_model(self, path='shapes_model.joblib'):
#         self.model = joblib.load(path)


# if __name__ == "__main__":
#     shapes = Shapes()

#     # Se for a primeira vez, treine e salve o modelo:
#     # shapes.train()
#     # shapes.save_model()

#     # Depois, carregue o modelo salvo
#     shapes.load_model('shapes_model.joblib')

#     # Prever imagem nova fora do padrão original
#     resultado = shapes.predict('square.jpg')
#     print(f"A forma na imagem 'square.jpg' é: {resultado}")
    
import os
import joblib
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class Shapes:
    def __init__(self, dataset_path=None, img_size=(28,28)):
        # Ajusta dataset_path para caminho absoluto relativo ao arquivo
        if dataset_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))  # api/modelsML
            self.dataset_path = os.path.join(base_dir, 'shapes')
        else:
            self.dataset_path = dataset_path
            
        self.img_size = img_size
        self.class_names = ['circles', 'triangles', 'squares']
        self.label_map = {name: idx for idx, name in enumerate(self.class_names)}
        self.model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

    def load_data(self):
        X, y = [], []
        for label_name in self.class_names:
            folder = os.path.join(self.dataset_path, label_name)
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Pasta para classe '{label_name}' não encontrada: {folder}")
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.img_size)
                img = img.astype(np.float32) / 255.0  # Normaliza para [0,1]
                X.append(img.flatten())
                y.append(self.label_map[label_name])
        return np.array(X), np.array(y)

    def train(self):
        X, y = self.load_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)

        print("Relatório de classificação na validação:")
        print(classification_report(y_val, y_pred, target_names=self.class_names))

        def predict(self, features):
        # features já é um vetor numpy (1, -1)
          pred_idx = self.model.predict(features)[0]
        return self.class_names[pred_idx]

    def preprocess_image_from_path(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Imagem não encontrada ou inválida: {img_path}")
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        return img.flatten().reshape(1, -1)

    def save_model(self, path=None):
        if path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, 'shapes_model.joblib')
        joblib.dump(self.model, path)

    def load_model(self, path=None):
        if path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, 'shapes_model.joblib')
        self.model = joblib.load(path)


if __name__ == "__main__":
    shapes = Shapes()

    # Descomente para treinar e salvar o modelo uma vez:
    # shapes.train()
    # shapes.save_model()

    # Depois carregue o modelo salvo:
    shapes.load_model()

    # Teste uma previsão
    resultado = shapes.predict('square.jpg')
    print(f"A forma na imagem 'square.jpg' é: {resultado}")
