from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    binary = np.array(image) < 128  # Binariza a imagem

    coords = np.argwhere(binary)
    if coords.size == 0:
        return np.array([[0, 0, 0]])  # Retorna 3 características (largura, altura, 0)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min

    # Adicionando uma característica extra (por exemplo, área ou perímetro)
    area = width * height  # Exemplificando com a área
    features = np.array([width, height, area]).reshape(1, -1)  # Agora 3 características
    return features
