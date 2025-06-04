from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    binary = np.array(image) < 128  # Binariza, assumindo forma escura no fundo claro

    coords = np.argwhere(binary)
    if coords.size == 0:
        # Se não tem forma detectada, retorna zeros
        return np.array([[0, 0]])
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min

    features = np.array([width, height]).reshape(1, -1)
    return features
