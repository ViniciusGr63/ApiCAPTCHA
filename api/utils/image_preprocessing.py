# from PIL import Image
# import numpy as np
# import io

# def preprocess_image(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert('L')
#     binary = np.array(image) < 128                            # Binariza a imagem

#     coords = np.argwhere(binary)
#     if coords.size == 0:
#         return np.array([[0, 0, 0]])                          # Retorna 3 características (largura, altura, 0)

#     y_min, x_min = coords.min(axis=0)
#     y_max, x_max = coords.max(axis=0)
#     width = x_max - x_min
#     height = y_max - y_min

#     # Adicionando uma característica extra (area)
#     area = width * height  
#     features = np.array([width, height, area]).reshape(1, -1)  # Agora 3 características
#     return features

from PIL import Image
import numpy as np
import io
import cv2


def preprocess_image(image_bytes, img_size=(28, 28)):
    # Abre imagem a partir dos bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # escala de cinza
    
    # Converte para numpy array
    img_np = np.array(image)
    
    # Redimensiona para img_size usando cv2
    img_resized = cv2.resize(img_np, img_size)
    
    # Normaliza pixels para [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Achata para vetor 1D
    img_flat = img_normalized.flatten().reshape(1, -1)
    
    return img_flat

   
