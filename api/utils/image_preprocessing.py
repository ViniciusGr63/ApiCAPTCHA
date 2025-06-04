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

def preprocess_image(image_bytes):
    
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    binary = np.array(image) < 128

    coords = np.argwhere(binary)
    if coords.size == 0:
        return np.array([[0, 0, 0]])

    #coordenadas 
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    

    image_cv = np.array(image)
    contours, _ = cv2.findContours(image_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    area = 0
    perimeter = 0
    num_vertices = 0
    
    for contour in contours:
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull) 
        perimeter = cv2.arcLength(hull, True)  
        approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)  
        num_vertices = len(approx)  
        
    # Retorno(área, perímetro, número de vértices)
    return np.array([[area, perimeter, num_vertices]])
