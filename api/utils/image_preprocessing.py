from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes):
    # Exemplo: abre imagem, converte para grayscale e flatten
    image = Image.open(io.BytesIO(image_bytes)).convert('L').resize((28, 28))
    data = np.array(image).flatten() / 255.0
    return data.reshape(1, -1)  # formato esperado pelo sklearn
