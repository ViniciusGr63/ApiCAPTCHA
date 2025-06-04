# app captcha 
# python -m api.captcha_app
# source venv/bin/activate



# ApiCAPTCHA/
# ├── api/
# │   ├── captcha_app.py                 # inicia o app
# │   ├── captcha_routes.py              # Define as rotas e aponta para controllers
# │   │       
# │   ├── controllers/
# │   │     └── captcha_controller.py    # Controller para CAPTCHA
# │   ├── models/
# │   │     └── captcha_model.py         # Wrapper para os 4 modelos ML   
# │   ├── modelsML/                      # Modelos treinados (pkl)
# │   │     ├── base_model.py            # Classe base comum para as redes
# │   │     ├── triangle_model.py        # Rede neural para Triângulo
# │   │     ├── square_model.py          # Rede neural para Quadrado
# │   │     ├── circle_model.py          # Rede neural para Círculo
# │   │     └── x_model.py               # Rede neural para X                    
# │   │     
# │   └── utils/
# │         └── image_preprocessing.py   # Funções para tratar a imagem recebida
# │                  
# ├── venv
# └── requirements.txt





# scikit-learn (MLPClassifier)
# Usuário envia uma imagem para POST /captcha/recognize (via form-data).
# Controller recebe a imagem, chama pré-processamento.
# Features geradas são enviadas para o modelo ML.
# Modelo retorna qual forma reconheceu.
# API responde com JSON: { "shape": "triangle" }, por exemplo.



from api.captcha_routes import main  
from flask import Flask


def create_app():
    app = Flask(__name__)
    app.register_blueprint(main)


    @app.route('/')
    def home():
        return "Olá, Flask!"

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
