# app captcha 
# python -m api.captcha_app
# source venv/bin/activate 
# gunicorn "api.captcha_app:create_app()"

# python triangle_model.py dentro da pasta mlmodels, gera o pkl

#pip freeze > requirements.txt atualiza requirements



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
# │   │     ├── shapes/
# │   │     │  ├── circles           
# │   │     │  ├── squares       
# │   │     │  ├── triangles         
# │   │     ├── shapes_model.joblib          # treino
# │   │     └── shapes.py                    # Rede neural para X                    
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



from flask import Flask
from flask_cors import CORS
from api.captcha_routes import main

def create_app():
    app = Flask(__name__)

    CORS(app)


    # Libera o CORS para origens específicas
    CORS(app, resources={r"/captcha/*": {"origins": [
        "https://obscure-spoon-4xjj4wgrxj72jvr-3000.app.github.dev"
    ]}})

    app.register_blueprint(main)

    @app.route('/')
    def home():
        return "Olá, Flask!"

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
