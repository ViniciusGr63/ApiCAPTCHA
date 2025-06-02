from flask import Blueprint
from .controllers import captcha_controller



main = Blueprint('main', __name__)


main.route('/captcha/recognize', methods=['POST'])(captcha_controller.recognize)


@main.route('/')
def index():
    return "_captcha_api"

