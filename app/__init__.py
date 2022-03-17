from flask import Flask
from flask_cors import CORS

import app.power as power

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('config.py')

    CORS(app)

    app.register_blueprint(power.bp)

    # test connection
    app.route('/ok')(lambda : 'OK')
    return app