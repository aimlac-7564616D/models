from flask import Flask
from flask_cors import CORS
from app.database import init_app_database

import app.power_routes as power

from flask import g


def create_app():
    app = Flask(__name__)

    # use the development configuration if FLASK_ENV == 'production'
    if app.config["ENV"] == "production":
        app.config.from_object("app.flask_config.Production")
    else:
        app.config.from_object("app.flask_config.Development")

    # register CORS
    CORS(app)

    # register database
    init_app_database(app)

    app.register_blueprint(power.bp)

    # test connection
    app.route("/ok")(lambda: "OK")
    return app
