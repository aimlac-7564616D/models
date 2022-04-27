from flask import Flask
from flask_cors import CORS
from server.database import init_app_database

from server.routes import routes

from flask import g


def create_app():
    app = Flask(__name__)

    if app.config["ENV"] == "production":
        app.config.from_object("server.flask_config.Production")
    else:
        app.config.from_object("server.flask_config.Development")

    # register CORS
    CORS(app)

    # register database
    init_app_database(app)
    for route in routes:
        app.register_blueprint(route)

    # test connection
    app.route("/ok")(lambda: "OK")
    return app
