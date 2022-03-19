import os

SECRET_KEY = os.urandom(1024)


class Config(object):
    MARIADB_USER = "aimlac"
    MARIADB_PASSWORD = "aimlac"
    MARIADB_DATABASE = "llanwrytd"
    SECRET_KEY = SECRET_KEY
    MARIADB_PORT = 3306


class Development(Config):
    DEBUG = True
    TESTING = True
    FLASK_ENV = "development"
    MARIADB_HOST = "127.0.0.1"


class Production(Config):
    DEBUG = False
    TESTING = False
    MARIADB_HOST = "aimlac-database"
