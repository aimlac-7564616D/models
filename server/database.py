import mariadb

from flask import current_app, g


def _connect_to_database():
    """establish and return a connection object"""
    try:
        conn = mariadb.connect(
            user=current_app.config["MARIADB_USER"],
            password=current_app.config["MARIADB_PASSWORD"],
            host=current_app.config["MARIADB_HOST"],
            port=current_app.config["MARIADB_PORT"],
            database=current_app.config["MARIADB_DATABASE"],
        )
    except mariadb.Error as e:
        current_app.logger.warn(f"Could not connect to database: '{e}'")
        conn = None
    else:
        current_app.logger.info(f"Successfully connected to database")
    return conn


def get_database_conn():
    """creates and/or returns conn object stored in g._db_conn"""
    if "_db_conn" not in g:
        g._db_conn = _connect_to_database()

    return g._db_conn


def teardown_database(env):
    """teardown function to remove g._db_conn if exists"""
    conn = g.pop("_db_conn", None)

    if conn is not None:
        current_app.logger.info("Disconnecting from database")
        conn.close()


def get_database_cursor():
    return g.get_conn().cursor()


def execute_query(*args, **kwargs):
    cursor = g.get_cursor()
    cursor.execute(*args, **kwargs)
    return cursor.fetchall()


def register_database_to_context():
    """registers function to retrieve connection in g.get_conn"""
    g.get_conn = get_database_conn
    g.get_cursor = get_database_cursor
    g.query = execute_query


def init_app_database(app):
    """application init to register functions and cli commands"""
    app.teardown_appcontext(teardown_database)
    app.before_request(register_database_to_context)
