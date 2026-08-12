"""Signity Flask application factory."""

import os
import sys
from urllib import request

from cuda_config import prepare_tensorflow_cuda

prepare_tensorflow_cuda()

from flask import Flask

from config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    FLASK_DEBUG,
    MAX_CONTENT_LENGTH,
    SECRET_KEY,
    SESSION_COOKIE_HTTPONLY,
    SESSION_COOKIE_SECURE,
    SESSION_COOKIE_SAMESITE,
)

from database import initialize_database
from shared import csrf_token
from routes.admin import bp as admin_bp
from routes.auth import bp as auth_bp
from routes.main import bp as main_bp
from routes.predict import bp as predict_bp
from routes.profile import bp as profile_bp


def create_app():
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=SECRET_KEY,
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
        SESSION_COOKIE_HTTPONLY=SESSION_COOKIE_HTTPONLY,
        SESSION_COOKIE_SAMESITE=SESSION_COOKIE_SAMESITE,
        SESSION_COOKIE_SECURE=SESSION_COOKIE_SECURE,
    )

    initialize_database()

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(predict_bp)

    @app.errorhandler(413)
    def request_too_large(_error):
        from shared import api_error

        if request.path == "/predict":
            return api_error("Request is too large. Send fewer or smaller frames.", 413)
        return "Request is too large.", 413

    @app.context_processor
    def inject_template_globals():
        return {"csrf_token": csrf_token}

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        debug=FLASK_DEBUG,
    )
