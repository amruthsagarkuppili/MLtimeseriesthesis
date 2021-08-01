# app.py
from flask import Flask


def init_app():
    app = Flask(__name__)  # create an app instance
    with app.app_context():

        app.secret_key = 'Amruth Secret Key'

        # Import parts of our core Flask app
        from . import routes
        # Register Blueprints
        app.register_blueprint(routes.home_bp)

        from . import dbroutes

        app.register_blueprint(dbroutes.db_bp)

        # Import Dash application
        from .dashapplication.dashboard import init_dashboard
        app = init_dashboard(app)

        return app
