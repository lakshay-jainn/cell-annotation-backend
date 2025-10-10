from flask import Blueprint,jsonify
from utilities.auth_utility import protected
health_bp = Blueprint("health",__name__)
@health_bp.route("/health", methods=["GET"])
@protected
def health(decoded_token):
    return jsonify({"status": "ok"})