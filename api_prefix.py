from flask import Blueprint
from routes.health_route import health_bp
from routes.upload_img_route import upload_img_bp
from routes.login_route import login_route_bp
from routes.samples_route import samples_route_bp
from routes.annot_route import annote_route_bp
# from routes.cells_route import cells_route_bp
from routes.pulmo_route import pulmo_route_bp
from routes.patient_route import patient_route_bp
from routes.dynamic_cells_route import dynamic_cells_route_bp
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

api_bp.register_blueprint(health_bp)
api_bp.register_blueprint(upload_img_bp)
api_bp.register_blueprint(login_route_bp)
api_bp.register_blueprint(samples_route_bp)
api_bp.register_blueprint(annote_route_bp)
# api_bp.register_blueprint(cells_route_bp)
api_bp.register_blueprint(patient_route_bp)
api_bp.register_blueprint(dynamic_cells_route_bp)
api_bp.register_blueprint(pulmo_route_bp)