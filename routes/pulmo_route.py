from flask import Blueprint, request, jsonify, current_app
from utilities.auth_utility import protected
from utilities.logging_utility import ActivityLogger
from utilities.aws_utility import upload_to_s3, S3_BUCKET_NAME, get_presigned_url
from db.models import Patient, db
import os

pulmo_route_bp = Blueprint('pulmo_route', __name__ )

@pulmo_route_bp.route('/pulmo/upload-report', methods=['POST'])
@protected
def upload_report(decoded_token):
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")

    
    current_app.logger.info(f"Upload report request received - user_id: {user_id}, user_role: {user_role}")

    ActivityLogger.log_workflow_step(
        user_id=user_id,
        user_role=user_role,
        workflow_type="upload_report",
        step="start_upload_report",
        step_data={"endpoint": "/pulmo/upload-report"}
    )

    if user_role != "UPLOADER":
        current_app.logger.warning(f"Unauthorized upload attempt - user_id: {user_id}, role: {user_role}")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="unauthorized_upload_attempt",
            status="failure"
        )
        return jsonify({"message": "Unauthorized access"}), 403

    if 'report' not in request.files:
        current_app.logger.error(f"No report file provided - user_id: {user_id}")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="upload_report",
            status="failure",
            metadata={"reason": "no_file_provided"}
        )
        return jsonify({"message": "No report file provided"}), 400
    
    report_file = request.files['report']
    if report_file.filename == '':
        current_app.logger.error("Empty filename in upload request")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="upload_error",
            action_details="empty_filename",
            status="error"
        )
        return jsonify({"message": "No selected file"}), 400

    patient_id = request.form.get("patientId")

    if not patient_id:
        current_app.logger.error("Missing required field: Patient ID")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="upload_error",
            action_details="missing_patient_id",
            status="error"
        )
        return jsonify({"message": "Patient ID is required"}), 400
        # Handle patient creation/lookup

    current_app.logger.info(f"Looking up patient with Patient ID: {patient_id}")
    patient = Patient.query.filter_by(patient_id=patient_id).first()

    if not patient:
        current_app.logger.info(f"Patient not found, Hence upload report not possible.")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="upload_error",
            action_details="patient_not_found",
            status="error"
        )
        return jsonify({"message": "Patient not found"}), 404

      
    _, ext = os.path.splitext(report_file.filename or "")
    if ext == "":
        ext = ".csv"

    s3_prefix = f"uploads/reports/{patient_id}/"
    s3_object_key = f"{s3_prefix}report{ext}"

    current_app.logger.info(f"Starting S3 upload - bucket: {S3_BUCKET_NAME}, key: {s3_object_key}")
    
    try:
        upload_to_s3(S3_BUCKET_NAME, report_file, s3_object_key)
        current_app.logger.info(f"S3 upload successful, s3_key: {s3_object_key}")
        ActivityLogger.log_workflow_step(
            user_id=user_id,
            user_role=user_role,
            workflow_type="upload_report",
            step="s3_upload_success",
            step_data={"s3_key": s3_object_key}
        )
    except Exception as e:
        current_app.logger.error(f"S3 upload failed, error: {str(e)}", exc_info=True)
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="upload_error",
            action_details="s3_upload_failed",
            status="error",
            metadata={"s3_key": s3_object_key, "error": str(e)}
        )
        return jsonify({"message": "S3 upload failed", "details": str(e)}), 500

    patient.pulmonologist_report_s3_key = s3_object_key
    try:
        db.session.commit()
        current_app.logger.info(f"Patient record updated with report S3 key - patient_id: {patient_id}")
        ActivityLogger.log_workflow_step(
            user_id=user_id,
            user_role=user_role,
            workflow_type="upload_report",
            step="db_patient_record_update_success",
            step_data={"patient_id": patient_id, "s3_key": s3_object_key}
        )   

    except Exception as e:
        current_app.logger.error(f"Database update failed, error: {str(e)}", exc_info=True)
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="db_error",
            action_details="patient_update_failed",
            status="error",
            metadata={"patient_id": patient_id, "error": str(e)}
        )
        return jsonify({"message": "Database update failed", "details": str(e)}), 500

    current_app.logger.info(f"Upload report completed successfully - s3_key: {s3_object_key}")

    ActivityLogger.log_workflow_step(
        user_id=user_id,
        user_role=user_role,
        workflow_type="upload_report",
        step="upload_report_complete",
        step_data={"s3_key": s3_object_key}
    )

    return jsonify({"message": "Report uploaded successfully", "s3_key": s3_object_key}), 200


# GET REPORT PLACEHOLDER
@pulmo_route_bp.route('/pulmo/<patient_id>/report', methods=['GET'])
@protected
def get_report(decoded_token, patient_id):
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")

    current_app.logger.info(f"Get report request received - user_id: {user_id}, user_role: {user_role}, patient_id: {patient_id}")

    ActivityLogger.log_workflow_step(
        user_id=user_id,
        user_role=user_role,
        workflow_type="get_report",
        step="start_get_report",
        step_data={"endpoint": f"/pulmo/{patient_id}/report"}
    )

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        current_app.logger.info(f"Patient not found for report download - patient_id: {patient_id}")
        return jsonify({"message": "Patient not found"}), 404
    s3_key = patient.pulmonologist_report_s3_key
    if not s3_key:
        current_app.logger.info(f"No pulmonologist report uploaded for patient_id: {patient_id}")
        return jsonify({"message": "No pulmonologist report uploaded for this patient."}), 404
    try:
        s3_url = get_presigned_url(S3_BUCKET_NAME, s3_key)
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_report_success",
            status="success",
            metadata={"patient_id": patient_id, "s3_key": s3_key}
        )
        return jsonify({"message": "Report fetched successfully", "report_url": s3_url}), 200
    
    except Exception as e:
        current_app.logger.error(f"Failed to fetch report from S3: {str(e)}", exc_info=True)
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_report_failed",
            status="error",
            metadata={"patient_id": patient_id, "error": str(e)}
        )
        return jsonify({"message": "Failed to fetch report", "details": str(e)}), 500



@pulmo_route_bp.route('/pulmo/patients', methods=['GET'])
@protected
def get_patients(decoded_token):
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")

    current_app.logger.info(f"Get patients request received - user_id: {user_id}, user_role: {user_role}")

    ActivityLogger.log_workflow_step(
        user_id=user_id,
        user_role=user_role,
        workflow_type="get_patients",
        step="start_get_patients",
        step_data={"endpoint": "/pulmo/patients"}
    )

    if user_role != "UPLOADER":
        current_app.logger.warning(f"Unauthorized access attempt - user_id: {user_id}, role: {user_role}")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="unauthorized_access_attempt",
            status="failure"
        )
        return jsonify({"message": "Unauthorized access"}), 403

    # Pagination
    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=10, type=int)
    patients = Patient.query.filter_by(pulmonologist_id=user_id).paginate(page=page, per_page=per_page, error_out=False)

    paginated_items = []
    for patient in patients.items:
        paginated_items.append({
            "patient_id": patient.patient_id,
            "user_typed_id": patient.user_typed_id,
            "pulmonologist_report_s3_key": patient.pulmonologist_report_s3_key if patient.pulmonologist_report_s3_key else None,
            "total_samples": len(patient.samples) if patient.samples else 0
        })

    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="get_patients_list",
        status="success",
        metadata={"page": page, "per_page": per_page, "total_patients": patients.total}
    )

    return jsonify({
        "page": patients.page,
        "per_page": patients.per_page,
        "total": patients.total,
        "pages": patients.pages,
        "patients": paginated_items
    }), 200
    
