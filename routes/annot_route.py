from flask import Blueprint, jsonify, request, current_app
from utilities.auth_utility import protected
from utilities.logging_utility import ActivityLogger
import json
from datetime import datetime
from db.models import SampleAnnotation, PatientAnnotation, Sample, Patient, db
from utilities.aws_utility import upload_to_s3, s3_client, S3_BUCKET_NAME

annote_route_bp = Blueprint("annot", __name__)

def to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "y")
    return False   # default for None / unknown

@annote_route_bp.route("/annotate", methods=["POST"])
@protected
def annotate(decoded_token):
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="submit_annotation",
        status="start"
    )
    
    # role check
    if decoded_token.get("role") != "ANNONATOR":
        # allow both spellings for backwards compatibility, but prefer "ANNOTATOR"
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="annotation_access_denied",
            status="error",
            metadata={"reason": "insufficient_permissions", "required_role": "ANNONATOR"}
        )
        return jsonify({"message": "Unauthorized access"}), 401

    if not user_id:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="missing_user_id",
            status="error"
        )
        return jsonify({"message": "Missing user id in token"}), 401

    formdata = request.form

    # job/sample id
    job_id = formdata.get("job_id")
    if not job_id:
        current_app.logger.error("Missing job_id in annotation request")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="submit_annotation",
            status="error",
            metadata={"reason": "missing_job_id"}
        )
        return jsonify({"message": "Missing job_id"}), 400

    # Get sample to find patient_id
    sample = Sample.query.get(job_id)
    if not sample:
        current_app.logger.error(f"Sample not found for job_id: {job_id}")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="submit_annotation",
            status="error",
            metadata={"job_id": job_id, "reason": "sample_not_found"}
        )
        return jsonify({"message": "Sample not found"}), 404

    # safely get uploaded CSV (optional)
    csv_file = request.files.get('annotations_csv')  # returns None if not present
    if csv_file is not None:
        if getattr(csv_file, "filename", "") == "":
            csv_file = None

    # safe nested access - only image_quality and cells for individual slides
    image_quality = to_bool(formdata.get("image_quality"))
    cell_type_counts = formdata.get("cellTypeCounts")
    
    current_app.logger.info(f"Processing annotation for job_id: {job_id}, image_quality: {image_quality}, has_csv: {csv_file is not None}")
    
    # Remove adequacy/diagnosis from individual annotations - these are patient-level now
    # check if sample annotation already exists
    annot = SampleAnnotation.query.filter_by(user_id=user_id, sample_id=job_id).first()
    if annot:
        current_app.logger.warning(f"Annotation already exists for user: {user_id}, sample: {job_id}")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="submit_annotation",
            status="error",
            metadata={"job_id": job_id, "reason": "annotation_already_exists"}
        )
        return jsonify({"message": "Annotation already exists"}), 409

    s3_annotated_key = None
    if csv_file:
        try:
            s3_prefix = f"uploads/annotations/{user_id}/{job_id}/"
            s3_annotated_key = f"{s3_prefix}annotated.csv"
            # upload_to_s3 should accept (bucket, fileobj, key) or similar
            # If your upload_to_s3 expects bytes or filename, adapt accordingly.
            upload_to_s3(S3_BUCKET_NAME, csv_file, s3_annotated_key)
            ActivityLogger.log_workflow_step(
                user_id=user_id,
                user_role=user_role,
                workflow_type="annotation",
                step="csv_upload_success",
                step_data={"job_id": job_id, "s3_key": s3_annotated_key}
            )
        except Exception as e:
            current_app.logger.exception("S3 upload failed")
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="submit_annotation",
                status="error",
                metadata={"job_id": job_id, "reason": "s3_upload_failed", "error": str(e)}
            )
            return jsonify({"error": "S3 upload failed", "details": str(e)}), 500

    # create SampleAnnotation instance
    annot = SampleAnnotation(
        user_id=user_id,
        sample_id=job_id,
        patient_id=sample.patient_id,  # Link to patient
        s3_annotation_key=s3_annotated_key,
        image_quality=image_quality,
        cells=cell_type_counts
    )

    try:
        db.session.add(annot)
        db.session.commit()
        
        # Parse cell counts for logging
        cell_summary = {}
        if cell_type_counts:
            try:
                cell_summary = json.loads(cell_type_counts) if isinstance(cell_type_counts, str) else cell_type_counts
            except (json.JSONDecodeError, TypeError):
                cell_summary = {}
        
        ActivityLogger.log_workflow_step(
            user_id=user_id,
            user_role=user_role,
            workflow_type="annotation",
            step="annotation_saved",
            step_data={
                "job_id": job_id,
                "patient_id": sample.patient_id,
                "image_quality": image_quality,
                "cell_types_count": len(cell_summary),
                "total_cells": sum(cell_summary.values()) if cell_summary else 0
            }
        )
        
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="submit_annotation",
            status="success",
            metadata={"job_id": job_id, "annotation_id": annot.id, "has_csv": csv_file is not None}
        )
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("DB save failed")
        
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="submit_annotation",
            status="error",
            metadata={"job_id": job_id, "reason": "db_save_failed", "error": str(e)}
        )

        # attempt to remove S3 object if we uploaded it (best-effort cleanup)
        if s3_annotated_key:
            try:
                s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_annotated_key)
                ActivityLogger.log_workflow_step(
                    user_id=user_id,
                    user_role=user_role,
                    workflow_type="annotation",
                    step="cleanup_s3_after_db_failure",
                    step_data={"job_id": job_id, "s3_key": s3_annotated_key}
                )
            except Exception:
                current_app.logger.exception("Failed to cleanup S3 object after DB failure")

        return jsonify({"error": "DB save failed", "details": str(e)}), 500

    return jsonify({"message": "ok", "annotation_id": annot.id}), 201


@annote_route_bp.route("/annotate/patient", methods=["POST"])
@protected
def annotate_patient(decoded_token):
    """Submit or update patient-level annotation (adequacy, diagnosis, etc.)"""
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="submit_patient_annotation",
        status="start"
    )
    
    # role check
    if decoded_token.get("role") != "ANNONATOR":
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="patient_annotation_access_denied",
            status="error",
            metadata={"reason": "insufficient_permissions", "required_role": "ANNONATOR"}
        )
        return jsonify({"message": "Unauthorized access"}), 401

    if not user_id:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="missing_user_id",
            status="error"
        )
        return jsonify({"message": "Missing user id in token"}), 401

    data = request.get_json()
    if not data:
        return jsonify({"message": "Missing JSON data"}), 400

    # patient id
    patient_id = data.get("patient_id")
    if not patient_id:
        current_app.logger.error("Missing patient_id in patient annotation request")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="submit_patient_annotation",
            status="error",
            metadata={"reason": "missing_patient_id"}
        )
        return jsonify({"message": "Missing patient_id"}), 400

    # Verify patient exists
    patient = Patient.query.get(patient_id)
    if not patient:
        current_app.logger.error(f"Patient not found for patient_id: {patient_id}")
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="submit_patient_annotation",
            status="error",
            metadata={"patient_id": patient_id, "reason": "patient_not_found"}
        )
        return jsonify({"message": "Patient not found"}), 404

    # Extract patient-level fields
    adequacy = data.get("adequacy")
    inadequacy_reason = data.get("inadequacy_reason")
    provisional_diagnosis = data.get("provisional_diagnosis")
    provisional_diagnosis_reason = data.get("provisional_diagnosis_reason")
    annotation_completed = data.get("annotation_completed", False)
    
    current_app.logger.info(f"Processing patient annotation for patient_id: {patient_id}, adequacy: {adequacy}, completed: {annotation_completed}")
    
    # Check if patient annotation already exists for this user
    patient_annot = PatientAnnotation.query.filter_by(user_id=user_id, patient_id=patient_id).first()
    
    try:
        if patient_annot:
            # Update existing patient annotation
            patient_annot.adequacy = adequacy
            patient_annot.inadequacy_reason = inadequacy_reason
            patient_annot.provisional_diagnosis = provisional_diagnosis
            patient_annot.provisional_diagnosis_reason = provisional_diagnosis_reason
            patient_annot.annotation_completed = annotation_completed
            patient_annot.annotated_at = datetime.utcnow()
            
            action_type = "update"
        else:
            # Create new patient annotation
            patient_annot = PatientAnnotation(
                user_id=user_id,
                patient_id=patient_id,
                adequacy=adequacy,
                inadequacy_reason=inadequacy_reason,
                provisional_diagnosis=provisional_diagnosis,
                provisional_diagnosis_reason=provisional_diagnosis_reason,
                annotation_completed=annotation_completed
            )
            db.session.add(patient_annot)
            action_type = "create"

        db.session.commit()
        
        ActivityLogger.log_workflow_step(
            user_id=user_id,
            user_role=user_role,
            workflow_type="patient_annotation",
            step=f"patient_annotation_{action_type}_success",
            step_data={
                "patient_id": patient_id,
                "annotation_id": patient_annot.id,
                "adequacy": adequacy,
                "provisional_diagnosis": provisional_diagnosis,
                "annotation_completed": annotation_completed
            }
        )
        
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="submit_patient_annotation",
            status="success",
            metadata={"patient_id": patient_id, "annotation_id": patient_annot.id, "action": action_type}
        )
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("DB save failed for patient annotation")
        
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="submit_patient_annotation",
            status="error",
            metadata={"patient_id": patient_id, "reason": "db_save_failed", "error": str(e)}
        )

        return jsonify({"error": "DB save failed", "details": str(e)}), 500

    return jsonify({"message": "ok", "annotation_id": patient_annot.id, "action": action_type}), 201
