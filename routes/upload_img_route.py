# upload_img_bp.py
import os
import uuid
import botocore
from datetime import datetime
from db.models import db, Sample, Patient, PatientAnnotation
from flask import Blueprint, jsonify, request, current_app
from utilities.auth_utility import protected

from utilities.aws_utility import upload_to_s3
from utilities.logging_utility import ActivityLogger

upload_img_bp = Blueprint("upload_img", __name__)


S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "my-test-bucket")



@upload_img_bp.route("/upload_img", methods=["POST"])
@protected
def upload_img(decoded_token):
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_workflow_step(
        user_id=user_id,
        user_role=user_role,
        workflow_type="upload",
        step="start_upload",
        step_data={"endpoint": "upload_img"}
    )
    
    print(decoded_token)
    if decoded_token["role"] != "UPLOADER":
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="upload_access_denied",
            status="error",
            metadata={"reason": "insufficient_permissions", "required_role": "UPLOADER"}
        )
        jsonify({"message": "Unautorized access"}), 400

    # Validate file presence
    if 'image' not in request.files:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="upload_error",
            action_details="no_file_in_request",
            status="error"
        )
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="upload_error",
            action_details="empty_filename",
            status="error"
        )
        return jsonify({"message": "No selected file"}), 400

    # Read optional metadata fields from the form
    user_typed_patient_id = request.form.get("patientId")  # This is the ID entered by pulmonologist
    node_station = request.form.get("lymphNodeStation")
    needle_size = request.form.get("needleSize")
    sample_type = request.form.get("sampleType")
    microscope = request.form.get("microscope")
    magnification = request.form.get("magnification")
    stain = request.form.get("stain")
    camera = request.form.get("camera")
    entered_by = request.form.get("entered_by", None)

    if not user_typed_patient_id:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="upload_error",
            action_details="missing_patient_id",
            status="error"
        )
        return jsonify({"message": "Patient ID is required"}), 400

    # Handle patient creation/lookup
    patient = Patient.query.filter_by(user_typed_id=user_typed_patient_id).first()
    if not patient:
        # Create new patient
        patient = Patient(user_typed_id=user_typed_patient_id)
        try:
            db.session.add(patient)
            db.session.commit()
            ActivityLogger.log_workflow_step(
                user_id=user_id,
                user_role=user_role,
                workflow_type="upload",
                step="patient_created",
                step_data={"patient_id": user_typed_patient_id}
            )
            current_app.logger.info(f"Created new patient with user_typed_id: {user_typed_patient_id}")
        except Exception as e:
            db.session.rollback()
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="upload_error",
                action_details="patient_creation_failed",
                status="error",
                metadata={"patient_id": user_typed_patient_id, "error": str(e)}
            )
            current_app.logger.exception("Failed to create patient")
            return jsonify({"error": "Failed to create patient", "details": str(e)}), 500
    else:
        # If patient exists and has completed annotations, reset completion status
        # because new slides are being added that need to be annotated
        try:
            # Find all patient annotations that are marked as completed for this patient
            completed_patient_annotations = PatientAnnotation.query.filter_by(
                patient_id=patient.patient_id,
                annotation_completed=True
            ).all()
            
            if completed_patient_annotations:
                # Reset completion status for all pathologists who had completed this patient
                for patient_annot in completed_patient_annotations:
                    patient_annot.annotation_completed = False
                    patient_annot.annotated_at = datetime.utcnow()
                
                db.session.commit()
                
                pathologist_count = len(completed_patient_annotations)
                ActivityLogger.log_workflow_step(
                    user_id=user_id,
                    user_role=user_role,
                    workflow_type="upload",
                    step="patient_annotations_reset",
                    step_data={
                        "patient_id": user_typed_patient_id, 
                        "reason": "new_slides_added",
                        "affected_pathologists": pathologist_count
                    }
                )
                current_app.logger.info(f"Reset annotation completion status for {pathologist_count} pathologists for patient {user_typed_patient_id} due to new slide upload")
            else:
                current_app.logger.info(f"Adding new slides to existing patient {user_typed_patient_id} (no completed annotations to reset)")
                
        except Exception as e:
            db.session.rollback()
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="upload_error",
                action_details="patient_annotation_reset_failed",
                status="error",
                metadata={"patient_id": user_typed_patient_id, "error": str(e)}
            )
            current_app.logger.exception("Failed to reset patient annotation completion status")
            # Don't fail the upload, just log the error
            current_app.logger.warning("Continuing with upload despite annotation reset failure")

    # Create job_id (serves as sample id)
    job_id = str(uuid.uuid4())

    # Safely extract extension; default to .jpg if missing
    _, ext = os.path.splitext(file.filename or "")
    if ext == "":
        ext = ".jpg"

    # S3 layout: uploads/<job_id>/original<ext>
    s3_prefix = f"uploads/original/{job_id}/"
    s3_object_key = f"{s3_prefix}original{ext}"

    # Upload file to S3
    try:
        upload_to_s3(S3_BUCKET_NAME,file,s3_object_key)
        ActivityLogger.log_workflow_step(
            user_id=user_id,
            user_role=user_role,
            workflow_type="upload",
            step="s3_upload_success",
            step_data={"job_id": job_id, "s3_key": s3_object_key, "file_size": file.content_length}
        )
    except Exception as e:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="upload_error",
            action_details="s3_upload_failed",
            status="error",
            metadata={"job_id": job_id, "s3_key": s3_object_key, "error": str(e)}
        )
        return jsonify({"message": "S3 upload failed", "details": str(e)}), 500

    # Persist minimal DB row
    sample = Sample(
        job_id=job_id,
        patient_id=patient.patient_id,  # Link to the patient
        s3_object_key = s3_object_key,
        node_station=node_station,
        needle_size=needle_size,
        sample_type=sample_type,
        microscope=microscope,
        magnification=magnification,
        stain=stain,
        camera=camera,
        user_id=decoded_token["user_id"]  # Link to the uploader user
        
    )
    try:
        db.session.add(sample)
        db.session.commit()
        ActivityLogger.log_workflow_step(
            user_id=user_id,
            user_role=user_role,
            workflow_type="upload",
            step="db_save_success",
            step_data={"job_id": job_id, "patient_id": patient.patient_id}
        )
    except Exception as e:
        db.session.rollback()
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="upload_error",
            action_details="db_save_failed",
            status="error",
            metadata={"job_id": job_id, "patient_id": patient.patient_id, "error": str(e)}
        )
        current_app.logger.exception("DB save failed")
        # We uploaded to S3 already — you may want to delete the object on DB failure.
        return jsonify({"error": "DB save failed", "details": str(e)}), 500

    # DO NOT auto-enqueue - let admin manually queue samples
    # This gives better control over the processing pipeline
    
    ActivityLogger.log_workflow_step(
        user_id=user_id,
        user_role=user_role,
        workflow_type="upload",
        step="upload_complete_not_queued",
        step_data={"job_id": job_id, "note": "Sample saved but not queued - manual queueing required"}
    )
    
    return jsonify({
        "status": "success",
        "message": "File uploaded successfully. Sample is ready to be queued for processing.",
        "job_id": job_id,
        "s3_key": s3_object_key
    }), 200


