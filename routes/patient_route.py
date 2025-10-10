import os
from db.models import Patient, Sample, SampleAnnotation, PatientAnnotation, db
from flask import jsonify, Blueprint, request
from utilities.auth_utility import protected
from utilities.aws_utility import get_presigned_url
from utilities.logging_utility import ActivityLogger
from datetime import datetime
import logging
import json

log = logging.getLogger(__name__)
patient_route_bp = Blueprint("patients", __name__)

S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "my-test-bucket")
PUBLIC_HOST = os.environ.get("S3_PUBLIC_HOST", "localhost:4566")

def _rewrite_host(url: str) -> str:
    if "localstack" in url or "127.0.0.1" in url:
        try:
            proto, rest = url.split("://", 1)
            _, after_host = rest.split("/", 1)
            return f"{proto}://{PUBLIC_HOST}/{after_host}"
        except ValueError: 
            return url
    return url

@patient_route_bp.route("/patients", methods=["GET"])
@protected
def get_patients(decoded_token):
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="get_patients_list",
        status="start"
    )
    
    if decoded_token["role"] != "ANNONATOR":
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="patients_access_denied",
            status="error",
            metadata={"reason": "insufficient_permissions", "required_role": "ANNONATOR"}
        )
        return jsonify({"message": "Unauthorized Access"}), 401

    # Get pagination parameters
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 10))

    try:
        # Get patients with their samples, ordered by patient annotation completion status
        # Use DB-side pagination to avoid loading all rows into memory
        patients_with_samples = (
            Patient.query
            .join(Sample, Patient.patient_id == Sample.patient_id)
            .filter(Sample.inference_status == 'completed')
            .distinct()
            .order_by(Patient.created_at.desc())
            .paginate(page=page, per_page=per_page, error_out=False)
        )

        # Build items list with completion status for sorting within the current page
        items_with_status = []
        for patient in patients_with_samples.items:
            # Get user's sample annotations for this patient
            user_sample_annotations = SampleAnnotation.query.filter_by(
                user_id=decoded_token["user_id"],
                patient_id=patient.patient_id
            ).all()
            
            # Get user's patient annotation for this patient
            user_patient_annotation = PatientAnnotation.query.filter_by(
                user_id=decoded_token["user_id"],
                patient_id=patient.patient_id
            ).first()

            annotated_sample_ids = {ann.sample_id for ann in user_sample_annotations}
            
            # Count total samples and annotated samples for this patient
            total_samples = len([s for s in patient.samples if s.inference_status == 'completed'])
            annotated_samples = len(annotated_sample_ids)
            
            # Check completion status based on sample annotations AND patient annotation
            samples_completed = (annotated_samples == total_samples and total_samples > 0)
            patient_annotation_completed = user_patient_annotation and user_patient_annotation.annotation_completed
            is_completed = samples_completed and patient_annotation_completed

            patient_data = {
                "patient_id": patient.patient_id,
                "user_typed_id": patient.user_typed_id,
                "annotation_completed": is_completed,
                "adequacy": user_patient_annotation.adequacy if user_patient_annotation else None,
                "inadequacy_reason": user_patient_annotation.inadequacy_reason if user_patient_annotation else None,
                "provisional_diagnosis": user_patient_annotation.provisional_diagnosis if user_patient_annotation else None,
                "provisional_diagnosis_reason": user_patient_annotation.provisional_diagnosis_reason if user_patient_annotation else None,
                "total_samples": total_samples,
                "annotated_samples": annotated_samples,
                "progress_percentage": (annotated_samples / total_samples * 100) if total_samples > 0 else 0,
                "created_at": patient.created_at,
                # Add cell count summary for completed patients
                "cell_summary": get_patient_cell_summary(patient.patient_id, decoded_token["user_id"]) if is_completed else None
            }
            items_with_status.append((is_completed, patient_data))

        # Sort: In Progress (False) first, then Completed (True), within each group by created_at desc
        items_with_status.sort(key=lambda x: (x[0], -x[1]["created_at"].timestamp()))

        # Extract just the patient data (remove completion flag)
        paginated_items = [item[1] for item in items_with_status]

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_patients_list",
            status="success",
            metadata={"page": page, "per_page": per_page, "total_patients": patients_with_samples.total}
        )

        return jsonify({
            "page": patients_with_samples.page,
            "per_page": patients_with_samples.per_page,
            "total": patients_with_samples.total,
            "pages": patients_with_samples.pages,
            "items": paginated_items
        }), 200
        
    except Exception as e:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_patients_list",
            status="error",
            metadata={"error": str(e)}
        )
        log.exception("Failed to get patients list")
        return jsonify({"error": "Failed to retrieve patients", "details": str(e)}), 500

def get_patient_cell_summary(patient_id, user_id):
    """Helper function to get cell count summary for a patient"""
    try:
        user_sample_annotations = SampleAnnotation.query.filter_by(
            user_id=user_id,
            patient_id=patient_id
        ).all()
        
        total_cell_counts = {}
        for annotation in user_sample_annotations:
            if annotation.cells:
                try:
                    cell_counts = json.loads(annotation.cells) if isinstance(annotation.cells, str) else annotation.cells
                    for cell_type, count in cell_counts.items():
                        total_cell_counts[cell_type] = total_cell_counts.get(cell_type, 0) + count
                except (json.JSONDecodeError, TypeError):
                    continue
        
        total_cells = sum(total_cell_counts.values())
        return {
            "total_cells": total_cells,
            "cell_types": len(total_cell_counts),
            "breakdown": total_cell_counts
        }
    except Exception:
        return None

@patient_route_bp.route("/patient/<patient_id>/samples", methods=["GET"])
@protected
def get_patient_samples(decoded_token, patient_id):
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="get_patient_samples",
        status="start",
        metadata={"patient_id": patient_id}
    )
    
    if decoded_token["role"] != "ANNONATOR":
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="patient_samples_access_denied",
            status="error",
            metadata={"patient_id": patient_id, "reason": "insufficient_permissions", "required_role": "ANNONATOR"}
        )
        return jsonify({"message": "Unauthorized Access"}), 401

    patient = Patient.query.get(patient_id)
    if not patient:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_patient_samples",
            status="error",
            metadata={"patient_id": patient_id, "reason": "patient_not_found"}
        )
        return jsonify({"message": "Patient not found"}), 404

    try:
        # Get user's sample annotations for this patient
        user_annotations = SampleAnnotation.query.filter_by(
            user_id=decoded_token["user_id"],
            patient_id=patient_id
        ).all()
        
        # Get user's patient annotation for this patient
        user_patient_annotation = PatientAnnotation.query.filter_by(
            user_id=decoded_token["user_id"],
            patient_id=patient_id
        ).first()

        annotated_sample_ids = {ann.sample_id for ann in user_annotations}

        samples = []
        for sample in patient.samples:
            if sample.inference_status == 'completed':  # Only processed samples
                samples.append({
                    "job_id": sample.job_id,
                    "original_filename": sample.original_filename,
                    "created_at": sample.created_at,
                    "annotated": sample.job_id in annotated_sample_ids,
                    "annotation_status": sample.annotation_status
                })

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_patient_samples",
            status="success",
            metadata={"patient_id": patient_id, "samples_count": len(samples)}
        )

        # Check if user has completed patient annotation
        is_patient_completed = user_patient_annotation and user_patient_annotation.annotation_completed

        return jsonify({
            "patient_id": patient_id,
            "user_typed_id": patient.user_typed_id,
            "samples": samples,
            "annotation_completed": is_patient_completed
        }), 200
        
    except Exception as e:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_patient_samples",
            status="error",
            metadata={"patient_id": patient_id, "error": str(e)}
        )
        log.exception("Failed to get patient samples")
        return jsonify({"error": "Failed to retrieve patient samples", "details": str(e)}), 500

@patient_route_bp.route("/patient/<patient_id>/next-slide", methods=["GET"])
@protected
def get_next_slide(decoded_token, patient_id):
    """Get the next unannotated slide for a patient"""
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="get_next_slide",
        status="start",
        metadata={"patient_id": patient_id}
    )
    
    if decoded_token["role"] != "ANNONATOR":
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="next_slide_access_denied",
            status="error",
            metadata={"patient_id": patient_id, "reason": "insufficient_permissions", "required_role": "ANNONATOR"}
        )
        return jsonify({"message": "Unauthorized Access"}), 401

    patient = Patient.query.get(patient_id)
    if not patient:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_next_slide",
            status="error",
            metadata={"patient_id": patient_id, "reason": "patient_not_found"}
        )
        return jsonify({"message": "Patient not found"}), 404

    try:
        # Get user's annotations for this patient
        user_annotations = SampleAnnotation.query.filter_by(
            user_id=decoded_token["user_id"],
            patient_id=patient_id
        ).all()

        annotated_sample_ids = {ann.sample_id for ann in user_annotations}

        # Find the next unannotated sample
        next_sample = None
        for sample in patient.samples:
            if sample.inference_status == 'completed' and sample.job_id not in annotated_sample_ids:
                next_sample = sample
                break

        if not next_sample:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="get_next_slide",
                status="warning",
                metadata={"patient_id": patient_id, "reason": "no_more_slides"}
            )
            return jsonify({"message": "No more slides to annotate"}), 404

        # Generate presigned URLs
        try:
            csv_url = get_presigned_url(S3_BUCKET_NAME, next_sample.s3_inference_key, expires=3000)
            image_url = get_presigned_url(S3_BUCKET_NAME, next_sample.s3_object_key, expires=3000)

            image_url = _rewrite_host(image_url)
            csv_url = _rewrite_host(csv_url)

        except Exception as e:
            log.exception("Failed to generate presigned URLs for sample %s: %s", next_sample.job_id, e)
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="get_next_slide",
                status="error",
                metadata={"patient_id": patient_id, "sample_id": next_sample.job_id, "error": str(e)}
            )
            return jsonify({"message": "Failed to generate download URLs"}), 500

        ActivityLogger.log_workflow_step(
            user_id=user_id,
            user_role=user_role,
            workflow_type="annotation",
            step="slide_accessed",
            step_data={"patient_id": patient_id, "sample_id": next_sample.job_id}
        )

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_next_slide",
            status="success",
            metadata={"patient_id": patient_id, "sample_id": next_sample.job_id}
        )

        return jsonify({
            "sample": {
                "job_id": next_sample.job_id,
                "original_filename": next_sample.original_filename,
                "s3_object_key": next_sample.s3_object_key,
                "created_at": next_sample.created_at
            },
            "patient_id": patient_id,
            "user_typed_id": patient.user_typed_id,
            "image": {"url": image_url},
            "csv": {"url": csv_url},
            "expiry_seconds": 3000
        }), 200
        
    except Exception as e:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_next_slide",
            status="error",
            metadata={"patient_id": patient_id, "error": str(e)}
        )
        log.exception("Failed to get next slide")
        return jsonify({"error": "Failed to retrieve next slide", "details": str(e)}), 500

@patient_route_bp.route("/patient/<patient_id>/annotation-summary", methods=["GET"])
@protected
def get_patient_annotation_summary(decoded_token, patient_id):
    """Get detailed annotation summary for a patient with cell counts per slide"""
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="get_annotation_summary",
        status="start",
        metadata={"patient_id": patient_id}
    )
    
    if decoded_token["role"] != "ANNONATOR":
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="annotation_summary_access_denied",
            status="error",
            metadata={"patient_id": patient_id, "reason": "insufficient_permissions", "required_role": "ANNONATOR"}
        )
        return jsonify({"message": "Unauthorized Access"}), 401

    patient = Patient.query.get(patient_id)
    if not patient:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_annotation_summary",
            status="error",
            metadata={"patient_id": patient_id, "reason": "patient_not_found"}
        )
        return jsonify({"message": "Patient not found"}), 404

    try:
        # Get user's sample annotations for this patient, ordered by annotation time
        user_annotations = SampleAnnotation.query.filter_by(
            user_id=decoded_token["user_id"],
            patient_id=patient_id
        ).order_by(SampleAnnotation.annotated_at.asc()).all()
        
        # Get user's patient annotation for this patient
        user_patient_annotation = PatientAnnotation.query.filter_by(
            user_id=decoded_token["user_id"],
            patient_id=patient_id
        ).first()

        slides_summary = []
        total_cell_counts = {}

        for annotation in user_annotations:
            sample = Sample.query.get(annotation.sample_id)
            if not sample:
                continue

            # Parse cell counts from JSON
            cell_counts = {}
            if annotation.cells:
                try:
                    cell_counts = json.loads(annotation.cells) if isinstance(annotation.cells, str) else annotation.cells
                except (json.JSONDecodeError, TypeError):
                    cell_counts = {}

            slides_summary.append({
                "slide_id": sample.job_id,
                "slide_name": sample.original_filename,
                "cell_counts": cell_counts,
                "image_quality": annotation.image_quality,
                "annotated_at": annotation.annotated_at
            })

            # Add to total counts
            for cell_type, count in cell_counts.items():
                total_cell_counts[cell_type] = total_cell_counts.get(cell_type, 0) + count

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_annotation_summary",
            status="success",
            metadata={"patient_id": patient_id, "slides_count": len(slides_summary), "total_cells": sum(total_cell_counts.values())}
        )

        # Check if user has completed patient annotation
        is_patient_completed = user_patient_annotation and user_patient_annotation.annotation_completed

        return jsonify({
            "patient_id": patient_id,
            "user_typed_id": patient.user_typed_id,
            "slides_summary": slides_summary,
            "total_cell_counts": total_cell_counts,
            "total_slides": len(slides_summary),
            "annotation_completed": is_patient_completed
        }), 200
        
    except Exception as e:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_annotation_summary",
            status="error",
            metadata={"patient_id": patient_id, "error": str(e)}
        )
        log.exception("Failed to get patient annotation summary")
        return jsonify({"error": "Failed to retrieve annotation summary", "details": str(e)}), 500

@patient_route_bp.route("/patient/<patient_id>/download-report", methods=["GET"])
@protected
def download_patient_report(decoded_token, patient_id):
    """Generate and download detailed patient annotation report as CSV"""
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="download_patient_report",
        status="start",
        metadata={"patient_id": patient_id}
    )
    
    if decoded_token["role"] != "ANNONATOR":
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="download_report_access_denied",
            status="error",
            metadata={"patient_id": patient_id, "reason": "insufficient_permissions", "required_role": "ANNONATOR"}
        )
        return jsonify({"message": "Unauthorized Access"}), 401

    patient = Patient.query.get(patient_id)
    if not patient:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="download_patient_report",
            status="error",
            metadata={"patient_id": patient_id, "reason": "patient_not_found"}
        )
        return jsonify({"message": "Patient not found"}), 404

    try:
        # Get user's annotations for this patient
        user_annotations = SampleAnnotation.query.filter_by(
            user_id=decoded_token["user_id"],
            patient_id=patient_id
        ).all()

        # Prepare CSV data
        csv_rows = []
        total_cell_counts = {}

        # Header row
        csv_rows.append([
            "Patient ID", "Slide Name", "Cell Type", "Cell Count", 
            "Image Quality", "Annotation Date"
        ])

        for annotation in user_annotations:
            sample = Sample.query.get(annotation.sample_id)
            if not sample:
                continue

            # Parse cell counts
            cell_counts = {}
            if annotation.cells:
                try:
                    cell_counts = json.loads(annotation.cells) if isinstance(annotation.cells, str) else annotation.cells
                except (json.JSONDecodeError, TypeError):
                    cell_counts = {}

            # Add rows for this slide
            if cell_counts:
                # Add row for each cell type in this slide
                for cell_type, count in cell_counts.items():
                    csv_rows.append([
                        patient.user_typed_id,
                        sample.original_filename or sample.job_id,
                        cell_type,
                        count,
                        "Good" if annotation.image_quality else "Poor",
                        annotation.annotated_at.strftime("%Y-%m-%d %H:%M:%S") if annotation.annotated_at else ""
                    ])
                    
                    # Add to total counts
                    total_cell_counts[cell_type] = total_cell_counts.get(cell_type, 0) + count
            else:
                # Add row for slides with no cell counts (usually poor quality slides)
                csv_rows.append([
                    patient.user_typed_id,
                    sample.original_filename or sample.job_id,
                    "No cells annotated" if not annotation.image_quality else "No cells found",
                    0,
                    "Poor" if not annotation.image_quality else "Good",
                    annotation.annotated_at.strftime("%Y-%m-%d %H:%M:%S") if annotation.annotated_at else ""
                ])

        # Calculate slide statistics
        total_slides = len(user_annotations)
        good_quality_slides = sum(1 for ann in user_annotations if ann.image_quality)
        poor_quality_slides = total_slides - good_quality_slides
        slides_with_cells = sum(1 for ann in user_annotations if ann.cells and ann.cells != '{}' and ann.cells)
        
        # Add summary rows
        csv_rows.append([])  # Empty row
        csv_rows.append(["PATIENT SUMMARY"])
        csv_rows.append(["Patient ID", patient.user_typed_id, "", "", "", ""])
        csv_rows.append(["Total Slides", total_slides, "", "", "", ""])
        csv_rows.append(["Good Quality Slides", good_quality_slides, "", "", "", ""])
        csv_rows.append(["Poor Quality Slides", poor_quality_slides, "", "", "", ""])
        csv_rows.append(["Slides with Cell Annotations", slides_with_cells, "", "", "", ""])
        csv_rows.append([])  # Empty row
        
        csv_rows.append(["SUMMARY - Total Cell Counts"])
        csv_rows.append(["Patient ID", "Cell Type", "Total Count", "", "", ""])
        
        for cell_type, total_count in total_cell_counts.items():
            csv_rows.append([
                patient.user_typed_id,
                cell_type,
                total_count,
                "", "", ""
            ])
        
        # Add overall totals
        if total_cell_counts:
            total_cells = sum(total_cell_counts.values())
            csv_rows.append([])
            csv_rows.append(["OVERALL TOTAL", "", total_cells, "", "", ""])

        # Convert to CSV string
        import io
        import csv as csv_module
        
        output = io.StringIO()
        writer = csv_module.writer(output)
        writer.writerows(csv_rows)
        csv_content = output.getvalue()
        output.close()

        # Create response with CSV
        from flask import Response
        
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="download_patient_report",
            status="success",
            metadata={"patient_id": patient_id, "total_slides": total_slides, "total_cells": sum(total_cell_counts.values())}
        )
        
        response = Response(
            csv_content,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=patient_{patient.user_typed_id}_annotation_report.csv'
            }
        )
        
        return response
        
    except Exception as e:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="download_patient_report",
            status="error",
            metadata={"patient_id": patient_id, "error": str(e)}
        )
        log.exception("Failed to generate patient report")
        return jsonify({"error": "Failed to generate report", "details": str(e)}), 500

@patient_route_bp.route("/patient/<patient_id>/complete", methods=["POST"])
@protected
def complete_patient_annotation(decoded_token, patient_id):
    """Mark patient annotation as complete with adequacy and diagnosis info"""
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="complete_patient_annotation",
        status="start",
        metadata={"patient_id": patient_id}
    )
    
    if decoded_token["role"] != "ANNONATOR":
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="complete_annotation_access_denied",
            status="error",
            metadata={"patient_id": patient_id, "reason": "insufficient_permissions", "required_role": "ANNONATOR"}
        )
        return jsonify({"message": "Unauthorized Access"}), 401

    patient = Patient.query.get(patient_id)
    if not patient:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="complete_patient_annotation",
            status="error",
            metadata={"patient_id": patient_id, "reason": "patient_not_found"}
        )
        return jsonify({"message": "Patient not found"}), 404

    data = request.get_json()
    
    # Validate that all slides for this patient are actually annotated by this user
    user_annotations = SampleAnnotation.query.filter_by(
        user_id=decoded_token["user_id"],
        patient_id=patient_id
    ).all()
    
    annotated_sample_ids = {ann.sample_id for ann in user_annotations}
    total_samples = len([s for s in patient.samples if s.inference_status == 'completed'])
    annotated_samples = len(annotated_sample_ids)
    
    if annotated_samples < total_samples:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="complete_patient_annotation",
            status="error",
            metadata={"patient_id": patient_id, "annotated_samples": annotated_samples, "total_samples": total_samples, "reason": "incomplete_annotation"}
        )
        return jsonify({
            "message": f"Cannot complete annotation. {total_samples - annotated_samples} slides are still unannotated.",
            "annotated_samples": annotated_samples,
            "total_samples": total_samples
        }), 400
    
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val != 0
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes", "y")
        return False

    try:
        # Create or update patient annotation with final assessment
        adequacy = to_bool(data.get("adequacy"))
        inadequacy_reason = data.get("inadequacy_reason") if not adequacy else None
        provisional_diagnosis = to_bool(data.get("provisional_diagnosis"))
        provisional_diagnosis_reason = data.get("provisional_diagnosis_reason") if provisional_diagnosis else None
        
        # Check if patient annotation already exists for this user
        patient_annot = PatientAnnotation.query.filter_by(user_id=user_id, patient_id=patient_id).first()
        
        if patient_annot:
            # Update existing patient annotation
            patient_annot.adequacy = adequacy
            patient_annot.inadequacy_reason = inadequacy_reason
            patient_annot.provisional_diagnosis = provisional_diagnosis
            patient_annot.provisional_diagnosis_reason = provisional_diagnosis_reason
            patient_annot.annotation_completed = True
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
                annotation_completed=True
            )
            db.session.add(patient_annot)
            action_type = "create"

        db.session.commit()
        
        ActivityLogger.log_workflow_step(
            user_id=user_id,
            user_role=user_role,
            workflow_type="annotation",
            step="patient_annotation_completed",
            step_data={
                "patient_id": patient_id,
                "annotation_id": patient_annot.id,
                "adequacy": adequacy,
                "provisional_diagnosis": provisional_diagnosis,
                "total_slides": total_samples,
                "action": action_type
            }
        )
        
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="complete_patient_annotation",
            status="success",
            metadata={
                "patient_id": patient_id, 
                "annotation_id": patient_annot.id,
                "adequacy": adequacy, 
                "provisional_diagnosis": provisional_diagnosis,
                "action": action_type
            }
        )
        
        return jsonify({"message": "Patient annotation completed successfully", "action": action_type}), 200
        
    except Exception as e:
        db.session.rollback()
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="complete_patient_annotation",
            status="error",
            metadata={"patient_id": patient_id, "error": str(e)}
        )
        log.exception("Failed to complete patient annotation")
        return jsonify({"error": "Failed to complete annotation", "details": str(e)}), 500
