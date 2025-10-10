import os
from db.models import User, SampleAnnotation, Sample
from flask import request,jsonify,Blueprint,current_app
from utilities.auth_utility import protected
from utilities.aws_utility import get_presigned_url
from utilities.logging_utility import ActivityLogger
import logging
import json
log = logging.getLogger(__name__)
samples_route_bp = Blueprint("samples",__name__)

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
    


@samples_route_bp.route("/samples", methods=["GET"])
@protected
def get_user_samples(decoded_token):
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="get_user_samples",
        status="start"
    )
    
    if decoded_token["role"] != "ANNONATOR":
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="samples_access_denied",
            status="error",
            metadata={"reason": "insufficient_permissions", "required_role": "ANNONATOR"}
        )
        return jsonify({"message": "Unrestricted Access"}), 401
    
    user = User.query.get(decoded_token["user_id"])
    if not user:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_user_samples",
            status="error",
            metadata={"reason": "user_not_found"}
        )
        return jsonify({"message": "User not found"}), 404

    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 10))

    try:
        pagination  = (
        Sample.query.filter_by(inference_status='completed')
        .order_by(Sample.annotation_status.asc(),Sample.created_at.desc())  # False (0) first, True (1) later
        .paginate(page=page, per_page=per_page, error_out=False))

        annotations = SampleAnnotation.query.filter_by(user_id=decoded_token["user_id"]).all()
        
        annotated_by_samples = {}
        annotated_ids = []
        for ann in annotations: 
            annotated_ids.append(ann.sample_id)
            annotated_by_samples[ann.sample_id] = ann

        items = []
        for sample in pagination.items:
            sample_annotation = annotated_by_samples.get(sample.job_id)
            items.append({
                "job_id": sample.job_id,
                "common_annotation_status": sample.annotation_status,
                "annotated": sample.job_id in annotated_ids,
                "annotated_at": sample_annotation.annotated_at if sample_annotation else None,
                "image_quality": sample_annotation.image_quality if sample_annotation else None,
                "cells": json.loads(sample_annotation.cells) if sample_annotation and sample_annotation.cells else None,
            })

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_user_samples",
            status="success",
            metadata={"page": page, "per_page": per_page, "total_samples": pagination.total, "annotated_count": len(annotated_ids)}
        )

        return jsonify({
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total": pagination.total,
            "pages": pagination.pages,
            "items": items
        }), 200
        
    except Exception as e:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_user_samples",
            status="error",
            metadata={"error": str(e)}
        )
        log.exception("Failed to get user samples")
        return jsonify({"error": "Failed to retrieve samples", "details": str(e)}), 500




@samples_route_bp.route("/sample/<job_id>", methods=["GET"])
@protected
def get_user_sample(decoded_token, job_id):
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="get_user_sample",
        status="start",
        metadata={"job_id": job_id}
    )
    
    # auth: ensure user has right role
    if decoded_token["role"] != "ANNONATOR":  # <-- check spelling
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="auth_event",
            action_details="sample_access_denied",
            status="error",
            metadata={"job_id": job_id, "reason": "insufficient_permissions", "required_role": "ANNONATOR"}
        )
        return jsonify({"message": "Forbidden"}), 403

    sample = Sample.query.get(job_id)
    if not sample:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_user_sample",
            status="error",
            metadata={"job_id": job_id, "reason": "sample_not_found"}
        )
        return jsonify({"message": "Sample not found"}), 404

    # validate keys exist
    if not sample.s3_inference_key or not sample.s3_object_key:
        log.warning("Sample %s missing s3 keys", job_id)
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_user_sample",
            status="error",
            metadata={"job_id": job_id, "reason": "missing_s3_keys"}
        )
        return jsonify({"message": "Artifacts not available"}), 404

    try:
        csv_url = get_presigned_url(
            S3_BUCKET_NAME,
            sample.s3_inference_key,
            expires=3000,
        )

        image_url = get_presigned_url(
            S3_BUCKET_NAME,
            sample.s3_object_key,
            expires=3000,
        )

        image_url = _rewrite_host(image_url)
        csv_url = _rewrite_host(csv_url)    

        ActivityLogger.log_workflow_step(
            user_id=user_id,
            user_role=user_role,
            workflow_type="annotation",
            step="sample_accessed",
            step_data={"job_id": job_id}
        )

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_user_sample",
            status="success",
            metadata={"job_id": job_id}
        )

    except Exception as e:
        log.exception("Failed to generate presigned URLs for sample %s: %s", job_id, e)
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="get_user_sample",
            status="error",
            metadata={"job_id": job_id, "error": str(e)}
        )
        return jsonify({"message": "Failed to generate download URLs"}), 500

    return jsonify({
        "image": {"url": image_url},
        "csv":   {"url": csv_url},
        "expiry_seconds": 300
    })
