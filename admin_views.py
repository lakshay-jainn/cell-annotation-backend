# admin_views.py - Comprehensive admin views for all models
from flask_admin import BaseView, expose, AdminIndexView
from flask_admin.contrib.sqla import ModelView
from flask import redirect, url_for, request, flash, render_template, jsonify
from flask_login import current_user, login_required
from db.models import (
    db, Sample, User, Patient, SampleAnnotation, PatientAnnotation, 
    UserRole, UserActivityLog
)


class AuthModelView(ModelView):
    def is_accessible(self):
        return current_user.is_authenticated and current_user.role == UserRole.ADMIN

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for("admin_login"))


class AuthIndexView(AdminIndexView):
    def is_accessible(self):
        return current_user.is_authenticated and current_user.role == UserRole.ADMIN

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for("admin_login"))


def hash_password(password):
    from utilities.auth_utility import hash_password as hash_pw
    return hash_pw(password)


class UserAdmin(AuthModelView):
    """Admin view for User management with enhanced functionality."""
    column_list = ("id", "name", "email", "role", "hospital", "location", "created_samples", "sample_annotations", "patient_annotations")
    column_labels = {
        'id': 'User ID',
        'created_samples': '# Samples Uploaded',
        'sample_annotations': '# Sample Annotations',
        'patient_annotations': '# Patient Annotations'
    }
    column_searchable_list = ("name", "email", "hospital", "location")
    column_filters = ("role", "hospital", "location")
    column_sortable_list = ("name", "email", "role", "hospital", "location")
    form_excluded_columns = ("password", "sample_annotations", "patient_annotations", "samples", "activity_logs")
    
    # Add custom columns for counts
    column_formatters = {
        'created_samples': lambda v, c, m, p: len(m.samples) if m.samples else 0,
        'sample_annotations': lambda v, c, m, p: len(m.sample_annotations) if m.sample_annotations else 0,
        'patient_annotations': lambda v, c, m, p: len(m.patient_annotations) if m.patient_annotations else 0,
    }

    def scaffold_form(self):
        form_class = super().scaffold_form()
        from wtforms import PasswordField
        form_class.new_password = PasswordField("New Password")
        return form_class

    def on_model_change(self, form, model, is_created):
        if form.new_password.data:
            model.password = hash_password(form.new_password.data)
        super().on_model_change(form, model, is_created)


class PatientAdmin(AuthModelView):
    """Admin view for Patient management."""
    column_list = ("patient_id", "user_typed_id", "total_samples", "total_sample_annotations", "total_patient_annotations", "pulmonologist_report_s3_key" ,"pulmonologist_id","created_at", "updated_at")
    column_labels = {
        'patient_id': 'Patient ID',
        'user_typed_id': 'User Typed ID',
        'total_samples': '# Samples',
        'pulmonologist_id': 'Pulmonologist ID',
        'pulmonologist_report_s3_key': 'S3 Report Key',
        'total_sample_annotations': '# Sample Annotations',
        'total_patient_annotations': '# Patient Annotations'
    }
    column_searchable_list = ("user_typed_id", "patient_id")
    column_filters = ("created_at", "updated_at")
    column_sortable_list = ("user_typed_id", "created_at", "updated_at")
    form_excluded_columns = ("samples", "sample_annotations", "patient_annotations")
    
    # Add custom columns for counts
    column_formatters = {
        'total_samples': lambda v, c, m, p: len(m.samples) if m.samples else 0,
        'total_sample_annotations': lambda v, c, m, p: len(m.sample_annotations) if m.sample_annotations else 0,
        'total_patient_annotations': lambda v, c, m, p: len(m.patient_annotations) if m.patient_annotations else 0,
    }


class SampleAdmin(AuthModelView):
    """Admin view for Sample management."""
    column_list = ("job_id", "patient_id", "uploader_name", "inference_status", "annotation_count", "created_at", "updated_at", 
                   "node_station", "needle_size", "sample_type", "microscope", "magnification", "stain", "camera")
    column_labels = {
        'job_id': 'Sample ID',
        'patient_id': 'Patient ID',
        'uploader_name': 'Uploaded By',
        'annotation_count': '# Annotations',
        'inference_status': 'Processing Status'
    }
    column_searchable_list = ("job_id", "original_filename", "patient_id")
    column_filters = ("inference_status", "node_station", "needle_size", "sample_type", "microscope", "stain", "created_at")
    column_sortable_list = ("created_at", "updated_at", "inference_status")
    form_excluded_columns = ("sample_annotations", "patient", "uploader")
    
    # Add custom columns
    column_formatters = {
        'uploader_name': lambda v, c, m, p: m.uploader.name if m.uploader else 'Unknown',
        'annotation_count': lambda v, c, m, p: len(m.sample_annotations) if m.sample_annotations else 0,
    }


class SampleAnnotationAdmin(AuthModelView):
    """Admin view for Sample Annotations."""
    column_list = ("id", "user_name", "sample_id", "patient_id", "status", "image_quality", "cell_count", "annotated_at", "s3_annotation_key")
    column_labels = {
        'id': 'Annotation ID',
        'user_name': 'Annotated By',
        'sample_id': 'Sample ID',
        'patient_id': 'Patient ID',
        'cell_count': '# Cells Annotated',
        's3_annotation_key': 'S3 CSV Key'
    }
    column_searchable_list = ("id", "sample_id", "patient_id", "user_id")
    column_filters = ("status", "image_quality", "annotated_at")
    column_sortable_list = ("annotated_at", "status")
    form_excluded_columns = ("user", "sample", "patient")
    
    # Add custom columns
    column_formatters = {
        'user_name': lambda v, c, m, p: m.user.name if m.user else 'Unknown',
        'cell_count': lambda v, c, m, p: len(eval(m.cells)) if m.cells and m.cells != 'null' else 0,
        's3_annotation_key': lambda v, c, m, p: m.s3_annotation_key.split('/')[-1] if m.s3_annotation_key else 'None'
    }


class PatientAnnotationAdmin(AuthModelView):
    """Admin view for Patient Annotations."""
    column_list = ("id", "user_name", "patient_user_typed_id", "adequacy", "provisional_diagnosis", 
                   "annotation_completed", "annotated_at", "status")
    column_labels = {
        'id': 'Annotation ID',
        'user_name': 'Annotated By',
        'patient_user_typed_id': 'Patient ID',
        'annotation_completed': 'Completed'
    }
    column_searchable_list = ("id", "patient_id", "user_id")
    column_filters = ("adequacy", "provisional_diagnosis", "annotation_completed", "status", "annotated_at")
    column_sortable_list = ("annotated_at", "annotation_completed")
    form_excluded_columns = ("user", "patient")
    
    # Add custom columns
    column_formatters = {
        'user_name': lambda v, c, m, p: m.user.name if m.user else 'Unknown',
        'patient_user_typed_id': lambda v, c, m, p: m.patient.user_typed_id if m.patient else 'Unknown',
        'adequacy': lambda v, c, m, p: 'Yes' if m.adequacy else 'No' if m.adequacy is False else 'Not Set',
        'provisional_diagnosis': lambda v, c, m, p: 'Yes' if m.provisional_diagnosis else 'No' if m.provisional_diagnosis is False else 'Not Set',
        'annotation_completed': lambda v, c, m, p: '✓' if m.annotation_completed else '✗'
    }


class UserActivityLogAdmin(AuthModelView):
    """Admin view for User Activity Logs with enhanced filtering."""
    column_list = ("id", "user_name", "user_role", "action_type", "action_details", "status", "ip_address", "created_at")
    column_filters = ("user_id", "user_role", "action_type", "status", "created_at")
    column_searchable_list = ("user_id", "action_details", "ip_address")
    column_sortable_list = ("created_at", "user_id", "user_role", "action_type", "status")
    column_default_sort = ("created_at", True)  # Sort by created_at descending (most recent first)

    # Display activity_metadata as a formatted JSON field
    column_formatters = {
        'activity_metadata': lambda v, c, m, p: str(m.activity_metadata) if m.activity_metadata else "",
        'user_name': lambda v, c, m, p: m.user.name if m.user else 'Unknown User'
    }

    # Make the list view more readable
    column_labels = {
        'user_id': 'User ID',
        'user_name': 'User Name',
        'user_role': 'Role',
        'action_type': 'Action Type',
        'action_details': 'Action Details',
        'activity_metadata': 'Metadata',
        'ip_address': 'IP Address',
        'created_at': 'Timestamp'
    }

    # Exclude metadata from form to avoid complex JSON editing
    form_excluded_columns = ("activity_metadata", "user")

    # Set page size for better performance with potentially many logs
    page_size = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_export_list = self.column_list





class ModelProcessingView(BaseView):
    """Simplified admin view for manual model processing with API calls."""
    
    def is_accessible(self):
        return current_user.is_authenticated and current_user.role == UserRole.ADMIN

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for("admin_login"))
    
    def _get_processing_info(self):
        """Get basic information about processing status."""
        try:
            pending_count = Sample.query.filter_by(inference_status='pending').count()
            completed_count = Sample.query.filter_by(inference_status='completed').count()
            processing_count = Sample.query.filter_by(inference_status='processing').count()
            failed_count = Sample.query.filter_by(inference_status='failed').count()
            total_count = Sample.query.count()
            
            return {
                'pending': pending_count,
                'processing': processing_count,
                'completed': completed_count,
                'failed': failed_count,
                'total': total_count
            }
        except Exception as e:
            print(f"Error getting processing info: {e}")
            return {
                'pending': 0,
                'completed': 0,
                'total': 0
            }
    
    @expose('/')
    @login_required
    def index(self):
        if not (current_user.is_authenticated and current_user.role == UserRole.ADMIN):
            return redirect(url_for("admin_login"))
        
        # Get samples by status with user information
        pending_samples = db.session.query(Sample).options(
            db.joinedload(Sample.uploader)
        ).filter(
            Sample.inference_status == 'pending'
        ).all()
        
        processing_samples = db.session.query(Sample).options(
            db.joinedload(Sample.uploader)
        ).filter(
            Sample.inference_status == 'processing'
        ).all()
        
        # Get completed samples (last 10) with user information
        completed_samples = db.session.query(Sample).options(
            db.joinedload(Sample.uploader)
        ).filter(
            Sample.inference_status == 'completed'
        ).order_by(Sample.updated_at.desc()).limit(10).all()
        
        # Check queue status for pending samples on initial load
        from utilities.enqueue_utility import is_job_in_queue
        for sample in pending_samples:
            try:
                sample.in_queue = is_job_in_queue(sample.job_id)
            except Exception as e:
                print(f"Error checking queue for {sample.job_id}: {e}")
                sample.in_queue = False
        
        # Set display status and generate presigned URLs for all samples
        for sample in pending_samples + processing_samples + completed_samples:
            sample.display_status = sample.inference_status
            
            # Generate presigned URL for image download
            if sample.s3_object_key:
                try:
                    from utilities.aws_utility import get_presigned_url
                    import os
                    S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "my-test-bucket")
                    sample.image_download_url = get_presigned_url(
                        S3_BUCKET_NAME, 
                        sample.s3_object_key, 
                        expires=3600  # 1 hour expiry
                    )
                except Exception as e:
                    print(f"Error generating presigned URL for {sample.job_id}: {e}")
                    sample.image_download_url = None
            else:
                sample.image_download_url = None
        
        # Combine pending and processing for the template
        all_pending_samples = pending_samples + processing_samples
        
        # Get processing info
        processing_info = self._get_processing_info()
        
        # Get SQS queue status
        queue_info = {'messages_available': 0, 'messages_in_flight': 0, 'total_messages': 0}
        try:
            from sqs_processor import get_queue_status
            queue_status = get_queue_status()
            if queue_status['success']:
                queue_info = queue_status
        except Exception as e:
            print(f"Error getting queue status: {e}")
        
        return render_template(
            'model_processing/index.html',
            pending_samples=all_pending_samples,
            processing_samples=processing_samples,
            completed_samples=completed_samples,
            background_tasks=[],  # Not used anymore
            celery_available=False,  # Simplified approach
            processing_info=processing_info,
            queue_info=queue_info
        )
    
    @expose('/process-sample', methods=['POST'])
    @login_required
    def process_sample(self):
        if not (current_user.is_authenticated and current_user.role == UserRole.ADMIN):
            return redirect(url_for("admin_login"))

        # Handle both form data and JSON request body
        if request.headers.get('Content-Type') == 'application/json':
            sample_id = request.get_json().get('sample_id')
        else:
            sample_id = request.form.get('sample_id')
            
        if not sample_id:
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify({'status': 'error', 'message': 'No sample ID provided'}), 400
            else:
                flash('No sample ID provided', 'error')
                return redirect(url_for('modelprocessing.index'))

        # Find sample by job_id
        sample = db.session.query(Sample).filter_by(job_id=sample_id).first()
        if not sample:
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify({'status': 'error', 'message': f'Sample {sample_id} not found'}), 404
            else:
                flash(f'Sample {sample_id} not found', 'error')
                return redirect(url_for('modelprocessing.index'))

        # Immediately update sample to processing state
        from datetime import datetime
        sample.inference_status = 'processing'
        sample.updated_at = datetime.utcnow()
        db.session.commit()

        # Process sample in background using subprocess
        import subprocess
        import sys
        import os

        try:
            # Start background process
            script_path = os.path.join(os.path.dirname(__file__), 'background_processor.py')
            process = subprocess.Popen([
                sys.executable, script_path, 'sample', sample_id
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Update status to processing immediately
            sample.inference_status = 'processing'
            sample.updated_at = datetime.utcnow()
            db.session.commit()

            flash(f'Sample {sample.job_id} processing started in background (PID: {process.pid})', 'success')
            
            # Check if request is AJAX
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify({
                    'status': 'success',
                    'message': f'Sample {sample.job_id} processing started in background (PID: {process.pid})'
                })

        except Exception as e:
            print(f"Background processing failed for {sample_id}: {e}")
            sample.inference_status = 'failed'
            db.session.commit()
            
            # Check if request is AJAX
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify({
                    'status': 'error',
                    'message': f'Sample {sample.job_id} background processing failed: {str(e)}'
                }), 500
            else:
                flash(f'Sample {sample.job_id} background processing failed: {str(e)}', 'error')

        # For non-AJAX requests, redirect as before
        return redirect(url_for('modelprocessing.index'))
        

    @expose('/process-sqs', methods=['POST'])
    @login_required
    def process_sqs(self):
        """Process messages from SQS queue."""
        if not (current_user.is_authenticated and current_user.role == UserRole.ADMIN):
            return redirect(url_for("admin_login"))

        # Process SQS messages in background using subprocess
        import subprocess
        import sys
        import os

        max_messages = 1  # Process one message at a time to avoid blocking

        try:
            # Start background process
            script_path = os.path.join(os.path.dirname(__file__), 'background_processor.py')
            process = subprocess.Popen([
                sys.executable, script_path, 'sqs', str(max_messages)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Check if request is AJAX
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify({
                    'status': 'success',
                    'message': f'SQS message processing started in background (PID: {process.pid})'
                })
            else:
                flash(f'SQS message processing started in background (PID: {process.pid})', 'success')
                return redirect(url_for('modelprocessing.index'))

        except Exception as e:
            print(f"SQS background processing failed: {e}")
            
            # Check if request is AJAX
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify({
                    'status': 'error',
                    'message': 'SQS message background processing failed. Check logs for details.'
                }), 500
            else:
                flash('SQS message background processing failed. Check logs for details.', 'error')
                return redirect(url_for('modelprocessing.index'))    

    @expose('/status-api', methods=['GET'])
    @login_required
    def status_api(self):
        """AJAX endpoint for real-time status updates without full page reload."""
        if not (current_user.is_authenticated and current_user.role == UserRole.ADMIN):
            return jsonify({'error': 'Unauthorized'}), 401

        try:
            # Get processing info
            processing_info = self._get_processing_info()
            
            # Get SQS queue status

            
            # Get samples by status with complete data (same as main page)
            pending_samples = db.session.query(Sample).options(
                db.joinedload(Sample.uploader)
            ).filter(
                Sample.inference_status == 'pending'
            ).all()
            
            processing_samples = db.session.query(Sample).options(
                db.joinedload(Sample.uploader)
            ).filter(
                Sample.inference_status == 'processing'
            ).all()
            
            completed_samples = db.session.query(Sample).options(
                db.joinedload(Sample.uploader)
            ).filter(
                Sample.inference_status == 'completed'
            ).order_by(Sample.updated_at.desc()).limit(10).all()

            # Check queue status for pending samples
            from utilities.enqueue_utility import is_job_in_queue
            queue_status = {}
            for sample in pending_samples:
                try:
                    queue_status[sample.job_id] = is_job_in_queue(sample.job_id)
                except Exception as e:
                    print(f"Error checking queue for {sample.job_id}: {e}")
                    queue_status[sample.job_id] = False

            # Convert to dictionaries for JSON response with complete data
            def sample_to_dict(sample):
                sample_dict = {
                    'job_id': sample.job_id,
                    'filename': sample.original_filename,
                    'status': sample.inference_status,
                    'display_status': sample.display_status if hasattr(sample, 'display_status') else sample.inference_status,
                    'uploader_name': sample.uploader.name if sample.uploader else None,
                    'uploader_email': sample.uploader.email if sample.uploader else None,
                    'created_at': sample.created_at.isoformat() if sample.created_at else None,
                    'updated_at': sample.updated_at.isoformat() if sample.updated_at else None,
                    'image_download_url': sample.image_download_url if hasattr(sample, 'image_download_url') else None
                }
                # Add queue status for pending samples
                if sample.inference_status == 'pending':
                    sample_dict['in_queue'] = queue_status.get(sample.job_id, False)
                return sample_dict

            return jsonify({
                'processing_info': processing_info,
                'samples': {
                    'pending': [sample_to_dict(s) for s in pending_samples],
                    'processing': [sample_to_dict(s) for s in processing_samples],
                    'completed': [sample_to_dict(s) for s in completed_samples]
                }
            })

        except Exception as e:
            print(f"Error in status API: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @expose('/queue-sample', methods=['POST'])
    @login_required
    def queue_sample(self):
        """Queue a single sample to SQS"""
        if not (current_user.is_authenticated and current_user.role == UserRole.ADMIN):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

        # Handle both form data and JSON
        if request.headers.get('Content-Type') == 'application/json':
            sample_id = request.get_json().get('sample_id')
        else:
            sample_id = request.form.get('sample_id')
            
        if not sample_id:
            return jsonify({'status': 'error', 'message': 'No sample ID provided'}), 400

        # Find sample
        sample = db.session.query(Sample).filter_by(job_id=sample_id).first()
        if not sample:
            return jsonify({'status': 'error', 'message': f'Sample {sample_id} not found'}), 404

        # Enqueue to SQS
        from utilities.enqueue_utility import enqueue
        import os
        
        payload = {
            "jobId": sample.job_id,
            "s3ObjectKey": sample.s3_object_key,
            "s3BucketName": os.environ.get("S3_BUCKET_NAME", "my-test-bucket"),
            "status": "pending"
        }
        
        try:
            enqueue_result = enqueue(payload, check_duplicates=True)
            if enqueue_result.get('already_queued'):
                return jsonify({
                    'status': 'info',
                    'message': f'Sample {sample.job_id[:8]}... is already in queue',
                    'already_queued': True
                })
            elif enqueue_result.get('enqueued'):
                return jsonify({
                    'status': 'success',
                    'message': f'Sample {sample.job_id[:8]}... queued successfully'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to queue sample'
                }), 500
        except Exception as e:
            print(f"Error queuing sample: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Error queuing sample: {str(e)}'
            }), 500
    
    @expose('/queue-all-pending', methods=['POST'])
    @login_required
    def queue_all_pending(self):
        """Queue all pending samples to SQS"""
        if not (current_user.is_authenticated and current_user.role == UserRole.ADMIN):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

        # Get all pending samples
        pending_samples = db.session.query(Sample).filter_by(inference_status='pending').all()
        
        if not pending_samples:
            return jsonify({
                'status': 'success',
                'message': 'No pending samples to queue',
                'queued_count': 0
            })

        # Enqueue each sample with duplicate checking
        from utilities.enqueue_utility import enqueue
        import os
        
        S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "my-test-bucket")
        queued_count = 0
        skipped_count = 0
        failed_count = 0
        
        for sample in pending_samples:
            payload = {
                "jobId": sample.job_id,
                "s3ObjectKey": sample.s3_object_key,
                "s3BucketName": S3_BUCKET_NAME,
                "status": "pending"
            }
            
            try:
                enqueue_result = enqueue(payload, check_duplicates=True)
                if enqueue_result.get('already_queued'):
                    skipped_count += 1
                elif enqueue_result.get('enqueued'):
                    queued_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Error queuing sample {sample.job_id}: {e}")
                failed_count += 1
        
        message = f'Queued {queued_count} samples'
        if skipped_count > 0:
            message += f', {skipped_count} already in queue'
        if failed_count > 0:
            message += f', {failed_count} failed'
            
        return jsonify({
            'status': 'success' if failed_count == 0 else 'partial',
            'message': message,
            'queued_count': queued_count,
            'skipped_count': skipped_count,
            'failed_count': failed_count
        })
    
    @expose('/clear-queue', methods=['POST'])
    @login_required
    def clear_queue_endpoint(self):
        """Clear all messages from the SQS queue"""
        if not (current_user.is_authenticated and current_user.role == UserRole.ADMIN):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

        from utilities.enqueue_utility import clear_queue
        
        try:
            result = clear_queue()
            
            if result.get('cleared'):
                return jsonify({
                    'status': 'success',
                    'message': 'Queue cleared successfully'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': result.get('details', 'Failed to clear queue')
                }), 500
                
        except Exception as e:
            print(f"Error clearing queue: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @expose('/check-queue-status', methods=['POST'])
    @login_required
    def check_queue_status(self):
        """Check which samples are currently in the SQS queue"""
        if not (current_user.is_authenticated and current_user.role == UserRole.ADMIN):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

        from utilities.enqueue_utility import is_job_in_queue
        
        data = request.get_json() or {}
        sample_ids = data.get('sample_ids', [])
        
        if not sample_ids:
            return jsonify({'status': 'error', 'message': 'No sample IDs provided'}), 400
        
        # Check each sample
        queue_status = {}
        for sample_id in sample_ids:
            try:
                queue_status[sample_id] = is_job_in_queue(sample_id)
            except Exception as e:
                print(f"Error checking queue status for {sample_id}: {e}")
                queue_status[sample_id] = False
        
        return jsonify({
            'status': 'success',
            'queue_status': queue_status
        })
    
