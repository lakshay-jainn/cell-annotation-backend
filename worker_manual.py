# worker_manual.py - Manual processing utilities
"""
Model processing utilities for manual admin control.
This replaces the automated SQS polling with manual trigger functions.
"""

import os
import logging
import requests
import boto3
import tempfile
from datetime import datetime
from botocore.config import Config
from db.models import Sample
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker_manual")

# Environment variables
REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT = os.environ.get("AWS_ENDPOINT_URL")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
MODEL_API_URL = os.environ.get("MODEL_API_URL")
HF_API_KEY = os.environ.get("HF_API_KEY")

boto_conf = Config(retries={"max_attempts": 4, "mode": "standard"})

# Initialize AWS clients
if ENDPOINT:
    s3 = boto3.client("s3", region_name=REGION, endpoint_url=ENDPOINT, config=boto_conf)
else:
    s3 = boto3.client("s3", region_name=REGION, config=boto_conf)

# Database setup
SQL_URI = os.getenv("SQL_URI")
engine = create_engine(
    SQL_URI,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def download_from_s3(bucket, key, local_path):
    """Download file from S3 to local path."""
    log.info("Downloading file %s/%s to %s", bucket, key, local_path)
    try:
        s3.download_file(bucket, key, local_path)
        log.info("Download successful.")
    except Exception as e:
        log.error("Failed to download file from S3. Error: %s", e)
        raise

def upload_to_s3(bucket, local_path, key):
    """Upload file from local path to S3."""
    log.info("Uploading file %s to %s/%s", local_path, bucket, key)
    try:
        s3.upload_file(local_path, bucket, key)
        log.info("Upload successful.")
    except Exception as e:
        log.error("Failed to upload file to S3. Error: %s", e)
        raise

def call_model_api(local_image_path):
    """
    Call the model API with an image file.
    Returns the CSV content.
    """
    log.info("Sending image to model API at %s", MODEL_API_URL)
    
    if not MODEL_API_URL:
        raise ValueError("MODEL_API_URL environment variable not set")
    
    try:
        with open(local_image_path, 'rb') as img_file:
            files = {"image": img_file}
            headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
            
            response = requests.post(MODEL_API_URL, files=files, headers=headers, timeout=180)
            response.raise_for_status()
            
            log.info("Model API call successful. Received %d bytes.", len(response.content))
            return response.content
            
    except requests.exceptions.RequestException as e:
        log.error("Model API call failed. Error: %s", e)
        raise

def process_sample_manually(sample_id):
    """
    Process a single sample manually (called from admin interface).
    
    Args:
        sample_id (str): Job ID (primary key) of the sample to process
        
    Returns:
        dict: Processing result with success status and details
    """
    temp_image_path = None
    temp_csv_path = None
    
    try:
        # Get sample from database using SessionLocal
        with SessionLocal() as session:
            # Use filter_by since job_id is the primary key but session.get expects the actual column name
            sample = session.query(Sample).filter_by(job_id=sample_id).first()
            if not sample:
                return {"success": False, "error": f"Sample {sample_id} not found"}
            
            if not sample.s3_object_key:
                return {"success": False, "error": f"Sample {sample.job_id} has no S3 object key"}
            
            log.info(f"Starting manual processing of sample {sample.job_id}")
            
            # Update status to processing (only if not already processing)
            if sample.inference_status != 'processing':
                sample.inference_status = 'processing'
                sample.updated_at = datetime.utcnow()
                session.commit()
                log.info(f"Updated sample {sample.job_id} status to 'processing'")
            else:
                log.info(f"Sample {sample.job_id} already in 'processing' status")
            
            # Get the S3 key and job directory
            s3_key = sample.s3_object_key
            job_dir = os.path.dirname(s3_key)
            job_id = sample.job_id
            
            # Download image from S3 to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                temp_image_path = temp_img.name
            
            download_from_s3(S3_BUCKET_NAME, s3_key, temp_image_path)
            log.info(f"Downloaded image from S3: {s3_key}")
            
            # Call model API
            csv_content = call_model_api(temp_image_path)
            
            # Save CSV content to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
                temp_csv_path = temp_csv.name
                temp_csv.write(csv_content)
            
            # Upload CSV result to S3
            csv_key = f"{job_dir}/results.csv"
            upload_to_s3(S3_BUCKET_NAME, temp_csv_path, csv_key)
            
            # Update sample with success 
            sample.inference_status = 'completed'
            sample.s3_inference_key = csv_key
            sample.updated_at = datetime.utcnow()
            session.commit()
            
            # Delete the SQS message for this completed job
            try:
                from sqs_processor import delete_message_for_job_id
                delete_message_for_job_id(sample.job_id)
            except Exception as e:
                log.warning(f"Failed to delete SQS message for completed job {sample.job_id}: {e}")
            
            log.info(f"Successfully processed sample {sample.job_id}")
            return {
                "success": True, 
                "message": f"Sample {sample.job_id} processed successfully",
                "inference_key": csv_key
            }
            
    except Exception as e:
        log.error(f"Error in manual processing of sample {sample_id}: {e}")
        
        # Update sample status to failed (False = failed/incomplete)
        try:
            with SessionLocal() as session:
                sample = session.get(Sample, sample_id)
                if sample:
                    sample.inference_status = 'failed'
                    sample.updated_at = datetime.utcnow()
                    session.commit()
                    
                    # Delete the SQS message for this failed job to prevent reprocessing
                    try:
                        from sqs_processor import delete_message_for_job_id
                        delete_message_for_job_id(sample.job_id)
                        log.info(f"Deleted SQS message for failed job {sample.job_id}")
                    except Exception as sqs_error:
                        log.warning(f"Failed to delete SQS message for failed job {sample.job_id}: {sqs_error}")
        except Exception as db_error:
            log.error(f"Failed to update sample status to failed: {db_error}")
        
        return {"success": False, "error": str(e)}
        
    finally:
        # Clean up temporary files
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            log.info("Deleted temporary image file.")
        if temp_csv_path and os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            log.info("Deleted temporary CSV file.")

def process_all_pending_samples():
    """
    Process all pending samples manually.
    
    Returns:
        dict: Processing results summary
    """
    try:
        log.info("Starting process_all_pending_samples function")
        with SessionLocal() as session:
            pending_samples = session.query(Sample).filter(
                Sample.inference_status.in_(['pending', 'failed'])  # Process both pending and failed
            ).all()
            
            log.info(f"Found {len(pending_samples)} pending/failed samples")
            
            if not pending_samples:
                log.info("No pending samples to process")
                return {"success": True, "message": "No pending samples to process", "processed": 0, "failed": 0}
            
            log.info(f"Starting batch processing of {len(pending_samples)} samples")
            
            success_count = 0
            error_count = 0
            errors = []
            
            for sample in pending_samples:
                log.info(f"Processing sample {sample.job_id}")
                result = process_sample_manually(sample.job_id)
                log.info(f"Result for sample {sample.job_id}: {result}")
                if result["success"]:
                    success_count += 1
                else:
                    error_count += 1
                    errors.append(f"Sample {sample.job_id}: {result['error']}")
            
            log.info(f"Batch processing completed. Success: {success_count}, Errors: {error_count}")
            
            return {
                "success": True,
                "processed": success_count,
                "failed": error_count,
                "errors": errors[:10] if errors else []  # Limit to first 10 errors
            }
            
    except Exception as e:
        log.error(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def get_processing_status():
    """
    Get current processing status summary.
    
    Returns:
        dict: Status summary with counts
    """
    try:
        with SessionLocal() as session:
            pending_count = session.query(Sample).filter(
                Sample.inference_status == 'pending'
            ).count()
            
            processing_count = session.query(Sample).filter(
                Sample.inference_status == 'processing'
            ).count()
            
            completed_count = session.query(Sample).filter(
                Sample.inference_status == 'completed'
            ).count()
            
            failed_count = session.query(Sample).filter(
                Sample.inference_status == 'failed'
            ).count()
            
            return {
                "success": True,
                "pending": pending_count,
                "processing": processing_count,
                "completed": completed_count,
                "failed": failed_count,
                "total": pending_count + processing_count + completed_count + failed_count
            }
            
    except Exception as e:
        log.error(f"Error getting processing status: {e}")
        return {"success": False, "error": str(e)}

# Legacy function kept for compatibility - now does nothing
def poll_loop():
    """
    Legacy function - automatic polling has been disabled.
    Use process_sample_manually() or process_all_pending_samples() instead.
    """
    log.warning("Automatic polling has been disabled. Use manual processing through admin interface.")
    return

if __name__ == "__main__":
    # If run directly, show status instead of starting poll loop
    print("Worker converted to manual mode - automatic polling disabled")
    
    if not S3_BUCKET_NAME or not MODEL_API_URL:
        print("Error: S3_BUCKET_NAME or MODEL_API_URL environment variable is not set.")
        exit(1)
    
    status = get_processing_status()
    if status["success"]:
        print(f"Current status: {status['pending']} pending, {status['processing']} processing, {status['completed']} completed")
    else:
        print(f"Error getting status: {status['error']}")